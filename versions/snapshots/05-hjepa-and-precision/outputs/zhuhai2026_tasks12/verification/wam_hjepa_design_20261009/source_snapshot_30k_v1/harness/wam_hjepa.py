"""Opt-in H-JEPA-inspired WAM auxiliaries; the frozen WAM implementation is unchanged.

This is multi-horizon supervision, NOT a hierarchical JEPA or a causal prefix
world model. Future images exist only in the auxiliary training loss. Online
sampling delegates to the unchanged observation-only WAM sampler.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from harness.wam_v3 import bf16_from_bits, WamV3Wrapper
from harness.wam_behavior import load_upstream

HJEPA_VARIANTS = ('baseline', 'multiscale', 'idm', 'combined')
ARM_INDICES = tuple(range(7)) + tuple(range(8, 15))
GRIPPER_INDICES = (7, 15)
IDM_STEPS = 4


def variant_settings(variant):
    if variant not in HJEPA_VARIANTS:
        raise ValueError('unknown H-JEPA auxiliary variant')
    multi, inverse = variant in ('multiscale', 'combined'), variant in ('idm', 'combined')
    return dict(variant=variant, multi_horizon=multi, use_idm=inverse,
                future32_weight=.25 if multi else .5, future8_weight=.25 if multi else 0.,
                idm_weight=.1 if inverse else 0., future_horizons=[8,32] if multi else [32],
                idm_horizon=4, control_hz=10., idm_anchor='fixed_current_measured_q_t',
                label='(target_command[t+k+1]-measured_state[t])/TRAIN_residual_scale',
                future_target_layer=-4, future_target_stop_gradient=True,
                idm_visual_frames=[0,1,2,3,4], causal_prefix_claim=False)


@contextmanager
def isolated_aux_rng(device):
    # An added decoder/head must not change later flow-noise or sampler draws.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


class HjepaAuxHeads(nn.Module):
    """All added checkpoint keys live under action_model.hjepa_aux.*."""
    def __init__(self, hidden_dim, num_views, idm_scale, *, idm_hidden=512, step_embedding_dim=16):
        super().__init__()
        self.hidden_dim, self.num_views = int(hidden_dim), int(num_views)
        self.summary_dim = self.hidden_dim * self.num_views
        scale = torch.as_tensor(idm_scale, dtype=torch.float32).detach().clone()
        if scale.shape != (16,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('IDM needs sixteen finite positive TRAIN-only residual scales')
        self.register_buffer('idm_scale', scale)
        self.future8_offset = nn.Parameter(torch.zeros(self.hidden_dim))
        self.step_embedding = nn.Embedding(IDM_STEPS, step_embedding_dim)
        self.idm_head = nn.Sequential(
            nn.Linear(3*self.summary_dim + 16 + step_embedding_dim, idm_hidden),
            nn.GELU(), nn.LayerNorm(idm_hidden), nn.Linear(idm_hidden, 16))

    def inverse_actions(self, visual_summaries, current_q_normalized):
        """Five independently adapted frames, never mixed DiT tokens or future q."""
        b = visual_summaries.shape[0]
        if visual_summaries.shape != (b,5,self.summary_dim) or current_q_normalized.shape != (b,16):
            raise ValueError('IDM expects five pre-action summaries and only current native16 state')
        e = self.step_embedding(torch.arange(IDM_STEPS,device=visual_summaries.device)).unsqueeze(0).expand(b,-1,-1)
        x = torch.cat([visual_summaries[:,:1].expand(-1,IDM_STEPS,-1),
                       visual_summaries[:,:-1], visual_summaries[:,1:],
                       current_q_normalized[:,None,:].expand(-1,IDM_STEPS,-1), e],dim=-1)
        return self.idm_head(x)


def future_with_offset(decoder, cond_tokens, offset):
    """Exact original decoder recipe, except the 8-step query offset."""
    if offset.shape != (decoder.query_embed.shape[-1],):
        raise ValueError('horizon embedding dimension mismatch')
    memory = decoder.kv_norm(cond_tokens)
    queries = decoder.query_embed.expand(cond_tokens.shape[0],-1,-1) + offset[None,None,:]
    return decoder.out_proj(decoder.out_norm(decoder.decoder(queries,memory)))


def _view_tokens(model, adapted, batch_size):
    v = model.num_views
    if adapted.ndim != 3 or adapted.shape[0] != batch_size*v or adapted.shape[-1] != model.hidden_dim:
        raise ValueError('adapter output differs from the pre-action visual contract')
    return (adapted.reshape(batch_size,v,adapted.shape[1],model.hidden_dim)
            + model.type_emb_dino_layers[0].unsqueeze(1)
            + model.view_emb[:v].unsqueeze(0).squeeze(2))


def adapt_independent_frames(model, features):
    """B x time x view x layer x token x dim -> B x time x (view*hidden).

    No time mixing, BatchNorm or policy/DiT call is allowed on this path.
    Frozen encoder values detach; shared adapters remain differentiable.
    """
    if features.ndim != 6 or features.shape[2] != model.num_views or features.shape[3] != model.num_dino_layers:
        raise ValueError('future features must be (B,time,view,layer,token,dim)')
    b,t,v,l,n,d = features.shape
    x = features.detach()
    per_layer = [x[:,:,:,i].reshape(b*t*v,n,d) for i in range(l)]
    adapted = model.dino_adapters[0](model.concat_fusion(per_layer))
    return _view_tokens(model,adapted,b*t).mean(dim=2).reshape(b,t,-1)


def inverse_loss(prediction, raw_targets, scale):
    """FP32 targets/loss. Mean over steps, not sum; arms/grippers each get 1/2."""
    if prediction.ndim != 3 or prediction.shape[1:] != (4,16) or prediction.shape != raw_targets.shape:
        raise ValueError('IDM prediction/label must both be B x 4 x 16')
    target = raw_targets.detach().float() / scale.detach().float()[None,None,:]
    if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError('nonfinite valid IDM prediction or label')
    error = F.smooth_l1_loss(prediction.float(),target,reduction='none')
    return .5*error[:,:,ARM_INDICES].mean() + .5*error[:,:,GRIPPER_INDICES].mean()


class HjepaWrapper(nn.Module):
    """Drop-in cached-feature loss/sampling facade around an unmodified WamV3Wrapper."""
    def __init__(self, base, variant):
        super().__init__()
        self.base, self.settings = base, variant_settings(variant)

    def __getattr__(self,name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('base'),name)

    @property
    def action_model(self):
        return self.base.action_model

    def _mask(self,batch,key,b):
        if key not in batch:
            raise ValueError('required auxiliary validity mask missing: '+key)
        return WamV3Wrapper._valid_mask(batch[key],(b,),self.device,key)

    def _future8(self,batch,out):
        b = out['cond_tokens'].shape[0]
        valid = self._mask(batch,'future_valid_mask_8',b)
        aux,dec = self.action_model.hjepa_aux,self.action_model.future_feat_decoder
        if not valid.any():
            return out['cond_tokens'].float().sum()*0 + aux.future8_offset.sum()*0,0
        target = bf16_from_bits(batch['future_feats_8'].to(self.device,non_blocking=True)).detach()
        if target.ndim != 4 or target.shape[:2] != (b,len(self.views)):
            raise ValueError('future_feats_8 must be B x V x P x D')
        target = target.reshape(b,-1,target.shape[-1])[valid].float()
        if not torch.isfinite(target).all():
            raise ValueError('nonfinite valid future8 target')
        with isolated_aux_rng(self.device), torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            pred = future_with_offset(dec,out['cond_tokens'][valid],aux.future8_offset)
        if pred.shape != target.shape:
            raise ValueError('future8 target/decoder shape mismatch')
        vla,_ = load_upstream()
        return vla.future_feature_cosine_loss(pred.float(),target),int(valid.sum())

    def _inverse(self,batch,current_adapted):
        b = batch['state'].shape[0]
        valid = self._mask(batch,'idm_valid_mask',b)
        aux = self.action_model.hjepa_aux
        if not valid.any():
            return current_adapted.float().sum()*0,0,None
        future = bf16_from_bits(batch['idm_future_feats'].to(self.device,non_blocking=True))
        if future.ndim != 6 or future.shape[:2] != (b,4):
            raise ValueError('IDM needs exactly four future visual frames')
        future = future[valid].detach()
        if not torch.isfinite(future).all():
            raise ValueError('nonfinite valid future IDM features')
        if self.device.type=='cpu':
            future = future.float()
        raw = batch['idm_target_raw'].to(self.device,dtype=torch.float32)
        if raw.shape != (b,4,16):
            raise ValueError('IDM raw labels must be B x 4 x 16')
        # Current tensor was captured BEFORE action mixing. Do not use out/cond_tokens here.
        current = _view_tokens(self.action_model,current_adapted,b).mean(dim=2).reshape(b,-1)[valid]
        q = self.base.normalize_state(batch['state'].to(self.device,dtype=torch.float32))[:,0][valid].detach()
        with isolated_aux_rng(self.device), torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            future_summaries = adapt_independent_frames(self.action_model,future)
            sequence = torch.cat([current[:,None,:],future_summaries],dim=1)
            pred = aux.inverse_actions(sequence,q)
        return inverse_loss(pred,raw[valid],aux.idm_scale),int(valid.sum()),pred

    def forward(self,batch,lambdas=None,action_valid_mask=None,future_valid_mask=None):
        # Variants define the objective; silently accepting old .5 here would
        # corrupt the intended .25/.25 comparison.
        expected = {'future':self.settings['future32_weight'],'future8':self.settings['future8_weight'],
                    'idm':self.settings['idm_weight'],'gate_kl':0.,'phase_ce':0.,'log_dist':0.}
        if lambdas is not None:
            if any(k not in expected or float(v)!=expected[k] for k,v in lambdas.items()):
                raise ValueError('loss weights differ from the registered auxiliary variant')
        base_batch = dict(batch)
        # Cached/frozen teacher values are never trainable targets, even when a
        # caller supplies floating tensors rather than bf16 cache bits.
        for key in ('feats','future_feats'):
            if isinstance(base_batch.get(key),torch.Tensor):
                base_batch[key] = base_batch[key].detach()
        captured = []
        handle = None
        if self.settings['use_idm']:
            handle = self.action_model.dino_adapters[0].register_forward_hook(lambda m,i,o: captured.append(o))
        try:
            total,info,out = self.base(base_batch,{'future':expected['future'],'gate_kl':0.,'phase_ce':0.,'log_dist':0.},
                                      action_valid_mask=action_valid_mask,future_valid_mask=future_valid_mask)
        finally:
            if handle is not None:
                handle.remove()
        out = dict(out)
        info = dict(info,future32=info.get('future',0.),hjepa_variant=self.settings['variant'])
        losses = {}
        if self.settings['multi_horizon']:
            loss,count = self._future8(batch,out)
            total = total + expected['future8']*loss
            losses['future8'] = loss
            info.update(future8=float(loss.detach()),future8_valid_samples=count)
        if self.settings['use_idm']:
            if len(captured)!=1:
                raise RuntimeError('expected exactly one current pre-action adapter evaluation')
            loss,count,pred = self._inverse(batch,captured[0])
            total = total + expected['idm']*loss
            losses['idm'] = loss
            info.update(idm=float(loss.detach()),idm_valid_samples=count)
            out['idm_pred'] = pred
        out['aux_losses'] = losses
        info.update(future32_weight=expected['future'],future8_weight=expected['future8'],idm_weight=expected['idm'])
        return total,info,out


def upgrade_wrapper(base_wrapper, *, variant, idm_scale, idm_hidden=512, step_embedding_dim=16, aux_init_seed=20261009):
    """Install after loading/migrating the base model. Never mutate a frozen checkpoint.

    Initialization is isolated from the base model's RNG; all four arms get the
    same auxiliary state keys. Loading a new checkpoint: strip hjepa_aux.* for
    the base load, install here, then strict-load the full action_model state.
    """
    settings = variant_settings(variant)
    model = base_wrapper.action_model
    if base_wrapper.action_dim!=16 or base_wrapper.state_dim!=16 or model.action_len!=32 or model.num_views!=3:
        raise ValueError('H-JEPA variant is explicitly native16 / chunk32 / three-view')
    if not model.use_future_feat or model.fusion_mode!='concat' or len(model.dino_adapters)!=1:
        raise ValueError('requires the original shared-adapter WAM with future decoder')
    if hasattr(model,'hjepa_aux'):
        raise ValueError('auxiliary heads already installed; do not silently replace trained weights')
    if any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('cross-sample BatchNorm could leak auxiliary future statistics')
    with torch.random.fork_rng(devices=[],enabled=True), torch.device('cpu'):
        # CPU-only initialization must not reseed CUDA/MPS generators.
        torch.random.default_generator.manual_seed(int(aux_init_seed))
        aux = HjepaAuxHeads(model.hidden_dim,model.num_views,idm_scale,idm_hidden=idm_hidden,step_embedding_dim=step_embedding_dim)
    aux.future8_offset.requires_grad_(settings['multi_horizon'])
    aux.idm_head.requires_grad_(settings['use_idm'])
    aux.step_embedding.requires_grad_(settings['use_idm'])
    model.add_module('hjepa_aux',aux.to(base_wrapper.device))
    return HjepaWrapper(base_wrapper,variant)


def gradient_evidence(wrapper):
    """After backward, report actual auxiliary/shared gradients without claims of task success."""
    def norm(parameters):
        values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else 0.
    m=wrapper.action_model
    encoder=getattr(wrapper.base,'vision_encoder',None)
    return dict(shared_visual_grad_norm=norm(list(m.concat_fusion.parameters())+list(m.dino_adapters.parameters())),
                idm_head_grad_norm=norm(list(m.hjepa_aux.idm_head.parameters())+list(m.hjepa_aux.step_embedding.parameters())),
                future8_offset_grad_norm=norm([m.hjepa_aux.future8_offset]),
                dino_has_grad=False if encoder is None else any(p.grad is not None for p in encoder.parameters()),
                idm_scale_requires_grad=m.hjepa_aux.idm_scale.requires_grad)
