"""Online 7B NF4 worker for an explicitly experimental integrated run.
No GT inputs. All visual answers remain advisory, never verified outcomes.
"""
import argparse, base64, contextlib, socket, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import robotwin_vlm_worker as v
from harness.robotwin_atomic_visual import AtomicVisualVerifier
from harness.robotwin_policy_worker import Channel, envelope, decode_observation

class Backend(v.GpuBackend):
    def __init__(self, owner):
        v.check_gpu_owner(owner);v.configure_runtime()
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
        self.owner=owner;self.torch=torch;torch.set_num_threads(2);torch.set_num_interop_threads(1)
        self.guard()
        model=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model),device_map={'':'cuda:0'},dtype=torch.bfloat16,quantization_config=quant,local_files_only=True,trust_remote_code=False).eval()
        self.processor=AutoProcessor.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        if not self.model.is_loaded_in_4bit:raise RuntimeError('quantization_not_loaded')
        self.metadata={'id':'Qwen/Qwen2.5-VL-7B-Instruct','revision':model.name,'quantization':'NF4_double_quant_BF16','status':v.STATUS}
        self.guard()

def frame(episode,step,obs):
    rgb={}
    for src,dst in zip(('head_camera','left_camera','right_camera'),v.VIEWS):
        a=obs['rgb'][src]
        if a.shape!=v.SHAPE:raise ValueError('native_RGB_shape_mismatch')
        rgb[dst]={'encoding':'rgb8_base64','shape':list(a.shape),'data':base64.b64encode(a.tobytes()).decode()}
    return {'episode_id':episode,'observation_id':step,'rgb':rgb}

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-lock-owner',required=True);p.add_argument('--ipc-fd',required=True,type=int);args=p.parse_args()
    ch=Channel(socket.socket(fileno=args.ipc_fd))
    with contextlib.redirect_stdout(sys.stderr):backend=Backend(args.gpu_lock_owner)
    service=v.Service(backend,backend.guard);atomic=AtomicVisualVerifier(backend)
    ch.send(envelope('ready','startup',0,model=backend.metadata,resources=backend.resources()),30)
    while True:
        msg=ch.recv(1800);episode,step=msg['episode'],msg['step'];obs=decode_observation(msg['observation']);f=frame(episode,step,obs)
        replies={}
        if step==0:
            replies['ground']=service.handle({'version':1,'id':episode+':ground','operation':'ground','current':f,'instruction':'Move the object into the cabinet drawer using the trained put_object_cabinet capability.'})
        for pred in ('drawer_open','object_between_gripper_fingers','object_visibly_inside_open_drawer'):
            replies[pred]=atomic.query({'version':1,'id':episode+':'+str(step)+':'+pred,'predicate':pred,'current':f})
        ch.send(envelope('visual',episode,step,replies=replies),60)
        if any(r.get('outcome')=='runtime_error' for r in replies.values()):return 2
if __name__=='__main__':raise SystemExit(main())
