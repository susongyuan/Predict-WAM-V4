"""Residual task-condition projection for a frozen native14 whole-task WAM.

Zero initialization preserves the original zero-condition token. Nonzero VTTs
may affect actions only after explicitly trained adapter weights are loaded.
The adapter neither selects targets nor verifies physical task completion.
"""
import torch
from torch import nn
class ResidualVTTProjection(nn.Module):
    def __init__(self,base,hidden=768):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.normalizer=nn.LayerNorm(1024,elementwise_affine=False)
        self.residual=nn.Linear(1024,hidden,bias=False);nn.init.zeros_(self.residual.weight)
        self.enabled=True
    def forward(self,condition):
        token=self.base(torch.zeros_like(condition))
        if self.enabled:token=token+self.residual(self.normalizer(condition))
        return token
def install(action_model,state=None):
    p=ResidualVTTProjection(action_model.task_cond_proj)
    if state is not None:p.residual.load_state_dict(state,strict=True)
    action_model.task_cond_proj=p
    return p
