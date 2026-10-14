"""Choose action prefix length from commanded gripper transitions only.
This is a feedback schedule, not grasp detection or a safety/completion verdict.
No target actors, native predicates or model-invented task stages are inputs.
"""
import numpy as np

def execution_prefix(chunk,commanded_state,mode):
    if mode=='native8':return 8,'native_exec8'
    if mode!='contact_refresh':raise ValueError('unsupported_execution_mode')
    a=np.asarray(chunk,dtype=float);s=np.asarray(commanded_state,dtype=float)
    if a.shape!=(32,14) or s.shape!=(14,) or not np.isfinite(a).all() or not np.isfinite(s).all():raise ValueError('native14_finite_inputs_required')
    current=s[[6,13]];future=a[:8,[6,13]]
    transition=((current>.1)&(current<.9)).any()
    entering=(((current>=.9)[None,:])&(future<.9)).any() or (((current<=.1)[None,:])&(future>.1)).any()
    if transition or entering:return 1,'commanded_gripper_transition_refresh_not_grasp_verified'
    return 8,'native_exec8_outside_gripper_transition'
