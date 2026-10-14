"""Experimental VLM-supervised whole-task integration, not certified verification.
VLM task matching gates startup. A repeated observed visual loss requests at most
one measured HOLD / clear / fresh observation. This intervention is explicitly
unvalidated; it is never a verified failure, success or memory record.
"""
import time, json, hashlib
from dataclasses import asdict
from pathlib import Path
from harness.robotwin_system_adapter import RobotwinSystemAdapter, reject_gt
from harness.robotwin_policy_worker import PolicyClient,envelope,encode_observation
from harness.core import Memory, Scope, context_fingerprint
from harness.runtime import Observation
from harness.robotwin_task_supervisor import TaskSupervisor, ModelContract, ProviderBinding, CapabilityCandidate

class IntegrationBlocked(RuntimeError):pass

class IntegratedSystem(RobotwinSystemAdapter):
    def __init__(self,client,driver,*,visual,robot,refresh,out,episode):
        super().__init__(client,driver,max_age_s=30)
        self.visual=visual;self.robot=robot;self.refresh=refresh;self.out=Path(out)
        self.memory=Memory(self.out/'experience.sqlite',split_id='dev_integrated',run_id=episode)
        self.memory_written=False;self.full_system=False
        self.recoveries=0;self.last_visible=None;self.positive_description=None;self.loss_votes=0
        self.events=[]
        self.last_visual_step=-100
        self.last_feedback_commands=None
        scope=Scope('put_object_cabinet','put_object_cabinet',context_fingerprint({'episode':episode,'condition':'zero_single_task'}),'aloha_native14','native_TOPP','cabinet_30k','qwen7b_nf4','dev_integrated',episode)
        contract=ModelContract('put_object_cabinet',scope.wam_version,scope.robot_version,scope.controller_version)
        grounder=ProviderBinding('qwen7b','experimental_v1',scope.key(),lambda frame: self.last_visible['replies'].get('ground',{}))
        verifier=ProviderBinding('qwen7b_visible','atomic_v1',scope.key(),lambda frame: self.last_visible['replies'])
        self.strict=TaskSupervisor(contract=contract,scope=scope,required_predicates=('object_grasp_retained','drawer_open_held','object_fully_contained','object_released','placement_stable'),max_age_s=30,ask_timeout_s=30,grounder=grounder,verifier=verifier,memory=self.memory)
        self.strict.begin_episode(episode,CapabilityCandidate('put_object_cabinet','unbound_target'))
        self.report={'configuration':'integrated_experimental','certified_full_system':False,'system_verified_success':None,'memory_records':0,'recovery_attempts':0,'visual_queries':0,'intervention_basis':'unvalidated_visual_transition_not_physical_verification'}

    def event(self,event):
        self.events.append(event)
        with (self.out/'integrated_events.jsonl').open('a') as f:
            f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')

    def inspect(self,episode,step,observation):
        reject_gt(observation)
        reply=self.visual.request(envelope('inspect',episode,step,observation=encode_observation(observation)),'visual')
        self.report['visual_queries']+=1
        results=reply['replies']
        if any(r.get('outcome')=='runtime_error' for r in results.values()):raise IntegrationBlocked('visual_backend_runtime_error')
        if step==0:
            ground=results.get('ground',{}).get('parsed')
            if not ground or ground.get('decision')!='execute' or ground.get('capability')!='put_object_cabinet':
                self.event({'step':step,'kind':'grounding_block','reply':reply});raise IntegrationBlocked('grounding_unsupported_ask_or_unknown')
        held=results['object_between_gripper_fingers'].get('parsed') or {}
        desc=held.get('object_description','').lower().strip()
        state=held.get('state','unknown')
        # Exact visible description continuity is conservative, but not a tracker.
        # Never turn unknown, parse failure, or a different subject into a loss.
        if state=='true':self.positive_description=desc;self.loss_votes=0
        elif state=='false' and desc and desc==self.positive_description:self.loss_votes+=1
        else:self.loss_votes=0
        trigger=self.loss_votes>=2 and self.recoveries<1
        self.last_visible=reply
        self.strict.observe(Observation(episode+':'+str(step),time.monotonic(),{'rgb':observation['rgb']}),visible_objects=())
        strict_decision=self.strict.ground()
        self.report['strict_supervisor_last_decision']=asdict(strict_decision)
        self.event({'step':step,'kind':'visual_observation','reply':reply,'experimental_recovery_requested':trigger,'subject_tracking_verified':False,'strict_supervisor':asdict(strict_decision)})
        return trigger

    def prepare(self,episode,step,observation_id,observed_at,observation):
        from harness.robotwin_robot_feedback import robot_feedback
        feedback=robot_feedback(self.robot())
        commands=[feedback['grippers'][s]['commanded_aperture'] for s in ('left','right')]
        entering_close=self.last_feedback_commands is not None and any(old>=.9 and cur<.9 for old,cur in zip(self.last_feedback_commands,commands))
        in_transition=any(.1<cur<.9 for cur in commands)
        self.last_feedback_commands=commands
        self.event({'step':step,'kind':'robot_measurements','observation_id':observation_id,'feedback':feedback,'target_or_scene_GT_used':False})
        # Inspect during approach and commanded closure, rather than waiting
        # 64 actions. Still advisory, not a calibrated contact/recovery permit.
        if step in (0,16,32) or step-self.last_visual_step>=64 or (entering_close or in_transition) and step-self.last_visual_step>=8:
            trigger=self.inspect(episode,step,observation)
            self.last_visual_step=step
            if trigger:
                from harness.robotwin_robot_feedback import hold_measured
                hold=hold_measured(self.robot())
                clear=self.replan('experimental_repeated_visible_loss')
                observation,observed_at=self.refresh()
                self.recoveries+=1;self.report['recovery_attempts']=self.recoveries
                self.event({'step':step,'kind':'experimental_hold_clear_reobserve','hold':hold,'clear':clear,'history_reset':False,'verified_failure':None})
        # Captured-before-VLM timestamp remains unchanged unless genuinely refreshed.
        return super().prepare(episode,step,observation_id,observed_at,observation)

    def close(self):self.memory.connection.close()
