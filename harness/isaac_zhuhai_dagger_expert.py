"""Versioned actual-state recovery expert; retains strict scene scoring."""
import numpy as np
from harness import isaac_zhuhai_expert as original
from harness.isaac_zhuhai_task2_expert_repair_v3 import SmoothExpert as Repaired
from harness.zhuhai_robot_spec import HOME_ACTION


class RecoveryExpert(Repaired):
    def __init__(self,env,limiter):
        self.limiter=limiter
        super().__init__(env)

    def _drive(self,action):
        accepted,stats=self.limiter.step(action)
        self.env.expert_limit_stats=stats
        return super()._drive(accepted)

    def _task2_once(self):
        # Brake from the actual inherited command velocity, never reset a moving
        # controller or restore an object pose/episode state.
        self.hold(1.,'recovery:continuous_command_braking')
        target=self.current.copy();target[[7,15]]=1.
        self.move_joints(target,'recovery:open_grippers')
        for arm in ('left','right'):
            pose=self.fk_model.fk(arm,np.asarray(HOME_ACTION)[self._arm_slice(arm)])
            self.move_pose(arm,pose,f'recovery:{arm}:collision_checked_park')
        for key in ('bowl','spoon','cup','plate'):
            if self.env.rule_state()['score']['items'][key]['placement_valid']:
                self.trace.append(dict(phase=f'{key}:already_valid_skip'));continue
            spec=self.manifest['objects'][key];arm='left' if key in ('bowl','spoon') else 'right'
            acquired=False;tried=set()
            for retry in range(3):
                choices=[x for x in self._choose_grasp(key,spec,arm) if x[2] not in tried]
                if not choices:break
                _,_,index,grasp=choices[0];tried.add(index);pre=grasp.copy();pre[2,3]+=.11
                self.move_pose(arm,pre,f'{key}:pregrasp');self.move_pose(arm,grasp,f'{key}:approach')
                self.hold(.5,f'{key}:approach_settle')
                error=float(np.linalg.norm(self._actual_tool_pose(arm)[:3,3]-grasp[:3,3]))
                if error>.006:
                    self.trace.append(dict(phase=f'{key}:measured_approach_retry',position_error_m=error))
                    self.move_pose(arm,pre,f'{key}:failed_approach_exit');continue
                self.gripper(arm,0.,f'{key}:close');self.hold(.4,f'{key}:close_settle')
                lift=self._actual_tool_pose(arm).copy();lift[2,3]+=.18
                self.move_pose(arm,lift,f'{key}:lift');self.hold(5.2,f'{key}:physical_continuous_hold')
                if self.env.rule_state()['score']['items'][key]['grasp_awarded']:acquired=True;break
                self.gripper(arm,1.,f'{key}:failed_grasp_release')
            if not acquired:raise RuntimeError(f'{key}: recovery candidates exhausted')
            body=self._body_matrix(key);tool=self._actual_tool_pose(arm);relative=np.linalg.inv(body)@tool
            destination=original.object_destination_pose(key,spec,self.manifest)
            if key in ('bowl','spoon','plate'):
                destination[:3,:3]=body[:3,:3]
                rotated=np.asarray(spec['local_geometry_points_m'])@body[:3,:3].T
                center=(rotated.min(0)+rotated.max(0))/2;desired=destination[:2,3].copy()
                if key=='spoon':desired=np.array([-.275,.055])
                destination[:2,3]=desired-center[:2]
                destination[2,3]=self.manifest['bins'][original.TaskDestinations(self.manifest)[key]]['floor_world_m']-rotated[:,2].min()+.015
            place=destination@relative;transit=place.copy();transit[2,3]+=.08
            if key=='plate':place,transit=self._reachable_plate_placement(spec,body,relative,destination,arm)
            self.move_pose(arm,transit,f'{key}:raised_transfer')
            if not self.env.rule_state()['objects'][key]['touching_gripper']:raise RuntimeError(f'{key}: recovery grasp lost')
            self.move_pose(arm,place,f'{key}:placement');self.gripper(arm,1.,f'{key}:release')
            retreat=self._actual_tool_pose(arm).copy();retreat[2,3]+=.08
            self.move_pose(arm,retreat,f'{key}:retreat');self.hold(2.,f'{key}:released_stability')
            if not self.env.rule_state()['score']['items'][key]['placement_valid']:
                raise RuntimeError(f'{key}: recovery placement invalid')
