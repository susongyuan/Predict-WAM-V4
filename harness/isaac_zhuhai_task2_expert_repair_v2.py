"""Versioned task2 expert candidate: rim diversity and attitude-preserving carry.

Same frozen scene, force-limited joints, collision checks, safety and score. No
object pose writes, physical attachments, or post-failure teleport recovery.
"""
import numpy as np
from harness import isaac_zhuhai_expert as original

BaseExpert=original.SmoothExpert

class SmoothExpert(BaseExpert):
    def _choose_grasp(self,key,spec,arm):
        body=self._body_matrix(key);surface=self.manifest['table_surface_world_m']
        candidates=original.grasp_pose_candidates(key,spec,body,surface)
        if key=='plate':
            # Search physical plate rim locations, including the original one.
            scale=np.asarray(spec['scale']);radius=4.15*scale[2]
            # Original rim first; alternative rims only after original failure.
            originals=[]
            local=np.array([0.,.62,4.15])*scale;point=body[:3,:3]@local+body[:3,3]
            for sign in (1,-1):
                pose=np.eye(4);pose[:3,:3]=original.top_down_rotation(body[:3,2]*sign);pose[:3,3]=point;pose[2,3]=max(point[2],surface+.013);originals.append(pose)
            candidates=list(originals)
            for angle in (0.,np.pi/2,np.pi,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4):
                radial=np.array([np.sin(angle),0.,np.cos(angle)])
                point=body[:3,:3]@np.array([radius*radial[0],.62*scale[1],radius*radial[2]])+body[:3,3]
                for sign in (1,-1):
                    pose=np.eye(4);pose[:3,:3]=original.top_down_rotation(body[:3,:3]@radial*sign);pose[:3,3]=point
                    pose[2,3]=max(point[2],surface+.013);candidates.append(pose)
        elif key=='spoon':
            local=np.array([-40.,0.,5.])*np.asarray(spec['scale']);z=(body[:3,:3]@local+body[:3,3])[2]
            for pose in candidates:pose[2,3]=max(z,surface+.013)
        choices=[]
        for candidate_index,pose in enumerate(candidates):
            pre=pose.copy();pre[2,3]+=.11
            result=self.fk_model.ik(arm,pre,seed=self.current[self._arm_slice(arm)])
            if not result.success:continue
            grasp=self.fk_model.ik(arm,pose,seed=result.joints)
            if not grasp.success:continue
            margin=min(result.joints[3],grasp.joints[3])-self.fk_model.limits[arm][3,0]
            cost=float(np.linalg.norm(result.joints-self.current[self._arm_slice(arm)]))+.8/max(margin,.01)
            priority=1 if key=='plate' and candidate_index>=2 else 0
            choices.append((priority,cost,candidate_index,pose))
        if not choices:raise RuntimeError(f'{key}: no reachable collision-clear repaired rim candidate')
        return sorted(choices,key=lambda item:item[:2])

    def _task2_once(self):
        target=self.current.copy();target[[self.arm_dof,self.dof-1]]=1.;self.move_joints(target,'open_both_grippers')
        for key in ['bowl','spoon','cup','plate']:
            spec=self.manifest['objects'][key];arm='left' if key in ['bowl','spoon'] else 'right'
            acquired=False;tried=set()
            for retry in range(3):
                choices=self._choose_grasp(key,spec,arm)
                choices=[choice for choice in choices if choice[2] not in tried]
                if not choices:break
                _,_,index,grasp=choices[0];tried.add(index)
                pre=grasp.copy();pre[2,3]+=.11
                self.move_pose(arm,pre,f'{key}:pregrasp');self.move_pose(arm,grasp,f'{key}:approach')
                self.hold(.5,f'{key}:approach_settle')
                actual=self._actual_tool_pose(arm);error=float(np.linalg.norm(actual[:3,3]-grasp[:3,3]))
                if error>.006:
                    self.trace.append(dict(phase=f'{key}:measured_approach_retry',candidate_index=index,position_error_m=error))
                    self.move_pose(arm,pre,f'{key}:failed_approach_exit');continue
                self.gripper(arm,0.,f'{key}:close');self.hold(.4,f'{key}:close_settle')
                lift=self._actual_tool_pose(arm).copy();lift[2,3]+=.18
                self.move_pose(arm,lift,f'{key}:lift');self.hold(5.2,f'{key}:physical_continuous_hold')
                item=self.env.rule_state()['score']['items'][key]
                if item['grasp_awarded']:acquired=True;break
                self.trace.append(dict(phase=f'{key}:failed_physical_hold_retry',candidate_index=index,score=item))
                # Actual open-drive release, no object writes. Reobserve the body
                # before selecting another rim; failed candidate stays in demo.
                self.gripper(arm,1.,f'{key}:failed_grasp_release')
            if not acquired:raise RuntimeError(f'{key}: 3 measured approach/physical hold candidates exhausted')
            body=self._body_matrix(key);tool=self._actual_tool_pose(arm);object_to_tool=np.linalg.inv(body)@tool
            destination=original.object_destination_pose(key,spec,self.manifest)
            if key in {'bowl','spoon','plate'}:
                # No upright requirement for these objects. Preserve physical
                # grasp attitude instead of rotating the wrist around the rim.
                destination[:3,:3]=body[:3,:3]
                rotated=np.asarray(spec['local_geometry_points_m'])@body[:3,:3].T
                center=(rotated.min(0)+rotated.max(0))/2
                desired=destination[:2,3].copy()
                if key=='spoon':desired=np.array([-.275,.055])
                destination[:2,3]=desired-center[:2]
                color=original.TaskDestinations(self.manifest)[key]
                destination[2,3]=self.manifest['bins'][color]['floor_world_m']-rotated[:,2].min()+.015
            place=destination@object_to_tool;transit=place.copy();transit[2,3]+=.08
            self.move_pose(arm,transit,f'{key}:raised_transfer')
            if not self.env.rule_state()['objects'][key]['touching_gripper']:raise RuntimeError(f'{key}: physical grasp lost during raised transfer')
            self.move_pose(arm,place,f'{key}:placement');self.gripper(arm,1.,f'{key}:release')
            retreat=self._actual_tool_pose(arm).copy();retreat[2,3]+=.08;self.move_pose(arm,retreat,f'{key}:retreat')
            self.hold(2.0,f'{key}:released_stability')
            item=self.env.rule_state()['score']['items'][key]
            if not item['placement_valid']:raise RuntimeError(f'{key}: released stable containment/upright/support criterion failed; score={item}')
