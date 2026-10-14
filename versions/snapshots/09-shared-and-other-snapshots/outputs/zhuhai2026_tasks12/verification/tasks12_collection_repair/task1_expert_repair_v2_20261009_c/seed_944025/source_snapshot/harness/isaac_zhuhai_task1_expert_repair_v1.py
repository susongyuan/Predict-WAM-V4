"""Versioned task1 motion candidate; original PhysX scene and scoring unchanged.

Lift clear of the passive lid handle before sideways retreat. Reobserve garbage
and verify measured approach tracking before attempting a physical grasp. All
motion uses the original force-limited drives, strict IK and collision guards.
"""
import numpy as np
from harness import isaac_zhuhai_expert as original

BaseExpert = original.SmoothExpert


class SmoothExpert(BaseExpert):
    def _line(self, arm, start, offset, label, seconds=2.5):
        poses = []
        for fraction in np.linspace(0., 1., 9):
            pose = start.copy()
            pose[:3, 3] += np.asarray(offset) * fraction
            poses.append(pose)
        self.move_pose_path(arm, poses, label, seconds)

    def _open_lid(self, color, arm):
        spec = self.manifest['bins'][color]
        grasp = np.asarray(spec['closed_grasp_transform_world'], dtype=float).copy()
        grasp[:3, :3] = original.top_down_rotation([1. if arm == 'left' else -1., 0., 0.])
        pre = grasp.copy(); pre[2, 3] += .10
        self.move_pose(arm, pre, f'lid_{color}:pregrasp')
        self.move_pose(arm, grasp, f'lid_{color}:approach')
        self.gripper(arm, 0., f'lid_{color}:close_on_real_handle')
        self._debug_lid_state(f'lid_{color}:closed_grasp')
        from harness.isaac_zhuhai_task1 import lid_arc_world_poses
        arc = spec['expert_open_arc']
        poses = lid_arc_world_poses(spec, grasp, len(arc['gripper_world_pose_nodes']), rotate_gripper_with_lid=False)
        self.move_pose_path(arm, poses, f'lid_{color}:passive_hinge_contact_arc', arc['seconds'])
        self.hold(.3, f'lid_{color}:past_vertical_settle')
        self.gripper(arm, 1., f'lid_{color}:release')
        # Old failures opened fully, then closed during the low side retreat.
        # First clear the actual handle vertically; planned matrices only.
        self._line(arm, self._actual_tool_pose(arm), [0., 0., .045], f'lid_{color}:vertical_handle_exit')
        self._line(arm, self._actual_tool_pose(arm), [-.25 if arm == 'left' else .25, 0., -.025], f'lid_{color}:raised_outside_retreat')
        self._line(arm, self._actual_tool_pose(arm), [0., -.34, 0.], f'lid_{color}:clear_front_staging', 3.)
        self.hold(.5, f'lid_{color}:released_lid_check')
        if not self.env.rule_state()['score']['lids'][color]['open_now']:
            raise RuntimeError(f'lid_{color}: passive hinge failed unchanged released-open criterion')

    def _garbage_grasp_choices(self, key, spec, arm):
        body = self._body_matrix(key)
        candidates = []
        if key == 'milk_carton':
            from harness.isaac_zhuhai_task1 import primitive_world_bounds
            bounds = primitive_world_bounds(spec['parts'], body)
            center = (bounds.min(0) + bounds.max(0)) / 2
            # Pinch the upper carton body instead of driving deep through its
            # folded top. Recompute center/axes if a previous grasp moved it.
            for height in (.020, .008, 0.):
                for axis in (body[:3, 0], body[:3, 1]):
                    if np.linalg.norm(axis[:2]) < .5:
                        continue
                    for sign in (1., -1.):
                        pose = np.eye(4)
                        pose[:3, :3] = original.top_down_rotation(axis * sign)
                        pose[:3, 3] = center + [0., 0., height]
                        candidates.append(pose)
        else:
            pose = np.eye(4); pose[:3, :3] = spec['grasp_tcp_world_rotation']
            pose[:3, 3] = body[:3, :3] @ np.asarray(spec['grasp_point_local_m']) + body[:3, 3]
            if key == 'banana_peel':
                pose[2, 3] = max(pose[2, 3], self.manifest['table_surface_world_m'] + .013)
            candidates = [pose]
        choices = []
        for index, pose in enumerate(candidates):
            pre = pose.copy(); pre[2, 3] += .10
            reach = self.fk_model.ik(arm, pre, seed=self.current[self._arm_slice(arm)])
            if not reach.success:
                continue
            approach = self.fk_model.ik(arm, pose, seed=reach.joints)
            if approach.success:
                # Height tiers preserve the intended upper-body preference.
                choices.append((index // 4 if key == 'milk_carton' else 0,
                    float(np.linalg.norm(reach.joints - self.current[self._arm_slice(arm)])), index, pose))
        if not choices:
            raise RuntimeError(f'{key}: no strict reachable measured-body grasp candidate')
        return sorted(choices, key=lambda row: row[:2])

    def _acquire_garbage(self, key, spec, arm):
        tried = set()
        for retry in range(3):
            choices = [row for row in self._garbage_grasp_choices(key, spec, arm) if row[2] not in tried]
            if not choices:
                break
            _, _, index, grasp = choices[0]; tried.add(index)
            pre = grasp.copy(); pre[2, 3] += .10
            self.move_pose(arm, pre, f'{key}:pregrasp')
            self.move_pose(arm, grasp, f'{key}:approach')
            self.hold(.5, f'{key}:approach_settle')
            actual = self._actual_tool_pose(arm)
            error = float(np.linalg.norm(actual[:3, 3] - grasp[:3, 3]))
            if error > .006:
                self.trace.append(dict(phase=f'{key}:measured_approach_retry', candidate_index=index, position_error_m=error))
                exit_pose = actual.copy(); exit_pose[2, 3] += .10
                self.move_pose(arm, exit_pose, f'{key}:failed_approach_exit')
                self.hold(.6, f'{key}:failed_approach_object_settle')
                continue
            self.gripper(arm, 0., f'{key}:close')
            self.hold(.4, f'{key}:close_settle')
            lift = self._actual_tool_pose(arm).copy(); lift[2, 3] += .20
            self.move_pose(arm, lift, f'{key}:lift')
            self.hold(5.2, f'{key}:physical_continuous_hold')
            item = self.env.rule_state()['score']['items'][key]
            if item['grasp_awarded']:
                return
            self.trace.append(dict(phase=f'{key}:failed_physical_hold_retry', candidate_index=index, score=item))
            self.gripper(arm, 1., f'{key}:failed_grasp_release')
            self.hold(.6, f'{key}:failed_grasp_object_settle')
        raise RuntimeError(f'{key}: measured grasp retries exhausted without unchanged physical5s award')

    def _task1_once(self):
        if self.dof != 16:
            raise RuntimeError('Versioned task1 repair requires native16 measured dual-Panda')
        target = self.current.copy(); target[[self.arm_dof, self.dof - 1]] = 1.
        self.move_joints(target, 'open_both_grippers')
        for color, arm in [('dry', 'left'), ('wet', 'right')]:
            self._open_lid(color, arm)
        for key, spec in self.manifest['objects'].items():
            arm = spec['expert_arm']
            self._acquire_garbage(key, spec, arm)
            self._place_garbage_native16(key, spec, arm)
