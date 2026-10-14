"""One physically verified grasp or placement, then return control to WAM."""
import numpy as np
from harness import isaac_zhuhai_expert as original
from harness.isaac_zhuhai_dagger_expert import RecoveryExpert
from harness.zhuhai_dagger_segmented import goal_met
from harness.zhuhai_robot_spec import HOME_ACTION


class SegmentedExpert(RecoveryExpert):
    def prepare_grasp(self):
        target = self.current.copy()
        target[[7, 15]] = 1.
        self.move_joints(target, 'recovery:open_grippers')
        for arm in ('left', 'right'):
            pose = self.fk_model.fk(arm, np.asarray(HOME_ACTION)[self._arm_slice(arm)])
            self.move_pose(arm, pose, f'recovery:{arm}:collision_checked_park')

    def acquire(self, key, spec, arm):
        self.prepare_grasp()
        tried = set()
        for _ in range(3):
            choices = [x for x in self._choose_grasp(key, spec, arm) if x[2] not in tried]
            if not choices:
                break
            _, _, index, grasp = choices[0]
            tried.add(index)
            pre = grasp.copy()
            pre[2, 3] += .11
            self.move_pose(arm, pre, f'{key}:pregrasp')
            self.move_pose(arm, grasp, f'{key}:approach')
            self.hold(.5, f'{key}:approach_settle')
            error = float(np.linalg.norm(self._actual_tool_pose(arm)[:3, 3]-grasp[:3, 3]))
            if error > .006:
                self.trace.append(dict(phase=f'{key}:measured_approach_retry', position_error_m=error))
                self.move_pose(arm, pre, f'{key}:failed_approach_exit')
                continue
            self.gripper(arm, 0., f'{key}:close')
            self.hold(.4, f'{key}:close_settle')
            lift = self._actual_tool_pose(arm).copy()
            lift[2, 3] += .18
            self.move_pose(arm, lift, f'{key}:lift')
            self.hold(5.2, f'{key}:physical_continuous_hold')
            if goal_met(self.env.rule_state(), 'grasp', key):
                return
            self.gripper(arm, 1., f'{key}:failed_grasp_release')
        raise RuntimeError(f'{key}: local recovery grasp candidates exhausted')

    def place(self, key, spec, arm):
        body, tool = self._body_matrix(key), self._actual_tool_pose(arm)
        relative = np.linalg.inv(body)@tool
        destination = original.object_destination_pose(key, spec, self.manifest)
        if key in ('bowl', 'spoon', 'plate'):
            destination[:3, :3] = body[:3, :3]
            rotated = np.asarray(spec['local_geometry_points_m'])@body[:3, :3].T
            center = (rotated.min(0)+rotated.max(0))/2
            desired = destination[:2, 3].copy()
            if key == 'spoon':
                desired = np.array([-.275, .055])
            destination[:2, 3] = desired-center[:2]
            destination[2, 3] = self.manifest['bins'][original.TaskDestinations(self.manifest)[key]]['floor_world_m']-rotated[:, 2].min()+.015
        place = destination@relative
        transit = place.copy()
        transit[2, 3] += .08
        if key == 'plate':
            place, transit = self._reachable_plate_placement(spec, body, relative, destination, arm)
        self.move_pose(arm, transit, f'{key}:raised_transfer')
        if not self.env.rule_state()['objects'][key]['touching_gripper']:
            raise RuntimeError(f'{key}: grasp lost during local placement')
        self.move_pose(arm, place, f'{key}:placement')
        self.gripper(arm, 1., f'{key}:release')
        retreat = self._actual_tool_pose(arm).copy()
        retreat[2, 3] += .08
        self.move_pose(arm, retreat, f'{key}:retreat')
        self.hold(2., f'{key}:released_stability')
        if not goal_met(self.env.rule_state(), 'placement', key):
            raise RuntimeError(f'{key}: local placement not physically verified')

    def run_segment(self, goal, key):
        failure = None
        try:
            self.hold(1., 'recovery:continuous_command_braking')
            arm = 'left' if key in ('bowl', 'spoon') else 'right'
            spec = self.manifest['objects'][key]
            if not goal_met(self.env.rule_state(), 'grasp', key):
                self.acquire(key, spec, arm)
            if goal == 'placement':
                self.place(key, spec, arm)
            elif goal != 'grasp':
                raise ValueError('Unsupported local goal')
            # End with inherited velocities decelerated, without opening a
            # verified grasp or moving any other object to finish the task.
            self.hold(.5, f'{key}:local_goal_settle')
        except original._PhysicalEpisodeComplete:
            self.trace.append(dict(phase='full_task_complete_at_rule_boundary'))
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        evidence = self.env.rule_state()
        passed = failure is None and goal_met(evidence, goal, key)
        if not passed and failure is None:
            failure = 'LocalGoalNotMet: verified state did not survive handback settling'
        return dict(goal=goal, key=key, passed=bool(passed), failure=failure,
                    full_task_success=bool(evidence['score']['full_task_success']),
                    goal_evidence=dict(item=evidence['score']['items'][key],
                                       object=evidence['objects'][key], physics_safety=evidence['physics_safety']),
                    trace=self.trace, command_count=len(self.motion_commands), record_video=False)
