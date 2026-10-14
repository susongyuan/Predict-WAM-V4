"""Isolated MPLib Cartesian expert for Zhuhai; original robot modes stay intact.

The installed simulation environment has MPLib 0.2.1 but no CuRobo. This
declared protocol uses real IK/RRT/TOPP and robot drive targets. It never moves
task objects, replays joint trajectories, or participates in learned control.
"""
import numpy as np
import mplib

from envs.robot import Robot
from envs.robot.planner import MplibPlanner


class CartesianExpert(MplibPlanner):
    def __init__(self, urdf, srdf, group, base, entity, planner_type, scene):
        super().__init__(urdf, srdf, group, base, entity, 'mplib_RRT', scene)

    def plan_path(self, now_qpos, target_pose, constraint_pose=None, arms_tag=None, **kwargs):
        # Experts here request fully specified poses; nonzero partial-pose
        # costs from CuRobo are not silently reinterpreted as MPLib constraints.
        if constraint_pose is not None and np.any(np.asarray(constraint_pose)):
            raise ValueError('MPLib expert requires fully specified Cartesian poses; partial-pose constraint unsupported')
        goal = mplib.Pose(target_pose.p, target_pose.q)
        result = self.planner.plan_pose(goal, np.asarray(now_qpos), time_step=1/250,
                                       planning_time=2., wrt_world=True)
        if result.get('status')=='Success' and len(result['position']) > self.plan_step_lim:
            return {'status':'Fail','reason':'trajectory_step_limit'}
        return result

    def plan_batch(self, now_qpos, target_pose_list, constraint_pose=None, arms_tag=None):
        paths = [self.plan_path(now_qpos, pose, constraint_pose, arms_tag) for pose in target_pose_list]
        status = np.array([path.get('status','Failure') for path in paths], dtype=object)
        maximum = max((len(path['position']) for path in paths if path.get('status')=='Success'), default=0)
        result = {'status':status}
        if maximum:
            width = next(path['position'].shape[-1] for path in paths if path.get('status')=='Success')
            pos = np.zeros((len(paths), maximum, width)); vel=np.zeros_like(pos)
            for i,path in enumerate(paths):
                if path.get('status')=='Success':
                    n=len(path['position']); pos[i,:n]=path['position']; pos[i,n:]=path['position'][-1]
                    vel[i,:n]=path['velocity']
            result.update(position=pos, velocity=vel)
        return result


class ZhuhaiRobot(Robot):
    def _set_planner_mode(self, mode):
        if mode=='mplib_expert':
            if hasattr(self,'planner_mode') and self.planner_mode!=mode:
                raise ValueError('Create a new robot when switching planner modes')
            self.planner_mode=mode
        else:
            super()._set_planner_mode(mode)

    def set_planner(self, scene=None, planner_mode=None):
        self._set_planner_mode(planner_mode or self.planner_mode)
        if self.planner_mode!='mplib_expert':
            return super().set_planner(scene, planner_mode)
        self.communication_flag=False
        self._set_mplib_planners(scene, planner_class=CartesianExpert)
        self.left_planner,self.right_planner=self.left_mplib_planner,self.right_mplib_planner


def load_zhuhai_robot(env, **kwargs):
    env.robot = ZhuhaiRobot(env.scene, env.need_topp, **kwargs)
    env.robot.set_planner(env.scene)
    env.robot.init_joints()
    for link in env.robot.left_entity.get_links()+env.robot.right_entity.get_links():
        link.set_mass(1)
