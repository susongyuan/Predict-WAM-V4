"""Simulation demonstration generator ONLY. Never imported by pure WAM evaluation.
MPLib 0.2.1 screw planner adapted to official RoboTwin1 INSTALLATION.md section 3.2:
ignore the planner collision flag (joint limit and small-step guards remain).
No installed/shared MPLib file is modified. Not a collision-safe robot controller.
Upstream method SHA256: b8c157dda59a9c84d0261ee899960dab0becaeed8abcb9cab93ffcadf5349f5c
"""
import numpy as np
import toppra as ta
from mplib import Pose

def official_recipe_plan_screw(
    self,
    goal_pose: Pose,
    current_qpos: np.ndarray,
    *,
    qpos_step: float = 0.1,
    time_step: float = 0.1,
    wrt_world: bool = True,
    verbose: bool = False,
) -> dict[str, str | np.ndarray | np.float64]:
    # plan_screw ankor end
    """
    Plan from a start configuration to a goal pose of the end-effector using
    screw motion

    Args:
        goal_pose: pose of the goal
        current_qpos: current joint configuration (either full or move_group joints)
        qpos_step: size of the random step
        time_step: time step for the discretization
        wrt_world: if True, interpret the target pose with respect to the
            world frame instead of the base frame
        verbose: if True, will print the log of TOPPRA
    """
    current_qpos = self.pad_move_group_qpos(current_qpos.copy())
    self.robot.set_qpos(current_qpos, True)

    if wrt_world:
        goal_pose = self._transform_goal_to_wrt_base(goal_pose)

    def skew(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0],
        ])

    def pose2exp_coordinate(pose: Pose) -> tuple[np.ndarray, float]:
        def rot2so3(rotation: np.ndarray):
            assert rotation.shape == (3, 3)
            if np.isclose(rotation.trace(), 3):
                return np.zeros(3), 1
            if np.isclose(rotation.trace(), -1):
                return np.zeros(3), -1e6
            theta = np.arccos((rotation.trace() - 1) / 2)
            omega = (
                1
                / 2
                / np.sin(theta)
                * np.array([
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]).T
            )
            return omega, theta

        pose_mat = pose.to_transformation_matrix()
        omega, theta = rot2so3(pose_mat[:3, :3])
        if theta < -1e5:
            return omega, theta
        ss = skew(omega)
        inv_left_jacobian = (
            np.eye(3) / theta
            - 0.5 * ss
            + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
        )
        v = inv_left_jacobian @ pose_mat[:3, 3]
        return np.concatenate([v, omega]), theta

    self.pinocchio_model.compute_forward_kinematics(current_qpos)
    ee_index = self.link_name_2_idx[self.move_group]
    # relative_pose = T_base_goal * T_base_link.inv()
    relative_pose = goal_pose * self.pinocchio_model.get_link_pose(ee_index).inv()
    omega, theta = pose2exp_coordinate(relative_pose)

    if theta < -1e4:
        return {"status": "screw plan failed."}
    omega = omega.reshape((-1, 1)) * theta

    move_joint_idx = self.move_group_joint_indices
    path = [np.copy(current_qpos[move_joint_idx])]

    while True:
        self.pinocchio_model.compute_full_jacobian(current_qpos)
        J = self.pinocchio_model.get_link_jacobian(ee_index, local=False)
        delta_q = np.linalg.pinv(J) @ omega
        delta_q *= qpos_step / (np.linalg.norm(delta_q))
        delta_twist = J @ delta_q

        flag = False
        if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
            ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
            delta_q = delta_q * ratio
            delta_twist = delta_twist * ratio
            flag = True

        current_qpos += delta_q.reshape(-1)
        omega -= delta_twist

        def check_joint_limit(q):
            n = len(q)
            for i in range(n):
                if (
                    q[i] < self.joint_limits[i][0] - 1e-3
                    or q[i] > self.joint_limits[i][1] + 1e-3
                ):
                    return False
            return True

        within_joint_limit = check_joint_limit(current_qpos)
        self.planning_world.set_qpos_all(current_qpos[move_joint_idx])
        collide = self.planning_world.is_state_colliding()

        if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
            return {"status": "screw plan failed"}

        path.append(np.copy(current_qpos[move_joint_idx]))

        if flag:
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")
            times, pos, vel, acc, duration = self.TOPP(np.vstack(path), time_step)
            return {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
