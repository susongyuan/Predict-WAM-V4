"""Read-only, evaluation-only diagnostics; deliberately NOT integrated into snapshot.

No simulator/model imports, actuation, stepping, scoring assignment or policy handoff.
The predicate schema describes put_object_cabinet.check_success, not containment.
Call between physics steps; reads are sequential, NOT an atomic scene snapshot.
Main-thread integration must wait until its current single-seed run has finished.
"""
from __future__ import annotations


def native_diagnostics(env):
    """Return JSON-safe evidence. Unsupported/malformed reads are null + errors.

    API evidence: RoboTwin robot.py uses articulation.get_qpos/get_active_joints
    and joint.get_drive_target; SAPIEN physx.pyi exposes joint.get_dof; installed
    wrapper/scene.py:get_timestep reads physx_system.timestep. Multi-DOF mappings
    are deliberately rejected rather than assuming active-joint index == qpos.
    Ordinary read exceptions are recorded; cancellation BaseExceptions propagate.
    """
    import numpy as np

    out = {
        "schema": "robotwin-native-diagnostics-v1", "evaluation_only": True,
        "native_predicate_evidence": None, "arm_joint_tracking": {"left": None, "right": None},
        "scene_dt_seconds": None, "scene_elapsed_seconds": None, "errors": {},
        "limitations": ["Sequential read-only calls, not an atomic scene snapshot",
                        "Native predicate is not verified release/full containment/stability",
                        "Joint tracking excludes grippers; commands are not measured jaw opening",
                        "Scene dt is a configured physics increment, not elapsed time or hold duration"],
    }

    def read(key, fn):
        try:
            return fn()
        except Exception as exc:
            out["errors"][key] = f"{type(exc).__name__}: {exc}"
            return None

    def finite(value, shape=None):
        a = np.asarray(value)
        if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("expected finite numeric evidence")
        if shape is not None and a.shape != shape:
            raise ValueError(f"expected shape {shape}, got {a.shape}")
        return a.copy()

    def boolean(value):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("expected a boolean, not truthiness coercion")
        return bool(value)

    def predicate():
        # A task tag is an applicability check, not proof against a monkeypatched method.
        task = getattr(env, "task_name", type(env).__name__)
        if task != "put_object_cabinet":
            raise ValueError(f"unsupported native predicate task: {task}")
        p = {
            "task": task,
            "source": "third_party/RoboTwin/envs/put_object_cabinet.py:check_success",
            "object_reference": "object.get_pose().p (actor origin, NOT a functional point)",
            "object_functional_point_index": None,
            "object_functional_point_world": None,
            "object_functional_point_status": "not_used_by_native_predicate",
            "target_reference": "cabinet.get_functional_point(0), world xyz; native uses only xy",
            "cabinet_functional_point_index": 0,
            "object_position_world": None, "cabinet_functional_point_world": None,
            "cabinet_pose_world": None, "cabinet_joint_qpos": None,
            "origin_z": None, "arm_tag": None, "selected_gripper_arm": None,
            "gripper_commands": {"left": None, "right": None},
            "selected_gripper_command": None, "selected_native_gripper_open": None,
            "thresholds": {"abs_xy_lt_m": [0.05, 0.05], "height_delta_gt_m": 0.007,
                           "height_delta_lt_m": 0.12, "gripper_command_gt": 0.8,
                           "all_bounds_strict": True},
            "operand_dtypes": {}, "operand_scalar_kinds": {}, "numpy_version": np.__version__,
            "abs_xy_error_m": None, "height_delta_m": None, "conditions": None,
            "recomputed_predicate": None, "native_predicate": None,
            "recomputed_matches_native": None, "native_success_latch": None,
        }
        scalar_values = {}
        def scalar_operand(name, value):
            a = finite(value, ())
            # NumPy 2 treats Python scalars differently from NumPy scalars in
            # float32 arithmetic. Preserve that distinction for exact replay.
            python_scalar = type(value) in (int, float)
            scalar_values[name] = value if python_scalar else a[()]
            p["operand_scalar_kinds"][name] = "python" if python_scalar else "numpy"
            return a
        # Preserve operand dtypes for float32 boundary reproducibility, not just JSON floats.
        obj = read("predicate.object_position_world", lambda: finite(env.object.get_pose().p, (3,)))
        def target_xyz():
            target = finite(env.cabinet.get_functional_point(0))
            if target.shape not in ((3,), (7,)):
                raise ValueError("functional point must be world xyz or xyz+wxyz")
            return target[:3].copy()
        target = read("predicate.cabinet_functional_point_world", target_xyz)
        origin = read("predicate.origin_z", lambda: scalar_operand("origin_z", env.origin_z))
        for name, value in (("object_position_world", obj), ("cabinet_functional_point_world", target),
                            ("origin_z", origin)):
            if value is not None:
                p[name] = value.tolist()
                p["operand_dtypes"][name] = str(value.dtype)
        def cabinet_pose():
            pose = env.cabinet.get_pose()
            return np.concatenate((finite(pose.p, (3,)), finite(pose.q, (4,)))).tolist()
        p["cabinet_pose_world"] = read("predicate.cabinet_pose_world", cabinet_pose)
        def cabinet_qpos():
            q = finite(env.cabinet.get_qpos())
            if q.ndim != 1:
                raise ValueError("cabinet qpos must be a vector")
            return q.tolist()
        p["cabinet_joint_qpos"] = read("predicate.cabinet_joint_qpos", cabinet_qpos)
        def arm():
            tag = env.arm_tag
            # Native uses left if tag == 'left', otherwise right. Invalid tags are
            # recorded as unknown rather than silently interpreting corruption.
            if tag == "left":
                return "left"
            if tag == "right":
                return "right"
            raise ValueError("unrecognized ArmTag")
        p["arm_tag"] = p["selected_gripper_arm"] = read("predicate.arm_tag", arm)
        commands = {}
        for side in ("left", "right"):
            # Native is_*_gripper_open reads these stored commands directly.
            value = read(f"predicate.gripper_commands.{side}",
                         lambda side=side: scalar_operand(f"{side}_gripper_command",
                             getattr(env.robot, f"{side}_gripper_val")))
            commands[side] = value
            if value is not None:
                p["gripper_commands"][side] = value.tolist()
                p["operand_dtypes"][f"{side}_gripper_command"] = str(value.dtype)
        side = p["selected_gripper_arm"]
        if side is not None:
            command = commands[side]
            p["selected_gripper_command"] = p["gripper_commands"][side]
            p["selected_native_gripper_open"] = read("predicate.selected_native_gripper_open",
                lambda: boolean(getattr(env.robot, f"is_{side}_gripper_open")()))
            if all(v is not None for v in (obj, target, origin, command)):
                def recompute():
                    # Match native NumPy arithmetic, including scalar float32 subtraction.
                    xy = abs(obj[:2] - target[:2])
                    dz = obj[2] - scalar_values["origin_z"]
                    conditions = {"xy_within": bool(np.all(xy < np.array([0.05, 0.05]))),
                                  "above_min_height": bool(dz > 0.007),
                                  "below_max_height": bool(dz < 0.12),
                                  "gripper_command_open": bool(scalar_values[f"{side}_gripper_command"] > 0.8)}
                    p.update(abs_xy_error_m=finite(xy, (2,)).tolist(),
                             height_delta_m=finite(dz, ()).tolist(), conditions=conditions)
                    return all(conditions.values())
                p["recomputed_predicate"] = read("predicate.recomputed_predicate", recompute)
        # Original check_success is a read-only method for this task; never set eval_success.
        p["native_predicate"] = read("predicate.native_predicate", lambda: boolean(env.check_success()))
        p["native_success_latch"] = read("predicate.native_success_latch", lambda: boolean(env.eval_success))
        if p["native_predicate"] is not None and p["recomputed_predicate"] is not None:
            p["recomputed_matches_native"] = p["native_predicate"] == p["recomputed_predicate"]
        return p

    out["native_predicate_evidence"] = read("native_predicate_evidence", predicate)

    def tracking(side):
        entity = getattr(env.robot, f"{side}_entity")
        active = list(entity.get_active_joints())
        joints = list(getattr(env.robot, f"{side}_arm_joints"))
        if not active or not joints or len({id(j) for j in active}) != len(active):
            raise ValueError("missing/duplicate active joint identities")
        if len({id(j) for j in joints}) != len(joints):
            raise ValueError("duplicate arm joint identities")
        if any(j.get_dof() != 1 for j in active):
            raise ValueError("only verified single-DOF active-joint indexing is supported")
        qpos = finite(entity.get_qpos(), (len(active),))
        indices = [active.index(j) for j in joints]  # same mapping used by native robot.py
        actual = qpos[indices].astype(np.float64)
        target = np.array([finite(j.get_drive_target(), (1,))[0] for j in joints], dtype=np.float64)
        error = finite(actual - target, (len(joints),))
        return {"joint_names": [str(j.get_name()) for j in joints],
                "active_joint_indices": indices, "actual_qpos": actual.tolist(),
                "drive_target": target.tolist(), "actual_minus_drive_target": error.tolist(),
                "max_abs_error": float(np.max(abs(error))),
                "units": "native joint units (rad revolute / m prismatic); no wrapping",
                "grippers_included": False}
    for side in ("left", "right"):
        out["arm_joint_tracking"][side] = read(f"arm_joint_tracking.{side}", lambda side=side: tracking(side))

    def dt():
        value = finite(env.scene.get_timestep(), ())
        if value <= 0:
            raise ValueError("scene timestep must be positive")
        return float(value)
    out["scene_dt_seconds"] = read("scene_dt_seconds", dt)
    # Intentionally no dt * control_step: native TOPP consumes variable physics steps.
    return out
