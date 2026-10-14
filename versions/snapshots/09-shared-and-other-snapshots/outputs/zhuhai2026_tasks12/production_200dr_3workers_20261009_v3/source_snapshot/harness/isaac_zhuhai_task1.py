"""Native OpenUSD/PhysX life task-1 scene; no SAPIEN or Kit startup dependency.

Call ``build(stage, root='/World/Task', seed=0, randomized=False)`` after the
coordinator has created an Isaac Sim stage. Objects are posed only during build.
Lids are passive world-anchored revolute rigid bodies, with no DriveAPI. Runtime
must move the imported robot's joints, read real contact pairs, and call
``RuntimeJudge.observe`` once per physics step. This file creates no cameras:
the coordinator attaches exactly head/left-wrist/right-wrist to robot links.
"""
from __future__ import annotations

import itertools
import math
from copy import deepcopy

import numpy as np

from harness.zhuhai_task1_rules import DESTINATIONS, PhysicalState, Task1Judge


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
RULE_VERSION = "zhuhai_life_task1_isaac_native16_20261009"
LID_GRASP_WORLD_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
OBJECT_GRASP_WORLD_ROTATION = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
_DROP_TILT = math.radians(30.)
OBJECT_DROP_ROTATION = np.array([[1., 0., 0.], [0., math.cos(_DROP_TILT), -math.sin(_DROP_TILT)],
                                [0., math.sin(_DROP_TILT), math.cos(_DROP_TILT)]])
OBJECT_DROP_TCP_ROTATION = OBJECT_DROP_ROTATION@OBJECT_GRASP_WORLD_ROTATION


def _part(name, dimensions, position=(0, 0, 0), color=(.8, .8, .8), yaw=0., kind="cube"):
    return {"name": name, "kind": kind, "dimensions_m": list(map(float, dimensions)),
            "position_m": list(map(float, position)), "yaw_rad": float(yaw),
            "color": list(map(float, color))}


def _bounds(parts):
    corners = []
    for part in parts:
        half = np.asarray(part["dimensions_m"])/2
        c, s = math.cos(part["yaw_rad"]), math.sin(part["yaw_rad"])
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners.extend(np.asarray(part["position_m"])+rotation@(half*np.array(sign))
                       for sign in itertools.product((-1, 1), repeat=3))
    points = np.array(corners)
    lo, hi = points.min(0), points.max(0)
    return np.array([np.where(np.array(sign)>0, hi, lo)
                     for sign in itertools.product((-1, 1), repeat=3)]).tolist()


def _dish_parts(kind, color=(.84, .86, .88)):
    radius, height = (.110, .021) if kind == "plate" else (.078, .054)
    # A compound ring retains a real open interior; a convex hull would seal it.
    parts = [_part("floor", (radius*2, radius*2, .006), (0, 0, .003), color, kind="cylinder")]
    wall = .006
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 20, endpoint=False)):
        center = [(radius-wall/2)*np.cos(angle), (radius-wall/2)*np.sin(angle), height/2]
        parts.append(_part(f"rim_{i:02d}", (wall, 2*radius*np.sin(np.pi/20)*1.04, height),
                           center, color, yaw=angle))
    return parts


def scene_spec(seed=0, randomized=False, table_height=.74, root="/World/Task"):
    """Pure CPU JSON specification, including every collision component."""
    if not root.startswith("/") or root == "/" or not .4 < table_height < 1.4:
        raise ValueError("invalid task USD root or table height")
    rng = np.random.default_rng(seed)
    objects, supports, bins = {}, {}, {}
    jitter = .020 if randomized else .008
    source_layout = {"milk_carton": (-.24, -.25), "squeeze_bottle": (0, -.19), "banana_peel": (.24, -.25)}
    for key, xy in source_layout.items():
        xy = np.array(xy)+rng.uniform(-jitter, jitter, 2)
        scale = float(rng.uniform(.90, 1.10)) if randomized else 1.0
        if key == "milk_carton":
            parts = [_part("carton", (.055, .055, .096), color=(.9, .91, .87)),
                     _part("folded_top", (.055, .046, .016), (0, 0, .053), (.18, .45, .67)),
                     _part("blue_label", (.040, .0016, .036), (0, -.0276, 0), (.12, .42, .66))]
            mass, grasp = .025, [0, 0, 0]
        elif key == "squeeze_bottle":
            parts = [_part("bottle_body", (.046, .046, .086), color=(.76, .19, .13), kind="cylinder"),
                     _part("cap", (.024, .024, .018), (0, 0, .052), (.91, .88, .76), kind="cylinder"),
                     _part("nozzle", (.014, .014, .014), (0, 0, .068), (.87, .70, .21), kind="cone")]
            mass, grasp = .030, [0, 0, .038]
        else:
            parts = [_part("peel_core", (.042, .034, .022), (0, 0, .008), (.91, .77, .18))]
            for i, angle in enumerate((0, 2*np.pi/3, -2*np.pi/3)):
                parts.append(_part(f"peel_petal_{i}", (.056, .022, .008),
                    (.030*np.cos(angle), .030*np.sin(angle), 0), (.85, .70, .16), yaw=angle))
            mass, grasp = .012, [0, 0, .012]
        for part in parts:
            part["dimensions_m"] = (np.asarray(part["dimensions_m"])*scale).tolist()
            part["position_m"] = (np.asarray(part["position_m"])*scale).tolist()
        corners = _bounds(parts)
        bottom = min(p[2] for p in corners)
        objects[key] = {"body_path": f"{root}/Objects/{key}", "initial_position_m": [*xy, table_height+.008-bottom],
                        "parts": parts, "mass_kg": mass*scale**3, "local_corners_m": corners,
                        "grasp_point_local_m": (np.asarray(grasp)*scale).tolist(),
                        "destination": DESTINATIONS[key], "expert_arm": "right" if key == "banana_peel" else "left",
                        "proxy": "rigid banana peel; no deformation/fragmentation" if key == "banana_peel" else "rigid empty container"}
        dish = "bowl" if key == "squeeze_bottle" else "plate"
        parts = _dish_parts(dish)
        supports[key] = {"body_path": f"{root}/Supports/{key}_{dish}", "initial_position_m": [*xy, table_height],
                         "parts": parts, "mass_kg": .30, "local_corners_m": _bounds(parts), "kind": dish}
    for color, x in (("dry", -.255), ("wet", .255)):
        a, b, h, wall = .145, .105, .170, .004
        xy = np.array([x, .02])+(rng.uniform(-.008, .008, 2) if randomized else 0)
        origin = np.array([*xy, table_height])
        rgb = (.28, .35, .44) if color == "dry" else (.15, .48, .23)
        body_parts = [_part("floor", (2*(a+wall), 2*(b+wall), .006), (0, 0, .003), rgb)]
        for name, xyz, dimensions in [("left", (-a-wall, 0, h/2), (2*wall, 2*(b+wall), h)),
                                      ("right", (a+wall, 0, h/2), (2*wall, 2*(b+wall), h)),
                                      ("front", (0, -b-wall, h/2), (2*a, 2*wall, h)),
                                      ("back", (0, b+wall, h/2), (2*a, 2*wall, h))]:
            body_parts.append(_part(name, dimensions, xyz, rgb))
        handle = [0, -b+.024, .023]
        knob = _part("handle", (.052, .032, .032), handle, (.76, .77, .78), kind="cylinder")
        knob["axis"] = "X"
        lid_parts = [_part("lid", (2*(a+wall), 2*(b+wall), .006), color=rgb),
                     _part("handle_stem", (.014, .014, .018), (0, handle[1], .009), (.70, .71, .72)), knob]
        bins[color] = {"body_path": f"{root}/Bins/{color}/Body", "lid_path": f"{root}/Bins/{color}/Lid",
                      "joint_path": f"{root}/Bins/{color}/LidHinge", "initial_position_m": origin.tolist(),
                      "body_parts": body_parts, "body_local_corners_m": _bounds(body_parts),
                      "lid_parts": lid_parts, "lid_local_corners_m": _bounds(lid_parts), "lid_mass_kg": .06,
                      "lid_initial_position_m": (origin+[0, 0, h+.004]).tolist(),
                      "hinge_world_m": (origin+[0, b+wall, h+.004]).tolist(),
                      "hinge_in_lid_m": [0, b+wall, 0], "handle_in_lid_m": handle,
                      "joint_axis": "X", "joint_limits_degrees": [-math.degrees(1.72), 0.0],
                      "destination_bounds_m": [(origin+[-a, -b, .006]).tolist(), (origin+[a, b, h-.003]).tolist()],
                      "mouth_height_m": table_height+h, "fixed_to_world": True,
                      "motor_enabled": False, "open_hold_mechanism": "passive past-vertical joint limit"}
    geometry = {}
    for key, obj in objects.items():
        target = np.array(bins[obj["destination"]]["destination_bounds_m"]).mean(0)
        if obj["destination"] == "dry":
            target[0] += -.055 if key == "milk_carton" else .055
        target[2] = bins[obj["destination"]]["mouth_height_m"]+.12
        obj.update(prim_path=obj["body_path"], local_geometry_points_m=obj["local_corners_m"],
                   initial_position_world_m=obj["initial_position_m"], quaternion_wxyz=[1., 0., 0., 0.],
                   desired_object_orientation_wxyz=[math.cos(_DROP_TILT/2), math.sin(_DROP_TILT/2), 0., 0.],
                   expert_grasp={"point_local_m": obj["grasp_point_local_m"], "pinch_axis_local": [1, 0, 0],
                                 "approach_axis_local": [0, 0, -1], "pregrasp_distance_m": .10},
                   grasp_offset_world_m=obj["grasp_point_local_m"], drop_position_world_m=target.tolist(),
                   grasp_tcp_world_rotation=OBJECT_GRASP_WORLD_ROTATION.tolist(),
                   drop_tcp_world_rotation=OBJECT_DROP_TCP_ROTATION.tolist(),
                   drop_grasp_offset_world_m=(OBJECT_DROP_ROTATION@np.asarray(obj["grasp_point_local_m"])).tolist(),
                   drop_tcp_position_world_m=(target+OBJECT_DROP_ROTATION@np.asarray(obj["grasp_point_local_m"])).tolist(),
                   expert_carry={"wrist_tilt_degrees": 30., "transition": "smooth after lift",
                                 "predrop_clearance_m": .06, "no_object_pose_reset": True})
        geometry[key] = (np.asarray(obj["local_corners_m"])+obj["initial_position_m"]).tolist()
        geometry[f"{key}_source_tableware"] = (np.asarray(supports[key]["local_corners_m"])+supports[key]["initial_position_m"]).tolist()
    for color, spec in bins.items():
        spec.update(root_path=f"{root}/Bins/{color}", floor_path=spec["body_path"]+"/floor",
                    interior_min_world_m=spec["destination_bounds_m"][0], interior_max_world_m=spec["destination_bounds_m"][1],
                    hinge_axis_world=[1., 0., 0.], lid_closed_orientation_wxyz=[1., 0., 0., 0.],
                    handle_closed_world_m=(np.array(spec["lid_initial_position_m"])+spec["handle_in_lid_m"]).tolist(),
                    expert_open_arc={"joint_angles_rad": (-1.68*(10*np.linspace(0., 1., 25)**3
                        -15*np.linspace(0., 1., 25)**4+6*np.linspace(0., 1., 25)**5)).tolist(),
                                     "seconds": 3.5, "path_interpolation": "minimum jerk in hinge angle",
                                     "actuation": "robot gripper contact only",
                                     "gripper_orientation_mode": "fixed; cylindrical handle rolls about jaw-normal/hinge axis",
                                     "rotate_gripper_with_lid": False})
        handle_matrix = np.eye(4)
        handle_matrix[:3, 3] = spec["handle_closed_world_m"]
        spec["closed_handle_transform_world"] = handle_matrix.tolist()
        grasp_matrix = handle_matrix.copy()
        grasp_matrix[:3, :3] = LID_GRASP_WORLD_ROTATION
        spec["closed_grasp_transform_world"] = grasp_matrix.tolist()
        spec["expert_open_arc"]["gripper_world_pose_nodes"] = [pose.tolist()
            for pose in lid_arc_world_poses(spec, grasp_matrix, 25)]
        geometry[f"bin_{color}"] = (np.asarray(spec["body_local_corners_m"])+spec["initial_position_m"]).tolist()
        geometry[f"lid_{color}"] = (np.asarray(spec["lid_local_corners_m"])+spec["lid_initial_position_m"]).tolist()
    return {"engine": "Isaac Sim 6.1 / PhysX", "schema_version": RULE_VERSION, "root": root, "root_path": root,
            "seed": int(seed), "randomized": bool(randomized), "track": "life", "task": "zhuhai_task1",
            "table_surface_world_m": float(table_height), "objects": objects, "supports": supports, "bins": bins,
            "geometry_world_m": geometry, "table_prim_path": f"{root}/Table", "table_height_bias_m": table_height-.74,
            "articulations": {color: {"joint_path": spec["joint_path"], "body_path": spec["lid_path"],
                                       "type": "passive world-anchored revolute rigid joint", "motor_enabled": False}
                              for color, spec in bins.items()},
            "cameras": {"names": list(CAMERA_NAMES), "count": 3, "attach_to_robot_links": True,
                        "head_position_robot_mast_world_m": [0, -.58, 1.85+table_height-.74],
                        "head_look_at_world_m": [0, .02, table_height+.08], "head_fovy_degrees": 65.,
                        "resolution_hw": [240, 320], "wrist_links_required": ["left", "right"],
                        "render_visibility_verified": False},
            "policy_inputs": ["three onboard RGB observations", "measured robot joint positions/gripper openings"],
            "scoring_privileged_not_policy_input": True, "video_recording": False,
            "third_view_camera": False, "physical_expert_validated": False,
            "limitations": ["rigid banana/squeeze-bottle proxy", "official object models and fixture layout unspecified",
                            "competition damage and banana-fragment partial scoring not represented"]}


def _semantics(prim, label):
    from pxr import Sdf
    try:
        from pxr import UsdSemantics
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([label])
    except ImportError:
        prim.CreateAttribute("userProperties:semanticLabel", Sdf.ValueTypeNames.String).Set(label)


def _author_body(stage, spec, parts_key="parts", position_key="initial_position_m", dynamic=True, material_path=None,
                 label=None):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    path = spec["body_path"]
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*spec[position_key]))
    prim = body.GetPrim()
    if label:
        _semantics(prim, label)
    if dynamic:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec["mass_kg"]))
        # Pure USD CPU authoring can precede loading NVIDIA's schema extension.
        # Kit resolves these registered Physx schemas when it opens the stage.
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(.03)
        prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(.30)
        prim.AddAppliedSchema("PhysxContactReportAPI")
        prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.)
    for part in spec[parts_key]:
        shape_path = f"{path}/{part['name']}"
        dimensions = np.asarray(part["dimensions_m"])
        if part["kind"] == "cube":
            shape = UsdGeom.Cube.Define(stage, shape_path)
            shape.CreateSizeAttr(1.)
            scaling = dimensions
        else:
            cls = {"cylinder": UsdGeom.Cylinder, "cone": UsdGeom.Cone}[part["kind"]]
            shape = cls.Define(stage, shape_path)
            shape.CreateAxisAttr(part.get("axis", "Z"))
            shape.CreateRadiusAttr(.5)
            shape.CreateHeightAttr(1.)
            scaling = dimensions
        xform = UsdGeom.Xformable(shape)
        xform.AddTranslateOp().Set(Gf.Vec3d(*part["position_m"]))
        xform.AddRotateZOp().Set(math.degrees(part["yaw_rad"]))
        xform.AddScaleOp().Set(Gf.Vec3f(*scaling))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*part["color"])])
        if label:
            _semantics(shape.GetPrim(), label)
        collider = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        collider.CreateCollisionEnabledAttr(True)
        shape.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.002)
        shape.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
        if material_path:
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(UsdShade.Material.Get(stage, material_path),
                                                                   materialPurpose="physics")
    return prim


def build(stage, root="/World/Task", seed=0, randomized=False, table_z_bias=0., source_seed=None, table_height=None):
    """Author scene geometry once and return a JSON-compatible runtime manifest."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    if stage.GetPrimAtPath(root):
        raise ValueError(f"Refusing to overwrite an existing task subtree: {root}")
    table_height = .74+table_z_bias if table_height is None else float(table_height)
    spec = scene_spec(seed, randomized, table_height, root)
    spec["source_seed"] = seed if source_seed is None else int(source_seed)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, root)
    material_path = f"{root}/Materials/Objects"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction = float(np.random.default_rng(seed+913).uniform(.45, .80)) if randomized else .60
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction*.85)
    physics_material.CreateRestitutionAttr(.02)
    spec["object_material"] = {"static_friction": friction, "dynamic_friction": friction*.85, "restitution": .02}
    table_spec = {"body_path": spec["table_prim_path"], "initial_position_m": [0, 0, table_height-.025],
                  "parts": [_part("surface", (1.30, .90, .05), color=(.50, .40, .30))]}
    _author_body(stage, table_spec, dynamic=False, material_path=material_path, label="table")
    for key, obj in spec["objects"].items():
        _author_body(stage, obj, material_path=material_path, label=key)
    for key, support in spec["supports"].items():
        _author_body(stage, support, material_path=material_path, label=f"{key}_source_tableware")
    for color, bin_spec in spec["bins"].items():
        _author_body(stage, bin_spec, parts_key="body_parts", dynamic=False, material_path=material_path, label=f"bin_{color}")
        lid_spec = {"body_path": bin_spec["lid_path"], "initial_position_m": bin_spec["lid_initial_position_m"],
                    "parts": bin_spec["lid_parts"], "mass_kg": bin_spec["lid_mass_kg"]}
        _author_body(stage, lid_spec, material_path=material_path, label=f"lid_{color}")
        joint = UsdPhysics.RevoluteJoint.Define(stage, bin_spec["joint_path"])
        # Body0 is the static world. Both joint frame positions are expressed
        # relative to their owning body; USD rotation limits are DEGREES.
        joint.CreateBody1Rel().SetTargets([Sdf.Path(bin_spec["lid_path"])])
        joint.CreateAxisAttr("X")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*bin_spec["hinge_world_m"]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*bin_spec["hinge_in_lid_m"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateLowerLimitAttr(bin_spec["joint_limits_degrees"][0])
        joint.CreateUpperLimitAttr(0.)
        joint.CreateCollisionEnabledAttr(False)
        joint.GetPrim().AddAppliedSchema("PhysxJointAPI")
        joint.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(.002)
        # Deliberately no angular drive/target or kinematic lid.
    return spec


def transform_corners(local_corners, position, orientation_wxyz):
    q = np.asarray(orientation_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid measured body quaternion")
    w, x, y, z = q/np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return np.asarray(local_corners)@rotation.T+np.asarray(position)


def primitive_world_bounds(parts, body_matrix):
    """Exact axis-aligned world support bounds of the authored primitive union.

    Unlike rotating a local AABB, analytic cylinder/cone support includes no
    fictitious corners and does not undersample curved surfaces.
    """
    matrix=np.asarray(body_matrix,dtype=float);bounds=[]
    for part in parts:
        angle=part['yaw_rad'];c,s=np.cos(angle),np.sin(angle)
        rotation=matrix[:3,:3]@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        center=matrix[:3,:3]@np.asarray(part['position_m'])+matrix[:3,3]
        half=np.asarray(part['dimensions_m'])/2
        if part['kind']=='cube':
            extent=np.abs(rotation)@half;lo,hi=center-extent,center+extent
        else:
            axis='XYZ'.index(part.get('axis','Z'));radial=[i for i in range(3) if i!=axis]
            radius=np.linalg.norm(rotation[:,radial]*half[radial],axis=1)
            axial=rotation[:,axis]*half[axis]
            if part['kind']=='cylinder':
                extent=np.abs(axial)+radius;lo,hi=center-extent,center+extent
            elif part['kind']=='cone':
                lo=center+np.minimum(axial,-axial-radius)
                hi=center+np.maximum(axial,-axial+radius)
            else:raise ValueError('Unknown physical primitive')
        bounds.append((lo,hi))
    lo=np.min([x[0] for x in bounds],axis=0);hi=np.max([x[1] for x in bounds],axis=0)
    return np.array(list(itertools.product(*zip(lo,hi))))


class RuntimeJudge:
    """Runtime observations must come from Isaac rigid views and contact reports.

    ``body_states`` maps semantic names to position_m, orientation_wxyz,
    linear_velocity_m_s, angular_velocity_rad_s. ``gripper_contacts`` and
    ``robot_contacts`` are semantic object-name sets from actual contact pairs.
    ``lid_angles_rad`` comes from measured lids/PhysX, never target commands.
    """
    def __init__(self, manifest, robot_prim_paths=(), gripper_prim_paths=()):
        self.manifest = deepcopy(manifest)
        self.robot_paths, self.gripper_paths = tuple(robot_prim_paths), tuple(gripper_prim_paths)
        self.judge = Task1Judge()
        self.elapsed = 0.
        self.last_states = {}
        self.contacts = []
        self.head_camera_coverage = None
        self.lid_angles = {color: 0. for color in manifest["bins"]}
        self.lid_geometry = {}

    @staticmethod
    def _matches(path, roots):
        return any(path == root or path.startswith(root+"/") for root in roots)

    @staticmethod
    def _body_matrix(raw):
        if "world_matrix" in raw:
            matrix = np.asarray(raw["world_matrix"], dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("invalid measured body world matrix")
            return matrix
        position = raw.get("position_world_m", raw.get("position_m"))
        quat = raw.get("quaternion_wxyz", raw.get("orientation_wxyz"))
        points = transform_corners(np.eye(3), [0, 0, 0], quat)
        matrix = np.eye(4)
        matrix[:3, :3] = points.T
        matrix[:3, 3] = position
        return matrix

    def observe(self, dt, body_states, contacts=(), lid_angles_rad=None, bin_fixed=True,
                gripper_contacts=None, robot_contacts=None):
        states = {}
        self.contacts = list(contacts)
        self.lid_geometry = {}
        for color, spec in self.manifest["bins"].items():
            raw = body_states.get(f"lid_{color}", body_states.get(f"{color}_lid"))
            if raw is not None:
                matrix = self._body_matrix(raw)
                self.lid_angles[color] = float(np.arctan2(matrix[2, 1], matrix[1, 1]))
                self.lid_geometry[f"lid_{color}"] = np.asarray(spec["lid_local_corners_m"])@matrix[:3, :3].T+matrix[:3, 3]
            elif lid_angles_rad is None:
                raise ValueError(f"Measured lid_{color} state or explicit measured lid_angles_rad is required")
        if lid_angles_rad is not None:
            self.lid_angles = {color: float(lid_angles_rad[color]) for color in self.manifest["bins"]}
        for key, spec in self.manifest["objects"].items():
            raw = body_states[key]
            matrix = self._body_matrix(raw)
            corners = primitive_world_bounds(spec["parts"], matrix)
            others = []
            for contact in contacts:
                if contact.get("point_count", 1) <= 0:
                    continue
                a, b = contact["body0_path"], contact["body1_path"]
                if self._matches(a, [spec["body_path"]]):
                    others.append(b)
                elif self._matches(b, [spec["body_path"]]):
                    others.append(a)
            gripped = (key in gripper_contacts) if gripper_contacts is not None else any(
                self._matches(path, self.gripper_paths) for path in others)
            touching = (key in robot_contacts) if robot_contacts is not None else any(
                self._matches(path, self.robot_paths) for path in others)
            support = [color for color, dest in self.manifest["bins"].items()
                       if any(self._matches(path, [dest["floor_path"]]) for path in others)]
            states[key] = PhysicalState(corners, raw["linear_velocity_m_s"], raw["angular_velocity_rad_s"],
                                        released=not gripped, touching_robot=touching,
                                        touching_gripper=gripped, support_destinations=support)
        bounds = {key: spec["destination_bounds_m"] for key, spec in self.manifest["bins"].items()}
        mouth = {key: spec["mouth_height_m"] for key, spec in self.manifest["bins"].items()}
        self.judge.observe(dt, states, bounds, self.manifest["table_surface_world_m"],
                           self.lid_angles, mouth, bin_fixed=bin_fixed)
        # Physics-engine execution demands contact-supported stability as well
        # as velocity/containment, so a floating frozen object cannot score.
        for key, state in states.items():
            if DESTINATIONS[key] not in state.support_destinations:
                self.judge.placement_valid[key] = False
                self.judge.stable[key] = 0.
        self.elapsed += dt
        self.last_states = states
        return self.rule_state()

    def rule_state(self):
        return {"engine": self.manifest["engine"], "schema_version": RULE_VERSION,
                "physics_time_seconds": float(self.elapsed), "policy_input_allowed": False,
                "objects": {key: state.serialize() for key, state in self.last_states.items()},
                "contacts": self.contacts, "lid_angles_rad": self.lid_angles,
                "score": self.judge.report(), "domain_parameters": self.manifest,
                "initial_head_camera_coverage": self.head_camera_coverage,
                "video_recording": False}

    report = rule_state

    def camera_geometry(self):
        points = deepcopy(self.manifest["geometry_world_m"])
        points.update({key: state.corners for key, state in self.last_states.items()})
        points.update(self.lid_geometry)
        return points


def expert_plan(manifest):
    """Declarative robot-only skill plan; a controller must realize every step.

    This is not a successful demonstration. Runtime may not realize these
    instructions by editing object/lid poses or by applying joint drives to lids.
    """
    plan = []
    for color, arm in (("dry", "left"), ("wet", "right")):
        spec = manifest["bins"][color]
        plan.append({"skill": "contact_open_lid", "arm": arm, "target_body_path": spec["lid_path"],
                     "handle_in_lid_m": spec["handle_in_lid_m"], "hinge_world_m": spec["hinge_world_m"],
                     "angle_target_rad": -1.68, "contact_required": True})
    for key, spec in manifest["objects"].items():
        color = spec["destination"]
        bounds = np.array(manifest["bins"][color]["destination_bounds_m"])
        xyz = bounds.mean(0)
        if color == "dry":
            xyz[0] += -.055 if key == "milk_carton" else .055
        xyz[2] = manifest["bins"][color]["mouth_height_m"]+.12
        plan.extend([{"skill": "grasp_and_lift", "object": key, "arm": spec["expert_arm"],
                      "grasp_point_local_m": spec["grasp_point_local_m"], "lift_m": .20},
                     {"skill": "hold_actual_grasp", "object": key, "arm": spec["expert_arm"], "seconds": 5.10},
                     {"skill": "release_above_open_bin", "object": key, "arm": spec["expert_arm"],
                      "destination": color, "object_target_position_m": xyz.tolist(),
                      "tcp_rotation_world": spec["drop_tcp_world_rotation"],
                      "tcp_position_world_m": spec["drop_tcp_position_world_m"]},
                     {"skill": "retreat_and_settle", "arm": spec["expert_arm"], "seconds": 1.20}])
    return plan


def lid_arc_world_poses(bin_spec, closed_gripper_world_matrix, samples=80, rotate_gripper_with_lid=None):
    """Minimum-jerk Cartesian rigid-hinge trajectory for a robot controller.

    Returns desired gripper matrices only; applying these matrices to lids or
    objects would violate the expert protocol. Robot IK must track them using
    joint drive targets and physical gripper contact.
    """
    if samples < 2:
        raise ValueError("lid arc requires at least two samples")
    start = np.asarray(closed_gripper_world_matrix, dtype=float)
    if start.shape != (4, 4) or not np.isfinite(start).all():
        raise ValueError("invalid closed gripper transform")
    anchor = np.asarray(bin_spec["hinge_world_m"])
    if rotate_gripper_with_lid is None:
        rotate_gripper_with_lid = bin_spec.get("expert_open_arc", {}).get("rotate_gripper_with_lid", False)
    relative = start[:3, 3]-anchor
    result = []
    for phase in np.linspace(0., 1., samples):
        blend = 10*phase**3-15*phase**4+6*phase**5
        angle = -1.68*blend
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        matrix = np.eye(4)
        matrix[:3, 3] = anchor+rotation@relative
        matrix[:3, :3] = rotation@start[:3, :3] if rotate_gripper_with_lid else start[:3, :3]
        result.append(matrix)
    return result
