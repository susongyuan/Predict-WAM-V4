"""Zhuhai life task-1 physical prototype, separate from frozen baselines.

Fixed hollow bins have unpowered revolute lids opened by robot contact. A
past-vertical joint stop holds the open lid mechanically. No object/joint pose
is set after initialization. Banana peel is a declared rigid three-petal proxy.
"""
import itertools

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import Actor, ArmTag, create_actor
from .zhuhai_camera import ZhuhaiOnboardCamera
from harness.zhuhai_task1_rules import PhysicalState, Task1Judge


class _ObservedScene:
    def __init__(self, scene, callback):
        self._scene, self._callback = scene, callback

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        result = self._scene.step()
        self._callback()
        return result


def _metadata(extents, contact_xyz=(0, 0, 0), center=(0, 0, 0)):
    contacts = []
    for rotation in ([[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                     [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                     [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]):
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, contact_xyz
        contacts.append(matrix.tolist())
    return {"center": list(center), "extents": list(extents), "scale": [1, 1, 1],
            "contact_points_pose": contacts, "functional_matrix": [np.eye(4).tolist()],
            "target_pose": [np.eye(4).tolist()], "transform_matrix": np.eye(4).tolist()}


class zhuhai_task1(Base_Task):
    def load_camera(self, **kwargs):
        self.cameras = ZhuhaiOnboardCamera(bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis, **kwargs)
        self.cameras.load_camera(self.scene)

    def setup_demo(self, **kwargs):
        if not kwargs.get("zhuhai_allow_unvalidated_probe", False):
            raise RuntimeError("Task-1 prototype needs an explicit unvalidated-probe flag before collection")
        if kwargs.get("zhuhai_track", "life") != "life":
            raise ValueError("This scene implements life task-1 only")
        self.zhuhai_randomized = bool(kwargs.get("zhuhai_randomized", False))
        self.judge = None
        super()._init_task_env_(**kwargs)
        self.judge = Task1Judge()
        self.physics_time, self.last_contacts, self.last_physical_states = 0.0, [], {}
        self.scene = _ObservedScene(self.scene, self._observe_physics)
        self.step_lim = int(kwargs.get("zhuhai_step_limit", 4000)) if self.eval_mode else None
        self.info["zhuhai_protocol"] = {
            "track": "life", "robot": "fixed Aloha-AgileX with parallel grippers",
            "video_recording": False, "competition_claim": False,
            "asset_fidelity": "parametric milk carton/squeeze bottle; rigid three-petal banana-peel proxy; existing tableware meshes",
            "lid_mechanism": "fixed-root hollow bin, unpowered contact-opened hinge, passive past-vertical stop",
            "lid_final_closed_required": False,
            "initial_relation": "all three garbage items start inside supporting tableware",
            "layout": "workspace-adjusted prototype; official fixture dimensions unspecified",
        }

    def _shape(self, builder, xyz, half, rgb, quat=(1, 0, 0, 0), collision=True):
        pose = sapien.Pose(xyz, quat)
        if collision:
            builder.add_box_collision(pose=pose, half_size=half, material=self.scene.default_physical_material)
        builder.add_box_visual(pose=pose, half_size=half, material=[*rgb, 1])

    def _bin(self, color, xy):
        a, b, height, thick = .145, .105, .170, .004
        rgb = (.28, .35, .44) if color == "dry" else (.15, .48, .23)
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_name(f"{color}_bin_fixed_body")
        self._shape(root, [0, 0, .003], [a + thick, b + thick, .003], rgb)
        for xyz, half in [([-a-thick, 0, height/2], [thick, b+thick, height/2]),
                          ([a+thick, 0, height/2], [thick, b+thick, height/2]),
                          ([0, -b-thick, height/2], [a, thick, height/2]),
                          ([0, b+thick, height/2], [a, thick, height/2])]:
            self._shape(root, xyz, half, rgb)
        lid = builder.create_link_builder(root)
        lid.set_name(f"{color}_bin_lid")
        lid.set_joint_name(f"{color}_bin_lid_hinge")
        lid.set_joint_properties("revolute", [[-1.72, 0.0]],
            sapien.Pose([0, b+thick, height+.004]), sapien.Pose([0, b+thick, 0]),
            friction=.002, damping=.025)
        self._shape(lid, [0, 0, 0], [a+thick, b+thick, .003], rgb)
        handle = [0, -b+.024, .023]
        self._shape(lid, handle, [.026, .014, .020], (.76, .77, .78))
        articulation = builder.build(fix_root_link=True)
        origin = np.array([*xy, .74+self.table_z_bias])
        articulation.set_root_pose(sapien.Pose(origin))
        articulation.set_qpos([0.0])
        articulation.set_qvel([0.0])
        joint = articulation.get_active_joints()[0]
        joint.set_drive_properties(stiffness=0.0, damping=.025)
        body = articulation.get_links()[0]
        lid_link = articulation.get_links()[1]
        lid_link.mass = .06
        # Link geometry and handle are one physical rigid link, not an attached
        # kinematic prop. The Actor wrapper supplies grasp candidate frames.
        grasp = Actor(lid_link.entity, _metadata([2*(a+thick), 2*(b+thick), .046], handle))
        self.bins[color] = articulation
        self.lid_actors[color] = grasp
        self.lid_hinges[color] = origin + np.array([0, b+thick, height+.004])
        self.bin_root_origins[color] = body.get_pose().p.copy()
        self.interiors[color] = (origin+[-a, -b, .006], origin+[a, b, height-.003])
        self.mouth_heights[color] = float(origin[2]+height)
        self.bin_dimensions[color] = (a, b, height)

    def _object(self, key, xy, support_z, scale):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        s = float(scale)
        if key == "milk_carton":
            self._shape(builder, [0, 0, 0], np.array([.0275, .0275, .048])*s, (.90, .91, .87))
            self._shape(builder, [0, 0, .053*s], np.array([.0275, .023, .008])*s, (.18, .45, .67))
            self._shape(builder, [0, -.0276*s, 0], np.array([.020, .0008, .018])*s, (.12, .42, .66), collision=False)
            extent, center, bottom, contact = np.array([.055, .057, .109])*s, [0, 0, .0065*s], -.048*s, [0, 0, 0]
            mass = .025
        elif key == "squeeze_bottle":
            # Empty squeezable container is modeled rigidly for the first probe.
            self._shape(builder, [0, 0, 0], np.array([.023, .023, .043])*s, (.76, .19, .13))
            self._shape(builder, [0, 0, .052*s], np.array([.012, .012, .009])*s, (.91, .88, .76))
            self._shape(builder, [0, 0, .068*s], np.array([.007, .007, .007])*s, (.87, .70, .21))
            extent, center, bottom, contact = np.array([.046, .046, .118])*s, [0, 0, .016*s], -.043*s, [0, 0, .012*s]
            mass = .03
        else:
            self._shape(builder, [0, 0, .008*s], np.array([.021, .017, .011])*s, (.91, .77, .18))
            for angle in (0, 2*np.pi/3, -2*np.pi/3):
                xyz = [.030*np.cos(angle)*s, .030*np.sin(angle)*s, 0]
                quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
                self._shape(builder, xyz, np.array([.028, .011, .004])*s, (.85, .70, .16), quat)
            extent, center, bottom, contact = np.array([.120, .108, .030])*s, [0, 0, .007*s], -.004*s, [0, 0, .012*s]
            mass = .012
        entity = builder.build(name=key)
        entity.set_pose(sapien.Pose([*xy, support_z-bottom+.002]))
        actor = Actor(entity, _metadata(extent, contact, center), mass=mass)
        self.objects[key] = actor

    def load_actors(self):
        self.objects, self.bins, self.lid_actors, self.supports = {}, {}, {}, {}
        self.interiors, self.mouth_heights, self.lid_hinges = {}, {}, {}
        self.bin_root_origins, self.bin_dimensions = {}, {}
        for color, xy in [("dry", (-.255, .19)), ("wet", (.255, .19))]:
            self._bin(color, xy)
        jitter = .020 if self.zhuhai_randomized else .008
        defs = {"milk_carton": ("003_plate", 0, (-.24, -.20), .751),
                "squeeze_bottle": ("002_bowl", 3, (0, -.15), .754),
                "banana_peel": ("003_plate", 0, (.24, -.20), .751)}
        for key, (model, model_id, xy, floor) in defs.items():
            xy = np.asarray(xy)+np.random.uniform(-jitter, jitter, 2)
            support = create_actor(self, sapien.Pose([*xy, .74], [.5, .5, .5, .5]),
                                   modelname=model, model_id=model_id, convex=True)
            support.set_name(f"{key}_source_tableware")
            support.set_mass(.35)
            self.supports[key] = support
            scale = np.random.uniform(.90, 1.10) if self.zhuhai_randomized else 1.0
            self._object(key, xy, floor+self.table_z_bias, scale)
        self.local_corners = {}
        for key, actor in self.objects.items():
            extent, center = np.asarray(actor.config["extents"]), np.asarray(actor.config["center"])
            self.local_corners[key] = np.array([center+extent/2*np.array(sign)
                                      for sign in itertools.product([-1, 1], repeat=3)])
        self.initial_poses = {key: actor.get_pose().p.tolist() for key, actor in self.objects.items()}
        for actor in self.objects.values():
            self.add_prohibit_area(actor, padding=.045)

    def _observe_physics(self):
        if self.judge is None:
            return
        touching, gripped, contacts = set(), set(), []
        gripper_names = set(self.robot.gripper_name)
        robot_entities = {int(link.entity.per_scene_id) for robot in (self.robot.left_entity, self.robot.right_entity)
                          for link in robot.get_links()}
        for contact in self.scene.get_contacts():
            if not contact.points:
                continue
            entities = [body.entity for body in contact.bodies]
            for key, actor in self.objects.items():
                if actor.actor in entities:
                    other = entities[1] if entities[0] is actor.actor else entities[0]
                    contacts.append({"object": key, "other_body": other.name,
                                     "point_count": len(contact.points)})
                    if other.name in gripper_names:
                        gripped.add(key)
                    if int(other.per_scene_id) in robot_entities:
                        touching.add(key)
        states = {}
        for key, actor in self.objects.items():
            matrix = actor.get_pose().to_transformation_matrix()
            corners = self.local_corners[key] @ matrix[:3, :3].T + matrix[:3, 3]
            body = actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            states[key] = PhysicalState(corners, body.linear_velocity, body.angular_velocity,
                                       released=key not in gripped, touching_robot=key in touching)
        angles = {color: float(art.get_qpos()[0]) for color, art in self.bins.items()}
        fixed = all(np.linalg.norm(art.get_links()[0].get_pose().p-self.bin_root_origins[color]) < .001
                    for color, art in self.bins.items())
        dt = float(self.scene.get_timestep())
        self.physics_time += dt
        self.last_physical_states, self.last_contacts = states, contacts
        self.judge.observe(dt, states, self.interiors, .74+self.table_z_bias,
                           angles, self.mouth_heights, bin_fixed=fixed)

    def rule_state(self):
        return {"physics_time_seconds": self.physics_time,
                "physics_timestep_seconds": float(self.scene.get_timestep()),
                "objects": {key: state.serialize() for key, state in self.last_physical_states.items()},
                "contacts": list(self.last_contacts),
                "lid_angles_rad": {color: float(art.get_qpos()[0]) for color, art in self.bins.items()},
                "destination_interiors_world_m": {color: [lo.tolist(), hi.tolist()]
                    for color, (lo, hi) in self.interiors.items()},
                "score": self.judge.report(), "policy_input_allowed": False}

    def camera_coverage(self, include_render_visibility=False):
        """Complete conservative object bounds plus actual head visibility on request."""
        points, entities = {}, {}
        for key, actor in {**self.objects, **{f"{k}_source_tableware": v for k, v in self.supports.items()}}.items():
            matrix = actor.get_pose().to_transformation_matrix()
            scale = np.asarray(actor.config["scale"])
            center = np.asarray(actor.config.get("center", [0, 0, 0]))*scale
            extent = np.asarray(actor.config["extents"])*scale
            local = np.array([center+extent/2*np.array(sign)
                              for sign in itertools.product([-1, 1], repeat=3)])
            points[key] = local @ matrix[:3, :3].T + matrix[:3, 3]
            entities[key] = actor.actor
        for color, art in self.bins.items():
            a, b, h = self.bin_dimensions[color]
            root, lid = art.get_links()
            for name, link, center, half in [(f"{color}_bin_body", root, [0, 0, h/2], [a+.008, b+.008, h/2]),
                                             (f"{color}_bin_lid", lid, [0, 0, .020], [a+.004, b+.004, .026])]:
                local = np.array([np.array(center)+np.array(half)*np.array(sign)
                                  for sign in itertools.product([-1, 1], repeat=3)])
                matrix = link.get_pose().to_transformation_matrix()
                points[name] = local @ matrix[:3, :3].T + matrix[:3, 3]
                entities[name] = link.entity
        return self.cameras.coverage_report(points, include_render_visibility=include_render_visibility,
                                            entities_by_name=entities)

    def _hold_seconds(self, seconds):
        for index in range(int(np.ceil(seconds/self.scene.get_timestep()))):
            self.scene.step()
            if self.save_freq and index % self.save_freq == 0:
                self._update_render()
                self._take_picture()

    def _open_lid(self, color, arm):
        self.move(self.grasp_actor(self.lid_actors[color], arm, pre_grasp_dis=.10))
        if not self.plan_success:
            return
        # Track the rigid body's measured hinge arc with the held gripper.
        # Only robot EE targets are sent; the articulation is never driven/reset.
        hinge = self.lid_hinges[color]
        ee = np.asarray(self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose())
        relative = ee[:3]-hinge
        for angle in np.linspace(-.10, -1.68, 18):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            from transforms3d.quaternions import mat2quat, quat2mat
            target = [*(hinge+rotation@relative), *mat2quat(rotation@quat2mat(ee[3:]))]
            self.move(self.move_to_pose(arm, target))
            if not self.plan_success:
                return
        self.move(self.open_gripper(arm))
        self.move(self.move_by_displacement(arm, y=-.10, z=.04))
        self.move(self.back_to_origin(arm))
        self._hold_seconds(.4)

    def play_once(self):
        self._open_lid("dry", ArmTag("left"))
        self._open_lid("wet", ArmTag("right"))
        for key, actor in self.objects.items():
            if not self.plan_success:
                break
            arm = ArmTag("right" if key == "banana_peel" else "left")
            self.move(self.grasp_actor(actor, arm, pre_grasp_dis=.10))
            self.move(self.move_by_displacement(arm, z=.20))
            self._hold_seconds(5.10)
            color = self.judge.destinations[key]
            lo, hi = self.interiors[color]
            xy = (lo[:2]+hi[:2])/2
            if color == "dry":
                xy[0] += -.055 if key == "milk_carton" else .055
            # Release above the open mouth, then gravity carries the item inside.
            # This avoids pushing a held gripper below a high wall.
            target_z = self.mouth_heights[color] + .11
            self.move(self.place_actor(actor, arm, [*xy, target_z, 1, 0, 0, 0],
                constrain="free", pre_dis=.12, dis=.0, pre_dis_axis=[0, 0, -1]))
            self.move(self.move_by_displacement(arm, z=.10))
            self.move(self.back_to_origin(arm))
            self._hold_seconds(1.20)
        self.info["zhuhai_task1_score"] = self.judge.report()
        self.info["info"] = {"task": "zhuhai_task1", "track": "life"}
        return self.info

    def check_success(self):
        return self.judge is not None and self.judge.report()["full_task_success"]
