"""Physical task-2 prototype using common tableware and fixed Aloha-AgileX.

Life mapping defaults to the PDF body text. The scoring-table conflict is
reported separately. All object pose writes occur during initialization;
expert execution moves only the robot. No evaluation-only camera is created.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import sapien
import transforms3d as t3d
import trimesh

from ._base_task import Base_Task
from .zhuhai_camera import ZhuhaiOnboardCamera
from .utils import Actor, ArmTag
from harness.zhuhai_task2_rules import PhysicalState, Task2Judge, PRODUCTION_SPECS

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets/objects"
UPRIGHT_Q = np.array([.5, .5, .5, .5])


@lru_cache(maxsize=24)
def _mesh_bounds_and_hull(filename):
    mesh = trimesh.load(filename, force="mesh")
    return mesh.bounds.copy(), mesh.convex_hull.vertices.copy()


class _ObservedScene:
    def __init__(self, scene, callback):
        self._scene, self._callback = scene, callback

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        result = self._scene.step()
        self._callback()
        return result


class zhuhai_task2(Base_Task):
    def setup_demo(self, **kwargs):
        self.zhuhai_track = kwargs.get("zhuhai_track", "life")
        self.zhuhai_mapping = kwargs.get("zhuhai_life_mapping") or "body_text"
        if self.zhuhai_track == "life" and self.zhuhai_mapping not in {"body_text", "scoring_table"}:
            raise ValueError("Unknown task-2 life mapping")
        self.zhuhai_randomized = bool(kwargs.get("zhuhai_randomized", False))
        self.zhuhai_seed = int(kwargs.get("seed", 0))
        self.zhuhai_source_seed = int(kwargs.get("zhuhai_source_seed", self.zhuhai_seed))
        self.zhuhai_cpu_only = False
        self.judge = None
        parameters = deepcopy(kwargs)
        parameters.update(eval_video_save_dir=None, eval_video_log=False, render_freq=0)
        parameters["data_type"] = dict(parameters.get("data_type") or {}, third_view=False, observer=False)
        super()._init_task_env_(**parameters)
        self._start_observer()
        geometry,entities = self.camera_geometry()
        self.head_camera_coverage = self.cameras.coverage_report(geometry,
            include_render_visibility=True,entities_by_name=entities)
        self.step_lim = int(kwargs.get("zhuhai_step_limit", 3000)) if self.eval_mode else None
        self.info["zhuhai_protocol"] = self.protocol()
        self.info["zhuhai_domain_parameters"] = self.domain_parameters()

    def load_camera(self, **kwargs):
        self.cameras = ZhuhaiOnboardCamera(bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis, **kwargs)
        self.cameras.load_camera(self.scene)
        self.scene.step()
        self.scene.update_render()

    def protocol(self):
        return {"track": self.zhuhai_track, "mapping": self.zhuhai_mapping,
            "robot": "fixed Aloha-AgileX with parallel grippers", "video_recording": False,
            "camera_scope": "head_camera,left_camera,right_camera only",
            "draft_discrepancy": "body text: bowl/spoon white, cup green, plate blue; scoring table: bowl/plate white, cup green, spoon blue",
            "asset_fidelity": "RoboTwin bowl/cup/plate meshes; Objaverse spoon/011 visual and decomposed collision mesh",
            "validation_scope": "declared simulation prototype; competition fixture unspecified",
            "competition_claim": False}

    def _box(self, name, position, half, color, static=True):
        entity = sapien.Entity()
        entity.set_name(name)
        body = sapien.physx.PhysxRigidStaticComponent() if static else sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half, self.scene.default_physical_material))
        entity.add_component(body)
        if not self.zhuhai_cpu_only:
            render = sapien.render.RenderBodyComponent()
            render.attach(sapien.render.RenderShapeBox(half, sapien.render.RenderMaterial(base_color=[*color, 1])))
            entity.add_component(render)
        entity.set_pose(sapien.Pose(position))
        self.scene.add_entity(entity)
        return entity

    def _add_bin(self, color, xy, half_xy=(.16, .14), height=.065):
        rgb = {"white": (.88, .88, .88), "green": (.10, .65, .15),
               "blue": (.12, .25, .85), "red": (.8, .1, .1)}[color]
        floor_z = .746 + self.table_z_bias
        x, y = xy
        a, b = half_xy
        self.bin_parts.append(self._box(f"{color}_floor", [x, y, floor_z-.003], [a+.005, b+.005, .003], rgb))
        for dx, dy, half in [(-a-.003, 0, [.003,b+.006,height/2]),
                              (a+.003, 0, [.003,b+.006,height/2]),
                              (0,-b-.003,[a,.003,height/2]), (0,b+.003,[a,.003,height/2])]:
            self.bin_parts.append(self._box(f"{color}_wall", [x+dx,y+dy,floor_z+height/2], half, rgb))
        # Open-top bins: planar containment is strict, protrusion up to 20cm allowed.
        self.interiors[color] = ([x-a,y-b,floor_z-.001], [x+a,y+b,floor_z+.20])
        self.bin_floors[color] = floor_z

    def _mesh_actor(self, key, xy, yaw, size_factor, mass, friction, model=None, model_id=None):
        if key == "spoon":
            folder = ASSET_ROOT / "objaverse/spoon/011"
            collision, visual = folder/"coacd_collision.obj", folder/"textured.obj"
            scale = np.full(3, .00115 * size_factor)
            config = {"scale": scale.tolist()}
            base_q = [1,0,0,0]
        else:
            folder = ASSET_ROOT / model
            collision, visual = folder/f"collision/base{model_id}.glb", folder/f"visual/base{model_id}.glb"
            config = json.loads((folder/f"model_data{model_id}.json").read_text())
            scale = np.asarray(config["scale"]) * size_factor
            config["scale"] = scale.tolist()
            base_q = UPRIGHT_Q
        bounds, hull = _mesh_bounds_and_hull(str(collision))
        vbounds, vhull = _mesh_bounds_and_hull(str(visual))
        lower = np.minimum(bounds[0], vbounds[0])
        upper = np.maximum(bounds[1], vbounds[1])
        local_points = np.vstack([hull, vhull]) * scale
        config["center"] = ((lower + upper)/2).tolist()
        config["extents"] = (upper-lower).tolist()
        yaw_q = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
        orientation = t3d.quaternions.qmult(yaw_q, base_q)
        rotation = t3d.quaternions.quat2mat(orientation)
        z = .74 + self.table_z_bias - np.min(local_points @ rotation.T, axis=0)[2] + .002
        material = self.scene.create_physical_material(friction, friction*.8, 0)
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(filename=str(collision), scale=scale, material=material)
        if not self.zhuhai_cpu_only:
            builder.add_visual_from_file(filename=str(visual), scale=scale)
        entity = builder.build(name=key)
        entity.set_pose(sapien.Pose([*xy,z], orientation))
        if key == "plate":
            contact = np.eye(4)
            contact[:3,3] = [0,.62,4.15]
            opposite = contact.copy(); opposite[2,3] = -4.15
            config["contact_points_pose"] = [contact.tolist(),opposite.tolist()]
        elif key == "spoon":
            contact = np.eye(4)
            contact[:3,:3] = [[1,0,0],[0,0,-1],[0,1,0]]
            contact[:3,3] = [-40,0,5]
            config["contact_points_pose"] = [contact.tolist()]
        functional = np.eye(4)
        functional[:3,:3] = t3d.quaternions.quat2mat(base_q).T
        functional[1 if key != "spoon" else 2,3] = lower[1 if key != "spoon" else 2]
        config["functional_matrix"] = [functional.tolist()]
        actor = Actor(entity, config, mass=mass)
        self.local_corners[key] = local_points
        self.object_parameters[key] = {"model": str(folder.relative_to(ASSET_ROOT)), "model_id": model_id,
            "mass_kg": float(mass), "static_friction": float(friction), "dynamic_friction": float(friction*.8),
            "scale": scale.tolist(), "size_factor": float(size_factor), "yaw_rad": float(yaw),
            "initial_pose_world_m": entity.get_pose().p.tolist(), "initial_quaternion_wxyz": entity.get_pose().q.tolist(),
            "collision_asset": str(collision), "visual_asset": str(visual), "geometry_hull_point_count": len(local_points)}
        return actor

    def _primitive(self, key, xy, dimensions, color, mass):
        half = np.asarray(dimensions)/2
        entity = self._box(key,[*xy,.742+self.table_z_bias+half[2]],half.tolist(),color,static=False)
        contact = np.eye(4); contact[:3,:3] = [[1,0,0],[0,0,-1],[0,1,0]]
        functional = np.eye(4); functional[2,3] = -half[2]
        actor = Actor(entity, {"scale":[1,1,1],"center":[0,0,0],"extents":list(dimensions),
            "contact_points_pose":[contact.tolist()],"functional_matrix":[functional.tolist()]},mass=mass)
        import itertools
        self.local_corners[key] = np.array([half*np.array(signs) for signs in itertools.product([-1,1],repeat=3)])
        self.object_parameters[key] = {"mass_kg":float(mass),"dimensions_m":list(dimensions),"proxy":True}
        return actor

    def load_actors(self):
        self.objects, self.interiors, self.bin_floors = {}, {}, {}
        self.bin_parts, self.local_corners, self.object_parameters = [], {}, {}
        jitter = .018 if self.zhuhai_randomized else .006
        yaw_range = np.deg2rad(18 if self.zhuhai_randomized else 3)
        if self.zhuhai_track == "life":
            for color, xy in zip(["white","green","blue"], [(-.34,.22),(0,.22),(.34,.22)]):
                self._add_bin(color,xy)
            definitions = {"bowl":("002_bowl",3,(-.20,-.25),.15),
                "cup":("021_cup",0,(.20,-.25),.10), "plate":("003_plate",0,(.20,-.005),.20),
                "spoon":(None,None,(-.20,-.025),.025)}
            for key,(model,model_id,xy,mass) in definitions.items():
                xy = np.asarray(xy)+np.random.uniform(-jitter,jitter,2)
                yaw = np.random.uniform(-yaw_range,yaw_range)
                factor = np.random.uniform(.95,1.05) if self.zhuhai_randomized else 1.0
                mass *= np.random.uniform(.8,1.2) if self.zhuhai_randomized else 1.0
                friction = np.random.uniform(.4,.8) if self.zhuhai_randomized else .6
                self.objects[key] = self._mesh_actor(key,xy,yaw,factor,mass,friction,model,model_id)
        elif self.zhuhai_track == "production":
            for color,xy in zip(["red","green","blue"],[(-.24,.12),(0,.12),(.24,.12)]):
                self._add_bin(color,xy,half_xy=(.09,.09),height=.005)
            for key,xy,color in [("cd294",(-.20,-.17),(.1,.1,.2)),
                                 ("inductor",(.20,-.17),(.3,.2,.1)), ("mkp62",(0,-.08),(.1,.2,.65))]:
                spec=PRODUCTION_SPECS[key]
                self.objects[key]=self._primitive(key,np.asarray(xy)+np.random.uniform(-jitter,jitter,2),
                    spec["dimensions_m"],color,np.random.uniform(*spec["mass_range_kg"]))
            self._add_bin("white",(0,-.14),half_xy=(.27,.105),height=.035)
        else:
            raise ValueError(self.zhuhai_track)
        self.initial_poses={key:actor.get_pose().p.tolist() for key,actor in self.objects.items()}
        for actor in self.objects.values():
            self.add_prohibit_area(actor,padding=.05)

    def _start_observer(self):
        self.judge=Task2Judge(self.zhuhai_track,self.zhuhai_mapping)
        alternative = "scoring_table" if self.zhuhai_mapping == "body_text" else "body_text"
        self.alternate_judge = Task2Judge("life",alternative) if self.zhuhai_track == "life" else None
        self.physics_time, self.last_physical_states, self.last_contacts = 0.0, {}, []
        self.scene=_ObservedScene(self.scene,self._observe_physics)
        self._observe_physics(dt=0.0)

    def _observe_physics(self, dt=None):
        if self.judge is None:
            return
        touching, touching_gripper, contacts = set(), set(), []
        supported = {key:set() for key in self.objects}
        gripper_names=set(self.robot.gripper_name)
        robot_names=set(gripper_names)
        for attribute in ("left_entity","right_entity"):
            entity=getattr(self.robot,attribute,None)
            if entity is not None:
                robot_names.update(link.get_name() for link in entity.get_links())
        object_entities = {actor.actor:key for key,actor in self.objects.items()}
        for contact in self.scene.get_contacts():
            if not contact.points:
                continue
            bodies=[contact.bodies[0].entity,contact.bodies[1].entity]
            for index,body in enumerate(bodies):
                key=object_entities.get(body)
                if key is None:
                    continue
                other=bodies[1-index]
                contacts.append({"object":key,"other_body":other.name,"point_count":len(contact.points),
                    "minimum_separation_m":min(float(p.separation) for p in contact.points)})
                if other.name in robot_names:
                    touching.add(key)
                if other.name in gripper_names:
                    touching_gripper.add(key)
                for color in self.interiors:
                    if other.name == f"{color}_floor":
                        supported[key].add(color)
        states={}
        for key,actor in self.objects.items():
            matrix=actor.get_pose().to_transformation_matrix()
            corners=self.local_corners[key] @ matrix[:3,:3].T+matrix[:3,3]
            body=actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            states[key]=PhysicalState(corners,body.linear_velocity,body.angular_velocity,
                released=key not in touching,upright_cosine=float(matrix[2,1] if key in {"cup","bowl","plate"} else matrix[2,2]),
                touching_robot=key in touching,touching_gripper=key in touching_gripper,
                support_destinations=sorted(supported[key]))
        dt=float(self.scene.get_timestep()) if dt is None else float(dt)
        self.physics_time+=dt
        self.last_physical_states,self.last_contacts=states,contacts
        self.judge.observe(dt,states,self.interiors,.74+self.table_z_bias)
        if self.alternate_judge is not None:
            self.alternate_judge.observe(dt,states,self.interiors,.74+self.table_z_bias)

    def domain_parameters(self):
        return {"seed":self.zhuhai_seed,"source_family_seed":self.zhuhai_source_seed,
            "randomized":self.zhuhai_randomized,"objects":self.object_parameters,
            "table_surface_world_m":.74+self.table_z_bias,"table_height_bias_m":float(self.table_z_bias),
            "destination_interiors_world_m":self.interiors,
            "source_ranges":{"position_jitter_m":.006,"yaw_degrees":3,"size_factor":1,"mass_factor":1,"static_friction":.6},
            "randomized_ranges":{"position_jitter_m":.018,"yaw_degrees":18,"size_factor":[.95,1.05],
                "mass_factor":[.8,1.2],"static_friction":[.4,.8]},
            "background":{"wall":getattr(self,"wall_texture",None),"table":getattr(self,"table_texture",None)},
            "lights":{"random_light":bool(getattr(self,"random_light",False)),
                "ambient":None if self.zhuhai_cpu_only else list(map(float,self.scene.ambient_light)),
                "point_colors":[list(map(float,light.color)) for light in getattr(self,"point_light_lst",[])],
                "direction_colors":[list(map(float,light.color)) for light in getattr(self,"direction_light_lst",[])]}}

    def camera_geometry(self):
        """Whole transformed collision/visual hulls and bin boundary points."""
        import itertools
        geometry={key:state.corners for key,state in self.last_physical_states.items()}
        entities={key:actor.actor for key,actor in self.objects.items()}
        for color in self.interiors:
            points=[]
            parts=[entity for entity in self.bin_parts if entity.name.startswith(color+'_')]
            for entity in parts:
                body=entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
                half=np.asarray(body.collision_shapes[0].half_size)
                matrix=entity.get_pose().to_transformation_matrix()
                points.extend([half*np.asarray(sign)@matrix[:3,:3].T+matrix[:3,3]
                    for sign in itertools.product([-1,1],repeat=3)])
            geometry['bin_'+color]=np.asarray(points)
            entities['bin_'+color]=parts
        return geometry,entities

    def rule_state(self):
        return {"physics_time_seconds":float(self.physics_time),
            "physics_timestep_seconds":float(self.scene.get_timestep()),
            "objects":{key:state.serialize() for key,state in self.last_physical_states.items()},
            "contacts":list(self.last_contacts),"destination_interiors_world_m":self.interiors,
            "table_surface_world_m":.74+self.table_z_bias,"score":self.judge.report(),
            "alternate_mapping_score":self.alternate_judge.report() if self.alternate_judge else None,
            "domain_parameters":self.domain_parameters(),"protocol":self.protocol(),
            "initial_head_camera_coverage":getattr(self,'head_camera_coverage',None),"policy_input_allowed":False}

    def _hold_seconds(self, seconds):
        for index in range(int(np.ceil(seconds/self.scene.get_timestep()))):
            self.scene.step()
            if self.save_freq and index%self.save_freq==0:
                self._update_render();self._take_picture()

    def play_once(self):
        for key,actor in self.objects.items():
            arm=ArmTag("left" if actor.get_pose().p[0]<0 else "right")
            ids={"bowl":[0,2][int(arm=="left")],"cup":[0,2][int(arm=="left")],"plate":None}.get(key)
            self.move(self.grasp_actor(actor,arm,pre_grasp_dis=.10,contact_point_id=ids))
            self.move(self.move_by_displacement(arm,z=.18))
            self._hold_seconds(5.10)
            if not self.plan_success:
                break
            color=self.judge.destinations[key]
            lo,hi=self.interiors[color]
            xy=(np.asarray(lo[:2])+hi[:2])/2
            quat=[1,0,0,0]
            if self.zhuhai_track=="life" and color=="white":
                xy[0]+=-.035 if key=="bowl" else .10
                if key=="spoon":
                    quat=[np.sqrt(.5),0,0,np.sqrt(.5)]
            self.move(self.place_actor(actor,arm,target_pose=[*xy,self.bin_floors[color]+.002,*quat],
                functional_point_id=0,constrain="free",pre_dis=.13,dis=.0,pre_dis_axis=[0,0,-1]))
            self.move(self.move_by_displacement(arm,z=.13))
            self.move(self.back_to_origin(arm))
            self._hold_seconds(1.10)
        self.info["zhuhai_task2_score"]=self.judge.report()
        self.info["info"]={"task":"zhuhai_task2","track":self.zhuhai_track,"mapping":self.zhuhai_mapping}
        return self.info

    def check_success(self):
        return self.judge is not None and self.judge.report()["full_task_success"]
