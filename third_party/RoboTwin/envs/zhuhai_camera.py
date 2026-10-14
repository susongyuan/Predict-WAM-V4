"""Shared three-sensor camera rig for Zhuhai prototypes, without observer cameras."""
from pathlib import Path
import numpy as np
import sapien
import yaml

from .camera import Camera
from ._GLOBAL_CONFIGS import CONFIGS_PATH

HEAD_POSITION = np.array([0., -.58, 1.72])
HEAD_TARGET = np.array([0., .03, .82])
HEAD_FOVY_DEGREES = 65.


def head_camera_transform(table_z_bias=0., position_offset=None):
    position = HEAD_POSITION.copy()
    position[2] += table_z_bias
    if position_offset is not None:
        position += position_offset
    target = HEAD_TARGET.copy(); target[2] += table_z_bias
    forward = target-position; forward /= np.linalg.norm(forward)
    left = np.array([-1.,0.,0.])
    up = np.cross(forward,left); up /= np.linalg.norm(up)
    transform = np.eye(4)
    transform[:3,:3] = np.stack([forward,left,up],axis=1)
    transform[:3,3] = position
    return transform


def declared_projection_report(points_by_name, transform, width=320, height=240, fovy_degrees=HEAD_FOVY_DEGREES):
    """Pinhole frustum coverage; this alone cannot verify occlusion."""
    focal = height/(2*np.tan(np.deg2rad(fovy_degrees)/2))
    result = {}
    for name,points in points_by_name.items():
        points = np.asarray(points,dtype=float).reshape(-1,3)
        local = (points-transform[:3,3]) @ transform[:3,:3]
        depth = local[:,0]
        u = width/2 - focal*local[:,1]/np.maximum(depth,1e-9)
        v = height/2 - focal*local[:,2]/np.maximum(depth,1e-9)
        inside = (depth>.1)&(depth<100)&(u>=0)&(u<width)&(v>=0)&(v<height)
        margins = np.minimum.reduce([u,width-1-u,v,height-1-v])
        result[name] = {"all_points_in_frame":bool(len(points) and np.all(inside)),
            "point_count":len(points),"in_frame_fraction":float(np.mean(inside)) if len(points) else 0.,
            "minimum_border_margin_pixels":float(np.min(margins)) if len(points) else None,
            "pixel_bounds_uv":[[float(np.min(u)),float(np.min(v))],[float(np.max(u)),float(np.max(v))]] if len(points) else None,
            "minimum_depth_m":float(np.min(depth)) if len(points) else None}
    return {"camera_names":["head_camera","left_camera","right_camera"],
        "head_position_world_m":transform[:3,3].tolist(),"head_fovy_degrees":fovy_degrees,
        "head_resolution_hw":[height,width],"all_geometry_in_frame":all(x["all_points_in_frame"] for x in result.values()),
        "geometric_projection":result,"occlusion_verified":False,
        "scope":"full object/bin geometry projection; separate RGB/segmentation evidence required for visibility"}


class ZhuhaiOnboardCamera(Camera):
    def load_camera(self, scene):
        with open(Path(CONFIGS_PATH)/"_camera_config.yml") as handle:
            configs = yaml.safe_load(handle)
        self.static_camera_list,self.static_camera_name,self.static_camera_config = [],[],[]
        self.head_camera_id = None
        self.head_transform = None
        self.static_camera_info_list = [dict(name="head_camera")]
        self.static_camera_num = 1

        def add(name,kind,fovy=None):
            config = dict(configs[kind])
            if fovy is not None:
                config["fovy"] = fovy
            return scene.add_camera(name=name,width=config["w"],height=config["h"],
                fovy=np.deg2rad(config["fovy"]),near=.1,far=100),config

        if self.collect_wrist_camera:
            self.left_camera,_ = add("left_camera",self.wrist_camera_type)
            self.right_camera,_ = add("right_camera",self.wrist_camera_type)
        if self.collect_head_camera:
            camera,config = add("head_camera",self.head_camera_type,HEAD_FOVY_DEGREES)
            offset = np.zeros(3)
            if self.random_head_camera_dis:
                direction = np.random.randn(3)
                offset = direction/np.linalg.norm(direction)*np.random.uniform(0,self.random_head_camera_dis)
            self.head_transform = head_camera_transform(self.table_z_bias,offset)
            camera.entity.set_pose(sapien.Pose(self.head_transform))
            self.head_camera_id = 0
            self.static_camera_list.append(camera)
            self.static_camera_name.append("head_camera")
            self.static_camera_config.append(config)
        self._scene = scene

    def coverage_report(self, points_by_name, include_render_visibility=False, entities_by_name=None):
        if self.head_transform is None:
            return {"all_geometry_in_frame":False,"reason":"head camera disabled"}
        config = self.static_camera_config[self.head_camera_id]
        result = declared_projection_report(points_by_name,self.head_transform,
            config["w"],config["h"],config["fovy"])
        if include_render_visibility:
            self._scene.update_render()
            camera = self.static_camera_list[self.head_camera_id]
            camera.take_picture()
            rgb = (camera.get_picture("Color")[:,:,:3]*255).clip(0,255).astype(np.uint8)
            segmentation = camera.get_picture("Segmentation")
            visibility = {}
            for name,entity in (entities_by_name or {}).items():
                group = entity if isinstance(entity,(list,tuple)) else [entity]
                instances = [int(part.per_scene_id) for part in group]
                mask = np.isin(segmentation[:,:,1],instances)
                visibility[name] = {"entity_per_scene_ids":instances,"visible_pixels":int(np.sum(mask)),
                    "visible":bool(np.any(mask))}
            result.update(render_visibility=visibility,occlusion_verified=bool(visibility),
                all_named_entities_visible=bool(visibility) and all(v["visible"] for v in visibility.values()),
                rgb_shape=list(rgb.shape),rgb_nonconstant=bool(np.std(rgb)>1),
                render_visibility_scope="actual head RGB and segmentation; some visible pixels does not prove full unoccluded surface")
        return result

    def get_observer_rgb(self):
        raise RuntimeError("Evaluation-only cameras are disabled")

    def get_world_pcd(self):
        raise RuntimeError("World cameras are disabled; use onboard observations")
