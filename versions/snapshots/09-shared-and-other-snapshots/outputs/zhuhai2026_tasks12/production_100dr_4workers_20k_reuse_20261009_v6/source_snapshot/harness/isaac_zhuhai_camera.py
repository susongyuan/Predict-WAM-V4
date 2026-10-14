"""USD/Isaac Sim 6 three-camera onboard rig and auditable head visibility.

Import is CPU safe. Render-product attachment requires a running SimulationApp.
No observer or world camera is authored, created, or rendered.
"""
from __future__ import annotations
import math
import numpy as np

HEAD_POSITION = np.array([0., -.58, 1.85])
HEAD_TARGET = np.array([0., .03, .82])
HEAD_FOVY_DEGREES = 65.
RESOLUTION = (320,240)


def head_pose(table_z_bias=0., offset=None):
    position=HEAD_POSITION.copy();position[2]+=table_z_bias
    if offset is not None:position+=np.asarray(offset)
    target=HEAD_TARGET.copy();target[2]+=table_z_bias
    forward=target-position;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    matrix=np.eye(4);matrix[:3,:3]=np.stack([right,up,-forward],axis=1);matrix[:3,3]=position
    return matrix


def projection_report(points_by_name, world_matrix, resolution=RESOLUTION, fovy=HEAD_FOVY_DEGREES):
    width,height=resolution;focal=height/(2*math.tan(math.radians(fovy)/2));items={}
    for name,points in points_by_name.items():
        points=np.asarray(points).reshape(-1,3)
        local=(points-world_matrix[:3,3])@world_matrix[:3,:3]
        depth=-local[:,2];safe=np.maximum(depth,1e-9)
        u=width/2+focal*local[:,0]/safe;v=height/2-focal*local[:,1]/safe
        inside=(depth>.1)&(depth<100)&(u>=0)&(u<width)&(v>=0)&(v<height)
        margin=np.minimum.reduce([u,width-1-u,v,height-1-v])
        items[name]={"all_points_in_frame":bool(len(points) and np.all(inside)),"point_count":len(points),
            "in_frame_fraction":float(np.mean(inside)) if len(points) else 0.,
            "minimum_border_margin_pixels":float(np.min(margin)) if len(points) else None,
            "pixel_bounds_uv":[[float(np.min(u)),float(np.min(v))],[float(np.max(u)),float(np.max(v))]] if len(points) else None}
    return {"camera_names":["head_camera","left_camera","right_camera"],"head_position_world_m":world_matrix[:3,3].tolist(),
        "head_fovy_degrees":fovy,"head_resolution_hw":[height,width],"geometry":items,
        "all_geometry_in_frame":bool(items) and all(item['all_points_in_frame'] for item in items.values()),
        "occlusion_verified":False,"scope":"geometry projection only until actual RGB and semantic segmentation are captured"}


class IsaacZhuhaiCameraRig:
    def __init__(self,stage,root='/World/Robot/OnboardSensors',table_z_bias=0.,seed=0,randomized=False):
        self.stage,self.root=stage,root
        rng=np.random.default_rng(seed+78439)
        offset=rng.uniform(-.012,.012,3) if randomized else np.zeros(3)
        self.head_world_matrix=head_pose(table_z_bias,offset)
        self.paths={name:f'{root}/{name}' for name in ['head_camera','left_camera','right_camera']}
        self.render_products,self.annotators={},{}
        self.frame_evidence=None

    def build(self,left_wrist_parent=None,right_wrist_parent=None):
        from pxr import Gf,UsdGeom
        for name,path in list(self.paths.items()):
            parent=left_wrist_parent if name=='left_camera' else right_wrist_parent if name=='right_camera' else None
            if parent:
                path=f'{parent}/{name}';self.paths[name]=path
            camera=UsdGeom.Camera.Define(self.stage,path)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.025,20))
            camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            vertical_aperture=24.
            fovy=HEAD_FOVY_DEGREES if name=='head_camera' else 65.
            camera.CreateVerticalApertureAttr(vertical_aperture)
            camera.CreateHorizontalApertureAttr(vertical_aperture*RESOLUTION[0]/RESOLUTION[1])
            camera.CreateFocalLengthAttr(vertical_aperture/(2*math.tan(math.radians(fovy)/2)))
            if name=='head_camera':self.set_world_pose(name,self.head_world_matrix)
            elif not parent:
                # Runtime must update these poses from actual robot wrist links.
                matrix=np.eye(4);matrix[:3,3]=[-.2 if name=='left_camera' else .2,-.3,1.0]
                self.set_world_pose(name,matrix)
        return {'camera_paths':dict(self.paths),'resolution_wh':list(RESOLUTION),'camera_count':3,
            'head_world_matrix':self.head_world_matrix.tolist(),'head_fovy_degrees':HEAD_FOVY_DEGREES,
            'head_mount':'fixed robot mast at y=-.58m','video_recording':False,
            'wrist_mounting_verified':bool(left_wrist_parent and right_wrist_parent)}

    def set_world_pose(self,name,matrix):
        from pxr import Gf,UsdGeom
        matrix=np.asarray(matrix,dtype=float)
        prim=self.stage.GetPrimAtPath(self.paths[name])
        xform=UsdGeom.Xformable(prim)
        transform=Gf.Matrix4d(*matrix.T.reshape(-1).tolist())
        if prim.GetParent().GetPath().pathString!='/World':
            parent_matrix=np.asarray(UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)).T
            transform=Gf.Matrix4d(*(np.linalg.inv(parent_matrix)@matrix).T.reshape(-1).tolist())
        xform.ClearXformOpOrder();xform.AddTransformOp().Set(transform)
        if name=='head_camera':self.head_world_matrix=matrix.copy()

    def attach_render_products(self):
        import omni.replicator.core as rep
        for name,path in self.paths.items():
            product=rep.create.render_product(path,RESOLUTION)
            self.render_products[name]=product
            rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
            self.annotators[name]={'rgb':rgb}
            if name=='head_camera':
                segmentation=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False})
                segmentation.attach(product);self.annotators[name]['semantic_segmentation']=segmentation
        return {name:str(product.path) for name,product in self.render_products.items()}

    def capture(self,points_by_name,required_labels=None):
        result=projection_report(points_by_name,self.head_world_matrix)
        images={name:np.asarray(values['rgb'].get_data())[:,:,:3].astype(np.uint8) for name,values in self.annotators.items()}
        if set(images)!=set(self.paths):raise RuntimeError('Exactly three RGB sensors must be attached')
        data=self.annotators['head_camera']['semantic_segmentation'].get_data()
        labels=data.get('info',{}).get('idToLabels',{}) if isinstance(data,dict) else {}
        segmentation=np.asarray(data.get('data')) if isinstance(data,dict) else np.asarray(data)
        visibility={}
        for name in required_labels or points_by_name:
            ids=[]
            for identifier,description in labels.items():
                values=description.values() if isinstance(description,dict) else [description]
                flattened=[]
                for value in values:
                    flattened.extend(value if isinstance(value,(list,tuple)) else [value])
                if name in flattened:ids.append(int(identifier))
            pixels=int(np.sum(np.isin(segmentation,ids))) if ids else 0
            visibility[name]={'semantic_ids':ids,'visible_pixels':pixels,'visible':pixels>0}
        result.update(occlusion_verified=bool(visibility),render_visibility=visibility,
            all_named_entities_visible=bool(visibility) and all(item['visible'] for item in visibility.values()),
            rgb_shapes={name:list(value.shape) for name,value in images.items()},
            rgb_nonconstant={name:bool(np.std(value)>1) for name,value in images.items()},
            render_scope='actual head RGB+semantic pixels; nonzero pixels verifies visibility, not every surface unoccluded')
        self.frame_evidence=result
        return images,result

    def close(self):
        for values in self.annotators.values():
            for annotator in values.values():annotator.detach()
        for product in self.render_products.values():product.destroy()
        self.annotators.clear();self.render_products.clear()
