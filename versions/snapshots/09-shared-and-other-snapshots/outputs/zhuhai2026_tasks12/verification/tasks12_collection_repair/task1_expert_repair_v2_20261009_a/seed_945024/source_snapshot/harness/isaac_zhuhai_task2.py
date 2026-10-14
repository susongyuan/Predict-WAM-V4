"""Isaac Sim 6 task-2 USD scene and strict PhysX evidence adapter.

Scene authoring is CPU safe with pxr. During execution this module never sets
object poses: runtime supplies actual PhysX states/contacts to the rule judge.
"""
from __future__ import annotations
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh

from harness.zhuhai_task2_rules import PhysicalState,Task2Judge

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'third_party/RoboTwin/assets/objects'
BODY_DESTINATIONS={'bowl':'white','spoon':'white','cup':'green','plate':'blue'}


def quaternion_matrix(q):
    q=np.asarray(q,dtype=float);q/=np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def box_corners(center,half):
    return np.array([np.asarray(center)+np.asarray(half)*np.asarray(s) for s in itertools.product([-1,1],repeat=3)])


@lru_cache(maxsize=20)
def load_mesh(filename):
    mesh=trimesh.load(filename,force='mesh')
    return np.asarray(mesh.vertices).copy(),np.asarray(mesh.faces).copy(),mesh.convex_hull.vertices.copy()


@lru_cache(maxsize=12)
def collision_parts(filename):
    scene=trimesh.load(filename,force='scene')
    meshes=list(scene.dump())
    parts=[]
    for mesh in meshes:
        mesh.merge_vertices()
        # CoACD meshes may contain disconnected convex components in one OBJ.
        pieces=mesh.split(only_watertight=False)
        for piece in pieces if len(pieces) else [mesh]:
            if len(piece.vertices)>=4:
                hull=piece.convex_hull
                parts.append((np.asarray(hull.vertices).copy(),np.asarray(hull.faces).copy()))
    if not parts:raise ValueError(f'No collision hulls: {filename}')
    return parts


def _semantics(prim,label):
    try:
        from pxr import UsdSemantics
        api=UsdSemantics.LabelsAPI.Apply(prim,'class')
        api.CreateLabelsAttr().Set([label])
    except ImportError:
        # Standard USD installations may lack Isaac's semantic schema.
        from pxr import Sdf
        prim.CreateAttribute('userProperties:semanticLabel',Sdf.ValueTypeNames.String).Set(label)


def _material(stage,path,color,friction=.6):
    from pxr import Gf,Sdf,UsdShade,UsdPhysics
    material=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/PreviewSurface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.25 if 'spoon' in path else .5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim());api.CreateStaticFrictionAttr().Set(friction)
    api.CreateDynamicFrictionAttr().Set(friction*.8);api.CreateRestitutionAttr().Set(0.)
    return material


def _box(stage,path,center,half,color,label=None):
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade
    cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center));cube.AddScaleOp().Set(Gf.Vec3f(*(np.asarray(half)*2)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mat=_material(stage,path+'_material',color)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if label:_semantics(cube.GetPrim(),label)
    return cube


def build(stage,root='/World/Task',seed=0,randomized=False,mapping='body_text',table_z_bias=0.,source_seed=None):
    """Return JSON-ready metadata used by runtime, scoring, and dataset manifests."""
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade
    if mapping not in {'body_text','scoring_table'}:raise ValueError(mapping)
    rng=np.random.default_rng(seed)
    UsdGeom.Xform.Define(stage,root)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    surface=.74+table_z_bias
    _box(stage,root+'/Table',[0,0,surface-.025],[.65,.45,.025],(.5,.4,.3),'table')
    bins={};geometry={};objects={}
    for color,xy in zip(['white','green','blue'],[(-.34,.02),(0,.02),(.34,.02)]):
        rgb={'white':(.88,.88,.88),'green':(.1,.65,.15),'blue':(.12,.25,.85)}[color]
        a,b,height=.16,.14,.065;x,y=xy;floor=surface+.006
        bin_root=f'{root}/Bins/{color}'
        parts=[('floor',[x,y,floor-.003],[a+.005,b+.005,.003])]
        for index,(dx,dy,half) in enumerate([(-a-.003,0,[.003,b+.006,height/2]),(a+.003,0,[.003,b+.006,height/2]),
                (0,-b-.003,[a,.003,height/2]),(0,b+.003,[a,.003,height/2])]):
            parts.append((f'wall_{index}',[x+dx,y+dy,floor+height/2],half))
        for name,position,half in parts:_box(stage,bin_root+'/'+name,position,half,rgb,'bin_'+color)
        bins[color]={'root_path':bin_root,'floor_path':bin_root+'/floor','floor_world_m':floor,
            'interior_min_world_m':[x-a,y-b,floor-.001],'interior_max_world_m':[x+a,y+b,floor+.20],
            'wall_height_m':height,'half_xy_m':[a,b], 'part_paths':[bin_root+'/'+part[0] for part in parts]}
        geometry['bin_'+color]=np.vstack([box_corners(p,h) for _,p,h in parts]).tolist()
    definitions={'bowl':('002_bowl',3,(-.34,-.27),.15,(.87,.83,.74)),
        'cup':('021_cup',0,(.10,-.27),.10,(.7,.15,.15)),
        'plate':('003_plate',0,(.36,-.285),.20,(.94,.94,.88)),
        'spoon':('objaverse/spoon/011',None,(-.10,-.27),.025,(.63,.66,.7))}
    for key,(model,model_id,xy,mass,color) in definitions.items():
        folder=ASSETS/model
        factor=float(rng.uniform(.95,1.05)) if randomized else 1.
        if key=='spoon':
            scale=np.full(3,.00115*factor);visual=folder/'textured.obj';collision=folder/'coacd_collision.obj'
            base_rotation=np.eye(3);up_axis=2
        else:
            config=json.loads((folder/f'model_data{model_id}.json').read_text())
            scale=np.asarray(config['scale'])*factor
            visual=folder/f'visual/base{model_id}.glb';collision=folder/f'collision/base{model_id}.glb'
            base_rotation=quaternion_matrix([.5,.5,.5,.5]);up_axis=1
        vertices,faces,visual_hull=load_mesh(str(visual))
        cvertices,cfaces,collision_hull=load_mesh(str(collision))
        complete_hull=np.vstack([visual_hull,collision_hull])*scale
        lower,upper=complete_hull.min(0),complete_hull.max(0)
        local_hull=np.asarray(trimesh.convex.convex_hull(complete_hull).vertices)
        jitter=.018 if randomized else .006
        xy=np.asarray(xy)+rng.uniform(-jitter,jitter,2)
        yaw_limit=(12 if key=='plate' else 18) if randomized else 3
        yaw=float(rng.uniform(-np.deg2rad(yaw_limit),np.deg2rad(yaw_limit)))
        yaw_matrix=quaternion_matrix([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        rotation=yaw_matrix@base_rotation
        position=np.array([*xy,surface-np.min(local_hull@rotation.T,axis=0)[2]+.002])
        mass=float(mass*(rng.uniform(.8,1.2) if randomized else 1.))
        friction=float(rng.uniform(.4,.8)) if randomized else .6
        path=root+'/Objects/'+key
        xform=UsdGeom.Xform.Define(stage,path)
        matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=position
        xform.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr().Set(mass)
        try:
            from pxr import PhysxSchema
            PhysxSchema.PhysxContactReportAPI.Apply(xform.GetPrim()).CreateThresholdAttr().Set(0.)
        except ImportError:pass
        material=_material(stage,root+'/Materials/'+key,color,friction)
        mesh=UsdGeom.Mesh.Define(stage,path+'/visual')
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices*scale]);mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist());mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);_semantics(mesh.GetPrim(),key)
        parts=collision_parts(str(collision))
        for index,(cv,cf) in enumerate(parts):
            child=UsdGeom.Mesh.Define(stage,path+f'/collision_{index:03d}')
            child.CreatePointsAttr([Gf.Vec3f(*v) for v in cv*scale]);child.CreateFaceVertexCountsAttr([3]*len(cf))
            child.CreateFaceVertexIndicesAttr(cf.reshape(-1).tolist());child.CreateSubdivisionSchemeAttr('none')
            child.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(child.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(child.GetPrim()).CreateApproximationAttr().Set('convexHull')
            UsdShade.MaterialBindingAPI.Apply(child.GetPrim()).Bind(material,materialPurpose='physics')
        _semantics(xform.GetPrim(),key)
        geometry[key]=(local_hull@rotation.T+position).tolist()
        objects[key]={'prim_path':path,'visual_path':path+'/visual','collision_paths':[path+f'/collision_{i:03d}' for i in range(len(parts))],
            'mass_kg':mass,'static_friction':friction,'dynamic_friction':friction*.8,'size_factor':factor,
            'scale':scale.tolist(),'model':model,'model_id':model_id,'yaw_rad':yaw,'up_axis_local':up_axis,
            'initial_position_world_m':position.tolist(),'initial_world_matrix':matrix.tolist(),
            'local_geometry_points_m':local_hull.tolist(),'local_corners_m':local_hull.tolist(),
            'geometry_bound_kind':'complete visual+collision convex hull vertices; actual transformed hull extrema, not fictitious rotated local-AABB corners',
            'visual_asset':str(visual),'collision_asset':str(collision),'collision_hull_count':len(parts)}
    return {'task':'zhuhai_task2','engine':'Isaac Sim 6 USD/PhysX','root_path':root,'seed':seed,
        'source_seed':seed if source_seed is None else source_seed,'randomized':randomized,'mapping':mapping,
        'objects':objects,'bins':bins,'geometry_world_m':geometry,'table_surface_world_m':surface,
        'table_prim_path':root+'/Table','table_height_bias_m':table_z_bias,
        'source_ranges':{'position_jitter_m':.006,'yaw_degrees':3,'size_factor':1.,'mass_factor':1.,'static_friction':.6},
        'randomized_ranges':{'position_jitter_m':.018,'yaw_degrees':18,'per_object_yaw_degrees':{'plate':12},
            'size_factor':[.95,1.05],'mass_factor':[.8,1.2],'static_friction':[.4,.8]},
        'video_recording':False,'camera_count_required':3,'competition_claim':False,
        'draft_conflict':'body_text bowl/spoon white, cup green, plate blue; scoring_table bowl/plate white, cup green, spoon blue',
        'runtime_admission':'actual PhysX contacts+velocity+full hull containment+continuous5s grasp+released1s stable+uprightcup'}


class RuntimeJudge:
    def __init__(self,manifest,robot_prim_paths=(),gripper_prim_paths=()):
        self.manifest=manifest;self.robot_paths=tuple(robot_prim_paths);self.gripper_paths=tuple(gripper_prim_paths)
        self.judge=Task2Judge('life',manifest['mapping'])
        self.alternate_judge=Task2Judge('life','scoring_table' if manifest['mapping']=='body_text' else 'body_text')
        self.time=0.;self.states={};self.contacts=[];self.head_camera_coverage=None;self.last_dt=0.
        self.interiors={color:(spec['interior_min_world_m'],spec['interior_max_world_m']) for color,spec in manifest['bins'].items()}
        self.local_hulls={key:np.asarray(spec['local_geometry_points_m']) for key,spec in manifest['objects'].items()}

    @staticmethod
    def _matches(path,roots):return any(path==root or path.startswith(root+'/') for root in roots)

    def observe(self,dt,body_states,contacts):
        """Contact dictionaries: body0_path/body1_path/point_count; real engine evidence."""
        states={};self.contacts=list(contacts)
        for key,spec in self.manifest['objects'].items():
            measured=body_states[key]
            if 'world_matrix' in measured:matrix=np.asarray(measured['world_matrix'],dtype=float)
            else:
                matrix=np.eye(4);matrix[:3,:3]=quaternion_matrix(measured['quaternion_wxyz']);matrix[:3,3]=measured['position_world_m']
            geometry=self.local_hulls[key]@matrix[:3,:3].T+matrix[:3,3]
            # Extrema AFTER rotating the actual hull are exactly sufficient
            # for an axis-aligned container. Do not rotate a local AABB.
            lo,hi=geometry.min(0),geometry.max(0)
            geometry=box_corners((lo+hi)/2,(hi-lo)/2)
            others=[]
            for contact in contacts:
                if contact.get('point_count',1)<=0:continue
                a,b=contact['body0_path'],contact['body1_path']
                if self._matches(a,[spec['prim_path']]):others.append(b)
                elif self._matches(b,[spec['prim_path']]):others.append(a)
            touching=any(self._matches(path,self.robot_paths) for path in others)
            gripper=any(self._matches(path,self.gripper_paths) for path in others)
            support=[color for color,s in self.manifest['bins'].items() if any(self._matches(path,[s['floor_path']]) for path in others)]
            states[key]=PhysicalState(geometry,measured['linear_velocity_m_s'],measured['angular_velocity_rad_s'],
                released=not touching,upright_cosine=float(matrix[2,spec['up_axis_local']]),touching_robot=touching,
                touching_gripper=gripper,support_destinations=support)
        self.judge.observe(dt,states,self.interiors,self.manifest['table_surface_world_m'])
        self.alternate_judge.observe(dt,states,self.interiors,self.manifest['table_surface_world_m'])
        self.time+=dt;self.states=states;self.last_dt=dt
        return self.report()

    def camera_geometry(self):
        result={k:s.corners for k,s in self.states.items()}
        result.update({k:v for k,v in self.manifest['geometry_world_m'].items() if k.startswith('bin_')})
        return result

    def report(self):
        compact_parameters={k:v for k,v in self.manifest.items() if k not in {'objects','geometry_world_m'}}
        compact_parameters['objects']={key:{k:v for k,v in spec.items() if k not in {'local_geometry_points_m','local_corners_m'}}
            for key,spec in self.manifest['objects'].items()}
        return {'physics_time_seconds':float(self.time),'physics_timestep_seconds':float(self.last_dt),
            'objects':{k:v.serialize() for k,v in self.states.items()},
            'contacts':self.contacts,'score':self.judge.report(),'alternate_mapping_score':self.alternate_judge.report(),
            'destination_interiors_world_m':self.interiors,'table_surface_world_m':self.manifest['table_surface_world_m'],
            'initial_head_camera_coverage':self.head_camera_coverage,'domain_parameters':compact_parameters,
            'engine':'Isaac Sim 6 USD/PhysX','policy_input_allowed':False,'video_recording':False}

    rule_state=report
