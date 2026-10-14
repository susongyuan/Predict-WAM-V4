"""CPU self-collision screening from the exact official Panda convex colliders.

FCL only screens candidate configurations/paths. Actual execution and all scores
remain in Isaac PhysX; no object or articulation pose is changed by this module.
"""
from __future__ import annotations
import itertools
import numpy as np

class PandaSelfCollision:
    def __init__(self,kinematics,clearance=.003):
        import fcl,trimesh
        from pxr import Usd,UsdGeom,UsdPhysics
        from harness.zhuhai_robot_spec import ASSET
        self.fcl=fcl;self.kinematics=kinematics;self.clearance=float(clearance);self.parts=[]
        stage=Usd.Stage.Open(str(ASSET));root=stage.GetDefaultPrim()
        root.GetVariantSets().GetVariantSet('Colliders').SetVariantSelection('convex_hulls')
        world=lambda p:np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            body=prim.GetParent()
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            name=body.GetName()
            if name not in {f'panda_link{i}' for i in range(8)}|{'panda_hand'}:continue
            local=np.linalg.inv(world(body))@world(prim)
            if prim.IsA(UsdGeom.Mesh):
                vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
                vertices=vertices@local[:3,:3].T+local[:3,3]
                hull=trimesh.convex.convex_hull(vertices)
                shape=fcl.BVHModel();shape.beginModel(len(hull.vertices),len(hull.faces))
                shape.addSubModel(np.asarray(hull.vertices,dtype=float),np.asarray(hull.faces,dtype=np.int32));shape.endModel()
                local=np.eye(4)
            elif prim.IsA(UsdGeom.Capsule):
                capsule=UsdGeom.Capsule(prim);scales=np.linalg.norm(local[:3,:3],axis=0)
                if not np.allclose(scales,scales[0],atol=1e-6):raise ValueError('Unsupported nonuniform capsule scale')
                shape=fcl.Capsule(float(capsule.GetRadiusAttr().Get()*scales[0]),float(capsule.GetHeightAttr().Get()*scales[0]))
                local[:3,:3]/=scales
                axis=capsule.GetAxisAttr().Get()
                if axis=='X':local[:3,:3]=local[:3,:3]@np.array([[0,0,1],[0,1,0],[-1,0,0.]])
                elif axis=='Y':local[:3,:3]=local[:3,:3]@np.array([[1,0,0],[0,0,1],[0,-1,0.]])
            else:raise ValueError(f'Unsupported Panda arm collider {prim.GetPath()}')
            self.parts.append((name,local,fcl.CollisionObject(shape)))
        # The task's wrist sensor case is fixed to the real hand body.
        forward=np.array([-.07,0,.18]);forward/=np.linalg.norm(forward)
        right=np.array([0.,-1.,0.]);rotation=np.column_stack([right,np.cross(right,forward),-forward])
        local=np.eye(4);local[:3,:3]=rotation;local[:3,3]=[.07,0,.025];local[:3,3]+=rotation@np.array([0,0,.024])
        self.parts.append(('panda_hand',local,fcl.CollisionObject(fcl.Box(.038,.026,.018))))
        adjacent={frozenset((f'panda_link{i}',f'panda_link{i+1}')) for i in range(7)}
        adjacent.add(frozenset(('panda_link7','panda_hand')))
        self.pairs=[(i,j) for i,j in itertools.combinations(range(len(self.parts)),2)
                    if self.parts[i][0]!=self.parts[j][0] and frozenset((self.parts[i][0],self.parts[j][0])) not in adjacent]
        self.calls=0;self.last_pair=None

    def collision_free(self,arm,q):
        self.calls+=1;transforms=self.kinematics.link_transforms(arm,q);fcl=self.fcl
        for name,local,obj in self.parts:
            t=transforms[name]@local;obj.setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
        for i,j in self.pairs:
            distance=fcl.distance(self.parts[i][2],self.parts[j][2],fcl.DistanceRequest(),fcl.DistanceResult())
            if distance<self.clearance:
                self.last_pair=(self.parts[i][0],self.parts[j][0],float(distance));return False
        self.last_pair=None;return True

    def path_free(self,arm,start,target):
        start,target=np.asarray(start),np.asarray(target)
        count=max(5,min(65,int(np.ceil(np.max(np.abs(target-start))/.08))+1))
        return all(self.collision_free(arm,start+u*(target-start)) for u in np.linspace(0,1,count))
