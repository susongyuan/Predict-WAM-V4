"""Expert-only arm/hand support screening from the actual scene USD cubes.

This reads geometry and evaluates FK/FCL copies. It never changes USD or PhysX
poses. The unchanged PhysX monitor remains the execution safety authority.
"""
import hashlib
import json
import numpy as np
from harness.isaac_zhuhai_collision import PandaSelfCollision

VERSION = 'panda-expert-fixed-support-screen-v1'


def support_geometry(stage, paths):
    from pxr import UsdGeom, UsdPhysics
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('Support screening requires the unchanged metre scene')
    rows = []
    for path in sorted(paths):
        prim = stage.GetPrimAtPath(path)
        if (not str(path).startswith('/World/FixedSupport/') or not prim
                or not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Cube)):
            raise ValueError('Every fixed support collider must be an actual USD cube')
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=float).T
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        rotation = matrix[:3, :3]/scale
        if (not np.isfinite(matrix).all() or (scale <= 0).any()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1.)):
            raise ValueError('Unsupported fixed support transform')
        sides = float(UsdGeom.Cube(prim).GetSizeAttr().Get())*scale
        rows.append(dict(path=str(path), sides_m=sides.tolist(),
                         center_world_m=matrix[:3, 3].tolist(), rotation=rotation.tolist()))
    if not rows or len({r['path'] for r in rows}) != len(paths):
        raise ValueError('Fixed support collider inventory is incomplete or duplicated')
    return rows


class PandaSupportCollision(PandaSelfCollision):
    def __init__(self, kinematics, stage, paths, support_clearance=.012):
        super().__init__(kinematics)
        self.geometry = support_geometry(stage, paths)
        self.support_clearance = float(support_clearance)
        if self.support_clearance < .003:
            raise ValueError('Support screening cannot remove the existing collision margin')
        self.supports = [(r['path'], self.fcl.CollisionObject(self.fcl.Box(*r['sides_m']),
            self.fcl.Transform(np.asarray(r['rotation']), np.asarray(r['center_world_m'])))) for r in self.geometry]
        self.geometry_sha256 = hashlib.sha256(json.dumps(self.geometry, sort_keys=True).encode()).hexdigest()
        self.support_rejections = 0

    def collision_free(self, arm, q):
        # The original self check also places every collider at its FK pose.
        if not super().collision_free(arm, q):
            return False
        for name, _, body in self.parts:
            if name == 'panda_link0':
                continue  # Same permitted base/support contact as PhysX.
            for path, support in self.supports:
                distance = self.fcl.distance(body, support, self.fcl.DistanceRequest(), self.fcl.DistanceResult())
                if distance < self.support_clearance:
                    self.support_rejections += 1
                    self.last_pair = (arm+'/'+name, path, float(distance))
                    return False
        self.last_pair = None
        return True

    def path_free(self, arm, start, target):
        start, target = np.asarray(start), np.asarray(target)
        count = max(5, int(np.ceil(np.max(np.abs(target-start))/.025))+1)
        if count > 257:
            raise ValueError('Support path exceeds the official joint travel range')
        return all(self.collision_free(arm, start+u*(target-start)) for u in np.linspace(0., 1., count))

    def report(self):
        return dict(version=VERSION, static_collider_count=len(self.geometry),
            support_geometry_sha256=self.geometry_sha256, support_geometry=self.geometry,
            self_clearance_m=self.clearance, support_clearance_m=self.support_clearance,
            maximum_path_joint_spacing_rad=.025, support_rejections=self.support_rejections,
            last_rejected_pair=self.last_pair,
            scope='expert arm/hand/wrist-housing against fixed supports; original self check retained; actual PhysX still checks every forbidden contact')
