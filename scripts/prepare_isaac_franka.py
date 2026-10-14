#!/usr/bin/env python3
"""Cache a bounded, byte-preserving official Isaac 6.1 Franka asset, CPU only."""
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/FrankaEmika/'
OUT=ROOT/'data/isaac_zhuhai/franka_official_v1'
MAX_BYTES=512*1024**2

def main():
    from pxr import Sdf, Usd
    OUT.mkdir(parents=True,exist_ok=True)
    pending={'franka_panda.usda'}; seen={}; builtins=set(); total=0
    def fetch(rel):
        p=(OUT/rel).resolve()
        if not p.is_relative_to(OUT.resolve()): raise ValueError(f'Unbounded dependency {rel}')
        url=urljoin(BASE,rel)
        if not url.startswith(BASE): raise ValueError(url)
        if p.exists(): b=p.read_bytes()
        else:
            with urlopen(url,timeout=45) as response:
                b=response.read(MAX_BYTES+1)
            if len(b)>MAX_BYTES: raise ValueError('Asset is above bounded download budget')
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(b)
        return rel,p,b,url
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            current=sorted(pending);pending=set()
            for rel,p,b,url in pool.map(fetch,current):
                total+=len(b)
                if total>MAX_BYTES:raise RuntimeError('Asset dependency budget exceeded')
                seen[rel]={'url':url,'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
                if p.suffix.lower() in {'.usd','.usda','.usdc'}:
                    layer=Sdf.Layer.FindOrOpen(str(p))
                    if not layer:raise RuntimeError(f'Unreadable official layer {p}')
                    for asset in set(re.findall(r'@([^@]+)@',layer.ExportToString())):
                        if asset in {'OmniPBR.mdl','OmniGlass.mdl','UsdPreviewSurface'}:
                            builtins.add(asset);continue
                        dep=urljoin(url,asset)
                        if not dep.startswith(BASE):raise RuntimeError(f'External dependency requires review: {dep}')
                        relative=unquote(dep[len(BASE):])
                        if relative and relative not in seen:pending.add(relative)
                print(f'cached {rel} {len(b)} bytes',flush=True)
            pending-=seen.keys()
    report={'status':'cached_official_asset','source_base_url':BASE,'asset_version':'Isaac 6.1',
        'asset':'data/isaac_zhuhai/franka_official_v1/franka_panda.usda','variant_selection':{'Colliders':'convex_hulls','Physics':'physx'},
        'bytes':total,'files':seen,'runtime_builtins':sorted(builtins),
        'scope':'official MuJoCo-Menagerie-derived kinematic/inertial/collision asset; runtime joint drives and fixed mounts audited separately',
        'reference':'https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/franka.py'}
    path=OUT/'asset_manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=report:raise RuntimeError('Refuse to overwrite different official asset receipt')
    else:
        with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'files':len(seen),'bytes':total,'manifest':str(path)}))


# Optional CPU audit also used to generate the exact USD-based FK contract.
def audit_kinematics():
    import numpy as np
    from pxr import Usd,UsdPhysics
    sys.path.insert(0,str(ROOT))
    from harness.zhuhai_robot_spec import ASSET,KINEMATIC_CHAIN,JOINT_LIMITS,TOOL_FROM_HAND
    stage=Usd.Stage.Open(str(ASSET));root=stage.GetDefaultPrim()
    root.GetVariantSets().GetVariantSet('Colliders').SetVariantSelection('convex_hulls')
    def quat(q):return [q.GetReal(),*q.GetImaginary()] if q is not None else [1,0,0,0]
    def frames(joint):
        return {'parent_frame':{'xyz':list(joint.GetLocalPos0Attr().Get()),'wxyz':quat(joint.GetLocalRot0Attr().Get())},
                'child_frame':{'xyz':list(joint.GetLocalPos1Attr().Get()),'wxyz':quat(joint.GetLocalRot1Attr().Get())}}
    byname={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    chain=[]
    for i in range(1,8):
        p=byname[f'panda_joint{i}'];joint=UsdPhysics.Joint(p)
        limits=[p.GetAttribute('physics:lowerLimit').Get(),p.GetAttribute('physics:upperLimit').Get()]
        assert np.max(np.abs(np.deg2rad(limits)-JOINT_LIMITS[i-1]))<2e-6
        chain.append(dict(name=p.GetName(),axis=p.GetAttribute('physics:axis').Get(),
                          limits_rad=JOINT_LIMITS[i-1],**frames(joint)))
    hand=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint) and '/panda_hand/' in str(p.GetPath()))
    masses=[]
    for p in stage.Traverse():
        # Zero-mass pad colliders are parts of their parent finger, not links.
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):continue
        mass=p.GetAttribute('physics:mass').Get();inertia=p.GetAttribute('physics:diagonalInertia').Get()
        if mass is not None and inertia is not None:
            a=np.asarray(inertia)
            valid=bool(mass>0 and (a>0).all() and a.max()<=a.sum()-a.max()+1e-8)
            if not valid:raise ValueError(f'Nonphysical link inertial tensor: {p.GetPath()}')
            masses.append({'link':p.GetName(),'mass_kg':mass,'principal_inertia_kg_m2':list(inertia),'positive_and_triangle_valid':valid})
    report={'asset_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),'joints':chain,
        'hand_fixed_joint':frames(UsdPhysics.Joint(hand)),'tool_from_hand':TOOL_FROM_HAND,
        'bodies':masses,'audit':'7 canonical Panda joint limits verified; rigid-link masses/inertia positive and triangle valid; no scaling'}
    if KINEMATIC_CHAIN.exists():
        if json.loads(KINEMATIC_CHAIN.read_text())!=report:raise RuntimeError('Refuse to replace a different kinematic contract')
    else:
        with KINEMATIC_CHAIN.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    return report

if __name__=='__main__':
    main()
    audit_kinematics()
