#!/usr/bin/env python3
"""Convert the frozen Aloha URDF into an isolated Isaac Sim 6.1 PhysX asset."""
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.progress1_log import log
from scripts.gpu_lock import owner

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gpu-lock-owner',required=True)
    args=ap.parse_args()
    if owner()!=args.gpu_lock_owner:raise RuntimeError('GPU lock owner mismatch')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
    tree=ET.parse(source)
    robot=tree.getroot()
    removed_links=set()
    for joint in list(robot.findall('joint')):
        if joint.get('name') in {'lr_base_joint','rr_base_joint'}:
            removed_links.add(joint.find('child').get('link'))
            robot.remove(joint)
    expanded=True
    while expanded:
        expanded=False
        for joint in list(robot.findall('joint')):
            if joint.find('parent').get('link') in removed_links:
                removed_links.add(joint.find('child').get('link'));robot.remove(joint);expanded=True
    for link in list(robot.findall('link')):
        if link.get('name') in removed_links:robot.remove(link)
    # The source is a mobile multi-arm assembly. This task uses a fixed dual-arm
    # rig: retain the exact two arm chains, replace the chassis/old cameras with
    # an explicitly authored fixed mechanical frame in the runtime.
    kept_joints={'fl_base_joint','fr_base_joint'}|{f'{p}_joint{i}' for p in ('fl','fr') for i in range(1,9)}
    kept_links={'footprint'}
    for joint in list(robot.findall('joint')):
        if joint.get('name') in kept_joints:
            kept_links.add(joint.find('parent').get('link'));kept_links.add(joint.find('child').get('link'))
        else:robot.remove(joint)
    for link in list(robot.findall('link')):
        if link.get('name') not in kept_links:
            removed_links.add(link.get('name'));robot.remove(link)
    for mesh in tree.findall('.//mesh'):
        path=Path(mesh.attrib['filename'])
        if not path.is_absolute():path=(source.parent/path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('filename',str(path))
    copied=out/'aloha_native14.urdf';tree.write(copied,encoding='utf-8',xml_declaration=True)
    result=dict(status='starting_isaac',engine={'name':'Isaac Sim','version':'6.1.0.0'},
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fixed_base=True,started_unix=time.time())
    result['removed_unused_rear_arm_links']=sorted(removed_links)
    result['robot_profile']='fixed two original ARX5 arm chains; mobile chassis and legacy camera branches removed; explicit fixed support frame in runtime'
    def write():
        temp=out/'result.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'result.json')
    write();log('Isaac6.1转换固定Aloha资产，原URDF不变，PID驱动参数明确',actor='isaac_asset')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':320,'height':240,'multi_gpu':False})
    try:
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf',True)
        from isaacsim.asset.importer.urdf import URDFImporter,URDFImporterConfig
        cfg=URDFImporterConfig(urdf_path=str(copied),usd_path=str(out/'usd'),fix_base=True,
            merge_fixed_joints=False,allow_self_collision=True,joint_drive_type='force',
            joint_target_type='position',override_joint_stiffness=1000.,override_joint_damping=200.)
        result['status']='converting';write()
        usd=URDFImporter(cfg).import_urdf()
        from pxr import Usd,UsdPhysics
        stage=Usd.Stage.Open(usd);default=stage.GetDefaultPrim()
        default.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
        stage.GetRootLayer().Save()
        joints=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        result.update(status='converted',usd=str(usd),default_prim=str(default.GetPath()),
                      joint_paths=joints,physics_variant='physx',seconds=time.time()-result['started_unix'])
        write();print(json.dumps(result),flush=True)
    except BaseException as exc:
        import traceback
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());write()
        raise
    finally:
        app.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
