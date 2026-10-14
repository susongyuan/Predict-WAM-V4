#!/usr/bin/env python3
"""Explicit cabinet-only cleanup, after verified evidence preservation; never process-wide cleanup."""
from pathlib import Path
import hashlib,json,shutil,subprocess,datetime,sys
ROOT=Path(__file__).resolve().parents[2]; A=Path(__file__).resolve().parent
EXPLICIT=['data/benchmarks/robotwin2/put_object_cabinet_500_v1','data/benchmarks/robotwin2/put_object_cabinet_randomized',
 'data/robotwin_features/2bd59a493c3c9c49d797a3d0842e23c712da0cd8b551ebf3373533786c5ec267',
 'outputs/runs/put_object_cabinet','outputs/training/cabinet_vtt_adapter_1000_20261006',
 'outputs/training/cabinet_vtt_adapter_smoke_20261006','outputs/training/wrist_target_negative_weight2_20261006',
 'outputs/visualizations/system10_success_20261006','outputs/visualizations/wam_native_success_20261005']
# Existing archived cabinet-specific diagnostics; generic RoboTwin assets/data preparation remain.
PREFIXES=('cabinet_','middle_','integrated_','system_comparison_','feedback_','target_','train32_','train455_',
          'vtt_retrieval_','wam_grasp_','wam_visual_','wrist_','robotwin_cabinet_', 'robotwin_start_states')
EXPLICIT += [str(p.relative_to(ROOT)) for p in (ROOT/'outputs/validation').iterdir() if p.is_dir() and p.name.startswith(PREFIXES)]
def main():
 manifest=json.loads((A/'source_manifest.json').read_text())
 for r in manifest['files']:
  p=A/r['archived']
  if hashlib.sha256(p.read_bytes()).hexdigest()!=r['archive_sha256']:raise RuntimeError(f'archive changed: {p}')
 # Refuse cleanup if an actual local cabinet training/evaluator command still references its run/data.
 blockers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().split(b'\0')
  except (OSError,ProcessLookupError):continue
  if any(Path(x.decode(errors='replace')).name in {'eval_robotwin_wam.py','robotwin_policy_worker.py','train_robotwin_wam.py','run_robotwin_middle_suite.py'} for x in args if x):
   if any(b'put_object_cabinet' in x or b'middle_comparison10_20261006' in x for x in args):blockers.append(str(p))
 if blockers:raise RuntimeError(f'cabinet jobs still alive: {blockers}')
 entries=[]; retained=[]
 for rel in sorted(set(EXPLICIT)):
  p=ROOT/rel
  if not p.exists():continue
  if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise RuntimeError(f'unsafe root {p}')
  if not any(r['source'].startswith(rel+'/') for r in manifest['files']):
   retained.append({'path':rel,'reason':'no preserved metadata; conservative retention'});continue
  files=[x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
  entries.append({'path':rel,'files':len(files),'logical_bytes':sum(x.stat().st_size for x in files),
    'disk_bytes':int(subprocess.check_output(['du','-sB1',str(p)]).split()[0])})
 record={'status':'planned','timezone':'Asia/Shanghai','timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
   'archive_verified_files':len(manifest['files']),'retained_without_metadata':retained,'targets':entries,'logical_bytes':sum(x['logical_bytes'] for x in entries),
   'disk_bytes':sum(x['disk_bytes'] for x in entries),'not_touched':['other session processes/files','models/','environments','RoboTwin shared assets','outputs/training/task090_v3_50k_20260930_s050000','fruit candidate data'],
   'irreversible_note':'Old cabinet checkpoints and full cache removed; metadata and selected evidence retained, not a fully restorable training run.'}
 (A/'cleanup_plan.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 if '--execute' not in sys.argv:print(json.dumps(record,ensure_ascii=False,indent=2));return
 record['free_bytes_before']=shutil.disk_usage(ROOT).free;record['removed']=[]
 for e in entries:
  shutil.rmtree(ROOT/e['path']);record['removed'].append(e['path'])
  (A/'cleanup_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 record.update(status='completed',free_bytes_after=shutil.disk_usage(ROOT).free,all_target_paths_absent=all(not (ROOT/e['path']).exists() for e in entries))
 (A/'cleanup_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ('targets','removed')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
