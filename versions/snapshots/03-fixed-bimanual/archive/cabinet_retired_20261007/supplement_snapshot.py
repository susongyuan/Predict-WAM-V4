from pathlib import Path
import json,hashlib,datetime
A=Path('archive/cabinet_retired_20261007');m=json.loads((A/'source_manifest.json').read_text());seen={r['source'] for r in m['files']};refs={}
for f in (A/'source/outputs/runs/put_object_cabinet').glob('*/plan.json'):
 j=json.loads(f.read_text())
 for k in ['source_files','config_files','seed_files']:
  for r in j.get(k,[]):
   p=Path(r['path']);refs[str(p)]=p
# Tiny TRAIN VTT prototypes, not feature cache or checkpoint.
for p in Path('outputs/validation/cabinet_vtt_bank_20261006').glob('*.npy'):
 if p.stat().st_size<2*1024**2:refs[str(p)]=p
for p in refs.values():
 if not p.is_file():continue
 p=p.relative_to(Path.cwd()) if p.is_absolute() else p
 if str(p) in seen or p.stat().st_size>2*1024**2:continue
 b=p.read_bytes();dst=A/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b);h=hashlib.sha256(b).hexdigest()
 m['files'].append({'source':str(p),'archived':str(dst.relative_to(A)),'mode':'copy','source_bytes':len(b),'preserved_bytes':len(b),'preserved_sha256':h,'archive_bytes':len(b),'archive_sha256':h});seen.add(str(p))
(A/'source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
receipt={'recorded_at':datetime.datetime.now().astimezone().isoformat(),'run':'outputs/runs/put_object_cabinet/middle10_vtt_only_seed10012_20261006','effective_classification':'user_cancelled_interrupted','scored':False,'policy_failure':False,'reason':'User explicitly stopped suite first, then current vtt-only seed10012 job, to retire cabinet experiments. Raw evaluator labels infrastructure_error because cleanup failed after SIGTERM; do not count as policy failure.','source_result_status':'infrastructure_error','cancellation_reason':'evaluator received signal 15','reported_control_steps':115,'evidence':['stopped_units_before.txt','stopped_units_after.txt','source/outputs/runs/put_object_cabinet/middle10_vtt_only_seed10012_20261006/result.json','source/outputs/validation/middle_comparison10_20261006/queue_status.json'],'stop_order':['p2-middle10-suite-20261006','p2-middle10-vtt-only-10012-20261006'],'operator_report':'user message in current archival task; no independent stop command issued by archival agent','current_services':'Read-only systemctl check: no running p2 service; suite MainPID=0, failed/exit-code; trial MainPID=0 inactive/dead.'}
(A/'user_cancelled_interruption_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
import gzip
errs=[]
for r in m['files']:
 p=A/r['archived'];b=gzip.decompress(p.read_bytes()) if r['mode']=='gzip' else p.read_bytes()
 if hashlib.sha256(b).hexdigest()!=r['preserved_sha256'] or len(b)!=r['preserved_bytes']:errs.append(str(p))
v={'files_preserved':len(m['files']),'archive_bytes':sum(r['archive_bytes'] for r in m['files']),'verification_errors':errs,'all_source_metrics_ready_for_authorized_cleanup':not errs}
(A/'snapshot_verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
