import ast, hashlib, json, os, sys, time
from pathlib import Path
R=Path('/home/user/Robotic/project2'); sys.path.insert(0,str(R))
from scripts import test_robotwin_eval_bridge as t
from scripts import eval_robotwin_wam as e
O=R/'outputs/validation/robotwin_eval_abort_fix_20261005'
m=json.loads((R/'outputs/runs/put_object_cabinet/wam_30k_20261005/manifest.json').read_text())
def walk(x):
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x and str(x['path']).endswith('.py'): yield x
  for v in x.values(): yield from walk(v)
 elif isinstance(x,list):
  for v in x: yield from walk(v)
rows=list(walk(m)); result=[]
for row in rows:
 p=Path(row['path']); h=hashlib.sha256(p.read_bytes()).hexdigest(); result.append(dict(path=str(p),expected=row['sha256'],actual=h,match=h==row['sha256']))
(O/'frozen_before.json').write_text(json.dumps(result,indent=2)); print('frozen',len(result),all(x['match'] for x in result))
tree=ast.parse((R/'scripts/eval_robotwin_wam.py').read_text()); main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'); fn=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='terminate')
ns={'InfrastructureError':e.InfrastructureError}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'legacy_signal','exec'),ns)
fallback=0
try: ns['terminate'](15,None)
except Exception: fallback+=50
print('legacy TOPP-shaped except Exception fallback steps',fallback)
args=e.parse_args(['--run-dir','/unused','--checkpoint','/unused','--training-run-dir','/unused','--plan-only']); args.episode_wall_seconds=.02
p=t.InjectedPolicy(); env=t.InjectedEnv()
def setup(**kw):
 try: time.sleep(.06)
 except: pass
 print('setup swallowed alarm and returned')
env.setup_demo=setup
out=O/'baseline_swallowed_setup'; out.mkdir()
r=e.run_evaluation(args,{'task_kwargs':{}},out,client_factory=lambda *a:p,env_factory=lambda:env,video_factory=lambda *a:(t.InjectedVideo(),None),owner_check=lambda *a:None,guard_factory=lambda:t.SimpleNamespace(check=lambda:{},peaks={}))
print('legacy result',r['status'],'actions',len(env.actions),'scored',r['scored_episodes'])
assert fallback==0, 'REGRESSION: cancellation swallowed by Exception fallback'
assert r['status']!='completed' and not env.actions, 'REGRESSION: swallowed timer allows expired actions/scoring'
