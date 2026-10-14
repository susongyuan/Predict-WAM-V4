import requests,json,pathlib,concurrent.futures
D=pathlib.Path('/tmp/robotwin_aloha_research');repo='RoboTwin-Platform/RoboTwin'
branches=requests.get(f'https://api.github.com/repos/{repo}/branches',params={'per_page':100},timeout=30).json();(D/'branches.json').write_text(json.dumps(branches))
def scan(b):
 name=b['name'];sha=b['commit']['sha'];r=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=60)
 (D/(name.replace('/','_')+'_tree.json')).write_text(r.text)
 if not r.ok:return name,r.status_code
 paths=[x['path'] for x in r.json()['tree'] if x['type']=='blob'];task=[p for p in paths if p.startswith(('envs/','task/','tasks/')) and p.endswith('.py')]
 return name,task,[p for p in paths if any(k in p.lower() for k in ['readme','download','data_config','_config.yaml','requirements','install'])][:80]
for result in concurrent.futures.ThreadPoolExecutor(max_workers=8).map(scan,branches): print(json.dumps(result))
