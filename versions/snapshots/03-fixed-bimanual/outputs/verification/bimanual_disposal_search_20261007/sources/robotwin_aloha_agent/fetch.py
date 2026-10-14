import requests,pathlib,concurrent.futures,json,re,html
D=pathlib.Path('/tmp/robotwin_aloha_research'); jobs={}
for branch,paths in {
'RoboTwin-1.0':['README.md','envs/put_apple_cabinet.py','envs/base_task.py','envs/utils/create_actor.py'],
'early_version':['README.md','envs/apple_cabinet_storage.py'],
'main':['README.md','envs/put_object_cabinet.py','envs/move_can_pot.py','envs/place_object_basket.py','envs/place_cans_plasticbox.py','envs/open_microwave.py','envs/_base_task.py','envs/utils/create_actor.py','envs/camera/camera.py','scripts/requirements.txt','assets/_download.py'],
'WBCD-2026':['README.md','envs/packing.py'],
'CVPR-Challenge-2025-Round1':['README.md','envs/utils/ipc_create_actor.py'],
}.items():
 for p in paths:jobs[branch+'__'+p.replace('/','__')]='https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/'+branch+'/'+p

def get(kv):
 k,u=kv;r=requests.get(u,timeout=40);(D/(k+'.txt')).write_text(r.text);return k,r.status_code,len(r.content)
for x in concurrent.futures.ThreadPoolExecutor(max_workers=10).map(get,jobs.items()):print(x)
for f in ['unleashed_project','aloha_project']:
 s=(D/(f+'.txt')).read_text();s=re.sub(r'<(script|style).*?</\1>','',s,flags=re.S);print('\n===',f,html.unescape(re.sub('<[^>]*>',' ',s)))
 print('LINKS',re.findall(r'href=["\']([^"\']+)',s))
