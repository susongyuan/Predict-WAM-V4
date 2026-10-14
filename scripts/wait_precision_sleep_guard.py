#!/usr/bin/env python3
"""Keep the explicit task units awake; never starts/stops them or owns GPU."""
import argparse,re,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--units',nargs='+',required=True);args=p.parse_args()
assert all(re.fullmatch(r'p2-zhuhai-[A-Za-z0-9_.-]+\.service',u) for u in args.units)
while True:
    active=[]
    for unit in args.units:
        r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True,timeout=5)
        if r.returncode:raise RuntimeError('Task unit status unreadable')
        active.append(r.stdout.strip() in ('active','activating','reloading'))
    if not any(active):break
    time.sleep(30)
