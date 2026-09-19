"""Retain exact child identity for occupancy attribution; never time this wrapper."""
import argparse
import os
import re
import subprocess
import threading
import time
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
a=argparse.ArgumentParser();a.add_argument('--method',required=True);a.add_argument('--kind',required=True)
a.add_argument('--tool',choices=['none','memcheck','synccheck','nsys'],default='none');a.add_argument('--record-id',required=True);args=a.parse_args()
check_frozen();check_r1();check_r2();command=[PY,str(H/'src/run_slot_r2.py'),'--method',args.method,'--kind',args.kind,'--record-id',args.record_id]
if args.tool in ['memcheck','synccheck']:
    command=['compute-sanitizer','--tool',args.tool,'--error-exitcode','99']+(['--leak-check','full'] if args.tool=='memcheck' else [])+command
elif args.tool=='nsys':
    command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=false',
             '-o',str(H/'artifacts'/args.record_id)]+command
log=H/'raw'/f'{args.record_id}.log'
with log.open('x') as f:
    child=subprocess.Popen(command,cwd=H,env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT)
    watchdog=threading.Timer(900,child.kill);watchdog.start()
    try:os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT);time.sleep(1);child.wait()
    finally:watchdog.cancel()
text=log.read_text(errors='replace');inner=H/'results'/f'inner_{args.record_id}.json'
r=dict(command=command,returncode=child.returncode,method=args.method,kind=args.kind,tool=args.tool,
       log_sha256=sha(log),child_identity_hold_seconds=1)
passed=child.returncode==0 and inner.exists()
if inner.exists():
    r['inner_path']=str(inner.relative_to(P));r['inner_sha256']=sha(inner)
    passed=passed and json.loads(inner.read_text())['correctness']['exact_contract_pass']
if args.tool in ['memcheck','synccheck']:
    passed=passed and 'ERROR SUMMARY: 0 errors' in text and not re.search(r'ERROR SUMMARY: [1-9]|LEAK SUMMARY: [1-9]',text)
r['correctness']=dict(exact_contract_pass=bool(passed));r['performance_admitted']=False
write(H/'results'/f'{args.record_id}.json',r)
print(json.dumps(r),flush=True)
raise SystemExit(0 if passed else 2)
