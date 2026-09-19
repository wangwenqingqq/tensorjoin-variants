"""Collect bounded actual-launch binding / one selected cuBLAS NCU function."""
import argparse
import os
import subprocess
import threading
import time
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3
PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
a=argparse.ArgumentParser();a.add_argument('--method',required=True);a.add_argument('--record-id',required=True);a.add_argument('--ncu',action='store_true');args=a.parse_args()
check_frozen();check_r1();check_r2();check_r3()
for path,value in json.loads((H/'artifacts/frozen_runtime_probe.json').read_text()).items():assert sha(P/path)==value,path
cmd=[PY,str(H/'src/runtime_probe.py'),'--method',args.method,'--record-id',args.record_id]
if args.ncu:
    cmd=['ncu','--nvtx','--nvtx-include','G19_BIND_fp32/','--kernel-name','regex:magma_sgemmEx_kernel',
         '--launch-count','1','--section','LaunchStats','--section','SourceCounters','--export',
         str(H/'artifacts'/args.record_id),'--page','raw','--csv']+cmd
log=H/'raw'/f'{args.record_id}.log'
with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT)
    timer=threading.Timer(900,child.kill);timer.start()
    try:os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT);time.sleep(1);child.wait()
    finally:timer.cancel()
inner=H/'results'/f'inner_{args.record_id}.json'
passed=child.returncode==0 and inner.exists() and json.loads(inner.read_text())['correctness']['exact_contract_pass']
if args.ncu:passed=passed and (H/'artifacts'/f'{args.record_id}.ncu-rep').is_file()
write(H/'results'/f'{args.record_id}.json',dict(command=cmd,returncode=child.returncode,inner_path=str(inner.relative_to(P)),
    inner_sha256=sha(inner) if inner.exists() else None,log_sha256=sha(log),correctness=dict(exact_contract_pass=passed),
    diagnostic_only=True,performance_admitted=False))
raise SystemExit(0 if passed else 2)
