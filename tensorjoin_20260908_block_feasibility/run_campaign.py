import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert json.loads((HERE/'artifacts/admission.json').read_text())['pass']
results = []
for index in range(3):
    label = f'confirm_{index:02d}_a0'
    cmd = [sys.executable,str(HERE/'src/guard_r2.py'),'--label',label,'--gpu','0',
           '--target','8p_gpu0','--',sys.executable,str(HERE/'src/runner.py'),'--index',str(index)]
    print(json.dumps(dict(starting=label,time=time.time())),flush=True)
    env = os.environ.copy()
    env.update(NVIDIA_TF32_OVERRIDE='0',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
    proc = subprocess.run(cmd,env=env,check=False)
    results.append(dict(label=label,exit_code=proc.returncode,ended=time.time()))
    (HERE/'results/campaign_progress.json').write_text(json.dumps(results,indent=2))
    if proc.returncode:
        raise SystemExit(proc.returncode)
print(json.dumps(dict(campaign_complete=True,processes=len(results))),flush=True)
