import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
admission=json.loads((HERE/'results/admission_a0.json').read_text())
assert admission['pass_']
progress=[]
env=os.environ.copy()
env.update(NVIDIA_TF32_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
for index in range(3):
    label=f'confirm_{index}'
    cmd=[sys.executable,str(HERE/'src/guard_r2.py'),'--label',label,'--gpu','2',
         '--target','8p_gpu2','--',sys.executable,str(HERE/'src/run.py'),
         '--phase','confirm','--label',label,'--index',str(index)]
    print(json.dumps(dict(starting=label,time=time.time())),flush=True)
    proc=subprocess.run(cmd,env=env,check=False)
    progress.append(dict(label=label,exit_code=proc.returncode,ended=time.time()))
    (HERE/'results/campaign_progress.json').write_text(json.dumps(progress,indent=2))
    if proc.returncode:raise SystemExit(proc.returncode)
print(json.dumps(dict(campaign_complete=True)),flush=True)
