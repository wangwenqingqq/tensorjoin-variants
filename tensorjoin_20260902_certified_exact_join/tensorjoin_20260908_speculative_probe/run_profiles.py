"""Separate traced diagnostics; never mix profiler times into confirmation."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
for i in range(3):assert json.loads((HERE/'results'/f'confirm_{i}.json').read_text())['pass_']
env=os.environ.copy()
env.update(NVIDIA_TF32_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
for method in ['pipeline','draft_serial','speculative']:
    label='profile_'+method
    prefix=HERE/'raw'/label
    cmd=[sys.executable,str(HERE/'src/guard_r2.py'),'--label',label,'--gpu','2',
         '--target','8p_gpu2','--','nsys','profile','--trace=cuda,nvtx',
         '--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
         '--capture-range-end=stop','--force-overwrite=false','--output='+str(prefix),
         sys.executable,str(HERE/'src/run.py'),'--phase','profile','--label',label,
         '--profile-method',method]
    print(json.dumps(dict(starting=label)),flush=True)
    rc=subprocess.run(cmd,env=env).returncode
    if rc:raise SystemExit(rc)
    subprocess.run(['nsys','export','--type=sqlite','--output='+str(prefix)+'.sqlite',
        str(prefix)+'.nsys-rep'],check=True,env=env)
