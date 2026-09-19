import json,os,resource,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
env=os.environ.copy()
env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
           CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
start=time.time()
with (here/'raw/probe.log').open('x') as f:
    try:
        p=subprocess.run([sys.executable,str(here/'src/probe.py')],cwd=here,env=env,
                         stdout=f,stderr=subprocess.STDOUT,timeout=1800)
        record=dict(pass_=p.returncode==0,exit_code=p.returncode)
    except subprocess.TimeoutExpired:
        record=dict(pass_=False,timeout=True)
record.update(started=start,ended=time.time(),cpu_only=True,
              child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
with (here/'results/process.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
raise SystemExit(0 if record['pass_'] else 1)
