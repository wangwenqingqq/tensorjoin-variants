import ctypes,fcntl,hashlib,json,os,platform,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
frozen=json.loads((HERE/'artifacts/timing_freeze.json').read_text())
for name,h in frozen.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
def query(args):return subprocess.check_output(['nvidia-smi']+args,text=True).strip()
state=query(['-i','2','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'])
processes=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']).splitlines()
target=[x for x in processes if x.startswith('GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245')]
assert not target,target
locks=[]
for p in ['/tmp/tensorjoin_gpu2_campaign.lock','/tmp/tensorjoin_g5_gpu2.lock']:
    with open(p,'a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(p)
        fcntl.flock(f,fcntl.LOCK_UN)
import numpy as np
pools=[]
for p in (Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'):
    lib=ctypes.CDLL(str(p))
    for name in ['openblas_get_num_threads64_','openblas_get_num_threads','scipy_openblas_get_num_threads64_']:
        try:
            n=getattr(lib,name)();assert n==4
            pools.append(dict(library=p.name,symbol=name,threads=n))
        except AttributeError:pass
assert pools
r=dict(pass_=True,checked_frozen_files=len(frozen),gpu=state,target_processes=target,
       locks_available=locks,openblas_pools=pools,environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
       host=platform.node(),time=time.time())
with (HERE/'results/final_resource_check.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
