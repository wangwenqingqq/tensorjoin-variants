"""Read final GPU state, release transient audit locks, and seal the archive."""
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
assert platform.node()=='gpu-host-8'
def query(args):return subprocess.check_output(['nvidia-smi']+args,text=True).strip()
uuid='GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
actual=query(['-i','2','--query-gpu=uuid','--format=csv,noheader']);assert actual==uuid
state=query(['-i','2','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'])
apps=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
active=[line for line in apps.splitlines() if line.startswith(uuid)]
locks={}
for name in ['/tmp/tensorjoin_gpu2_campaign.lock','/tmp/tensorjoin_g5_gpu2.lock']:
    with open(name,'a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks[name]=True;fcntl.flock(f,fcntl.LOCK_UN)
        except BlockingIOError:locks[name]=False
guards=[json.loads(p.read_text()) for p in (HERE/'results').glob('*_guard.json')]
assert all(g['pass'] for g in guards)
alive=[]
for g in guards:
    for pid,start in g.get('verified_descendants',{}).items():
        p=Path('/proc')/str(pid)/'stat'
        if p.exists():
            fields=p.read_text().rsplit(')',1)[1].split()
            if int(fields[19])==start:alive.append(int(pid))
record=dict(time=time.time(),host=platform.node(),gpu=state,gpu2_processes=active,
    audit_locks_available=locks,own_live_descendants=alive,
    all_guards_pass=True,guard_count=len(guards),own_resources_released=not alive and all(locks.values()))
(HERE/'results/final_resource_check.json').write_text(json.dumps(record,indent=2))
assert not alive
manifest={}
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='archive_manifest.json':continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    manifest[str(p.relative_to(HERE))]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
(HERE/'artifacts/archive_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(resources=record,archived_files=len(manifest))))
