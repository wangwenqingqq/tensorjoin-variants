"""Additional exact-code binding within the predeclared static/runtime gate."""
import fcntl
import subprocess
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3
PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
assert json.loads((H/'results/admission_r3.json').read_text())['passed']
check_frozen();check_r1();check_r2();check_r3()
write(H/'artifacts/frozen_runtime_probe.json',{str((H/'src'/f).relative_to(P)):sha(H/'src'/f)
    for f in ['runtime_probe.py','runtime_supervise.py','collect_runtime.py']})
rows=[]
with open('/tmp/tensorjoin_gpu3_campaign.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for method,ncu in [('tc',False),('fp32',True)]:
        rid=f'{method}_runtime_bind_r3';label='g19_'+rid
        expected=H/'results'/f'{rid}.json'
        cmd=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(expected.relative_to(P)),
             '--physical-gpu','3','--',PY,str(H/'src/runtime_supervise.py'),'--method',method,'--record-id',rid]+(['--ncu'] if ncu else [])
        print('BIND_START',json.dumps(cmd),flush=True);r=subprocess.run(cmd)
        path=P/'results'/f'g5_guard_{label}.json';g=json.loads(path.read_text()) if path.exists() else {}
        passed=r.returncode==0 and g.get('admitted',False) and not g['foreign_rows'] and not g['postflight_compute_rows']
        rows.append(dict(method=method,record_id=rid,command=cmd,guard_path=str(path.relative_to(P)),passed=bool(passed)))
        if not passed:break
        if ncu:
            for page,extra in [('source',['--print-source','sass']),('raw',[])]:
                p=H/'raw'/f'{rid}_{page}.csv'
                with p.open('x') as f:subprocess.run(['ncu','--import',str(H/'artifacts'/f'{rid}.ncu-rep'),'--page',page,'--csv']+extra,stdout=f,check=True)
write(H/'results/runtime_collection.json',dict(records=rows,passed=len(rows)==2 and all(r['passed'] for r in rows),performance_admitted=False))
raise SystemExit(0 if len(rows)==2 and all(r['passed'] for r in rows) else 2)
