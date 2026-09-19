"""Predeclared G19 admission slots, serialized and fail-closed."""
import fcntl
import os
import subprocess
from common import *
PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
assert os.environ['NVIDIA_TF32_OVERRIDE']=='0' and os.environ['OMP_NUM_THREADS']=='8'
out=H/'results/admission.json';assert not out.exists()
slots=[(m,'matrix','none') for m in ['rt_on','rt','tc','fp32']]
slots += [('rt','safety','memcheck'),('rt','safety','synccheck'),('tc','safety','memcheck'),('fp32','safety','memcheck')]
slots += [(m,'stress','none') for m in ['rt','tc','fp32']]
slots += [(m,'profile','nsys') for m in ['rt','tc','fp32']]
write(H/'artifacts/admission_slots.json',dict(slots=slots,source_sha256=sha(__file__)))
records=[]
with open('/tmp/tensorjoin_gpu3_campaign.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for method,kind,tool in slots:
        check_frozen();rid=f'{method}_{kind}_{tool}_a0';label='g19_'+rid
        expected=H/'results'/f'{rid}.json'
        cmd=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(expected.relative_to(P)),
             '--physical-gpu','3','--',PY,str(H/'src/supervise.py'),'--method',method,'--kind',kind,'--tool',tool,'--record-id',rid]
        print('SLOT_START '+json.dumps(cmd),flush=True);r=subprocess.run(cmd)
        g=P/'results'/f'g5_guard_{label}.json';guard=json.loads(g.read_text()) if g.exists() else {}
        passed=r.returncode==0 and guard.get('admitted',False) and not guard['foreign_rows'] and not guard['postflight_compute_rows']
        row=dict(method=method,kind=kind,tool=tool,record_id=rid,command=cmd,returncode=r.returncode,
                 guard_path=str(g.relative_to(P)),result_path=str(expected.relative_to(P)),passed=bool(passed))
        records.append(row)
        # This live campaign receipt is finalized only when all slots pass or a
        # slot stops the campaign. Per-slot logs/results are never overwritten.
        out.write_text(json.dumps(dict(records=records,complete=len(records)==len(slots),
                                      passed=len(records)==len(slots) and all(x['passed'] for x in records),
                                      performance_admitted=False,novelty_pass=False),indent=2)+'\n')
        print('SLOT_END '+json.dumps(row),flush=True)
        if not passed:break
print('ADMISSION_END',len(records),len(slots),flush=True)
