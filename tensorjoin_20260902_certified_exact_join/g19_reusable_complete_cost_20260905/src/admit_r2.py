"""Complete only failed/pending admission slots with explicit cache shutdown."""
import fcntl
import subprocess
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
out=H/'results/admission_r2.json';assert not out.exists()
old=json.loads((H/'results/admission_r1.json').read_text());assert not old['passed']
assert old['records'][-1]['record_id']=='tc_safety_memcheck_r1' and not old['records'][-1]['passed']
records=[r for r in old['records'] if r['passed']];assert len(records)==6
slots=[('tc','safety','memcheck'),('fp32','safety','memcheck')]
slots += [(m,'stress','none') for m in ['rt','tc','fp32']]
slots += [(m,'profile','nsys') for m in ['rt','tc','fp32']]
write(H/'artifacts/admission_r2_slots.json',dict(slots=slots,retained=[r['record_id'] for r in records],source_sha256=sha(__file__)))
with open('/tmp/tensorjoin_gpu3_campaign.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for method,kind,tool in slots:
        check_frozen();check_r1();check_r2();rid=f'{method}_{kind}_{tool}_r2';label='g19_'+rid
        expected=H/'results'/f'{rid}.json'
        cmd=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(expected.relative_to(P)),
             '--physical-gpu','3','--',PY,str(H/'src/supervise_r2.py'),'--method',method,'--kind',kind,'--tool',tool,'--record-id',rid]
        print('SLOT_START '+json.dumps(cmd),flush=True);r=subprocess.run(cmd)
        g=P/'results'/f'g5_guard_{label}.json';guard=json.loads(g.read_text()) if g.exists() else {}
        passed=r.returncode==0 and guard.get('admitted',False) and not guard['foreign_rows'] and not guard['postflight_compute_rows']
        row=dict(method=method,kind=kind,tool=tool,record_id=rid,command=cmd,returncode=r.returncode,
                 guard_path=str(g.relative_to(P)),result_path=str(expected.relative_to(P)),passed=bool(passed))
        records.append(row)
        out.write_text(json.dumps(dict(records=records,complete=len(records)==14,passed=len(records)==14 and all(x['passed'] for x in records),
            retained_original_campaign_pass=False,performance_admitted=False,novelty_pass=False),indent=2)+'\n')
        print('SLOT_END '+json.dumps(row),flush=True)
        if not passed:break
print('ADMISSION_R2_END',len(records),14,flush=True)
raise SystemExit(0 if len(records)==14 and all(r['passed'] for r in records) else 2)
