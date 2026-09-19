"""Two predeclared reversed-order complete-cost blocks; no replacements."""
import fcntl
import math
import statistics
import subprocess
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3

PY='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
assert json.loads((H/'results/admission_r3.json').read_text())['passed']
audit=json.loads((H/'results/code_audit.json').read_text());assert audit['passed']
orders=[['rt','tc','fp32'],['fp32','tc','rt']]
out=H/'results/screen.json';assert not out.exists()
files=[Path(__file__),H/'results/admission_r3.json',H/'results/code_audit.json',H/'artifacts/frozen_execution.json',
       H/'artifacts/frozen_r1.json',H/'artifacts/frozen_r2.json',H/'artifacts/frozen_r3.json',H/'ADDENDUM_OWNED_CUBLAS_R3.md',H/'PROTOCOL.md',H/'ADDENDUM_PINNED_LIFETIME_R1.md',H/'ADDENDUM_TORCH_SHUTDOWN_R2.md']
write(H/'artifacts/frozen_screen.json',dict(orders=orders,source_hashes={str(p.relative_to(P)):sha(p) for p in files},
    libraries=audit['library_selected']['containers'],observations=5,warmups=2,
    primary='Within each block: min(median(RT), median(FP32))/median(TC); both blocks >=1.10',
    confidence_interval=None,scope='Cheap screen only; two independent blocks'))
records=[]
with open('/tmp/tensorjoin_gpu3_campaign.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for block,order in enumerate(orders):
        for position,method in enumerate(order):
            check_frozen();check_r1();check_r2();check_r3()
            f=json.loads((H/'artifacts/frozen_screen.json').read_text())
            for path,h in f['source_hashes'].items():assert sha(P/path)==h,path
            for path,h in f['libraries'].items():assert sha(path)==h,path
            rid=f'screen_b{block}_p{position}_{method}_a0';label='g19_'+rid
            expected=H/'results'/f'{rid}.json'
            cmd=[PY,str(P/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(expected.relative_to(P)),
                '--physical-gpu','3','--',PY,str(H/'src/supervise_r3.py'),'--method',method,'--kind','screen','--record-id',rid]
            print('SCREEN_START '+json.dumps(cmd),flush=True);run=subprocess.run(cmd)
            guardpath=P/'results'/f'g5_guard_{label}.json';g=json.loads(guardpath.read_text()) if guardpath.exists() else {}
            passed=run.returncode==0 and g.get('admitted',False) and not g['foreign_rows'] and not g['postflight_compute_rows']
            row=dict(block=block,position=position,method=method,record_id=rid,command=cmd,
                     returncode=run.returncode,guard_path=str(guardpath.relative_to(P)),passed=bool(passed))
            inner=H/'results'/f'inner_{rid}.json'
            if inner.exists():
                r=json.loads(inner.read_text());observations=[v for v in r['rows'] if not v.get('warmup',True)]
                row['observations']=[dict(seconds=v['public_seconds'],output=v['audit'],work=v['work']) for v in observations]
                row['inner_sha256']=sha(inner)
                if passed:
                    required={entry[k]['sha256'] for entry in audit['owned_actual_compiled_objects']
                              if entry['method']==method for k in ['cubin','ptx']}
                    actual=set(r['cache'].values())
                    assert required.issubset(actual), ('Runtime-audited code missing in timing cache',method,required-actual)
                    row['runtime_audited_cache_hashes']=sorted(required)
                    row['runtime_audited_code_reproduced']=True
                    assert len(observations)==5 and all(v['audit']['exact'] for v in observations)
                    times=[v['public_seconds'] for v in observations]
                    row['median']=statistics.median(times)
                    row['p10']=float(np.quantile(times,.1));row['p90']=float(np.quantile(times,.9))
            records.append(row);write(H/'results'/f'screen_b{block}_p{position}.json',row)
            print('SCREEN_END '+json.dumps({k:v for k,v in row.items() if k not in ['observations','command']}),flush=True)
            if not passed:break
        if not records[-1]['passed']:break
complete=len(records)==6 and all(r['passed'] for r in records)
pairs=[]
if complete:
    for block in range(2):
        med={r['method']:r['median'] for r in records if r['block']==block}
        pairs.append(dict(block=block,medians=med,rt_over_tc=med['rt']/med['tc'],fp32_over_tc=med['fp32']/med['tc'],
                          strongest_control_over_tc=min(med['rt'],med['fp32'])/med['tc']))
ratios=[p['strongest_control_over_tc'] for p in pairs]
summary={}
if complete:
    marg={m:statistics.median(v['seconds'] for r in records if r['method']==m for v in r['observations']) for m in ['rt','tc','fp32']}
    summary=dict(process_block_wins=sum(r>1 for r in ratios),arithmetic_ratio_mean=statistics.mean(ratios),
        geometric_process_paired_ratio=math.exp(statistics.mean(map(math.log,ratios))),
        marginal_method_medians=marg,marginal_strongest_control_ratio=min(marg['rt'],marg['fp32'])/marg['tc'],
        confidence_interval=None,reason='Only two independent blocks; no formal CI/tail/sustained claim')
write(out,dict(complete=complete,records=records,pairs=pairs,descriptive_statistics=summary,
    narrow_screen_pass=complete and min(ratios)>=1.10,formal_or_sustained_promotion=False,novelty_pass=False))
print('SCREEN_COMPLETE',complete,'NARROW_PASS',complete and min(ratios)>=1.10,flush=True)
raise SystemExit(0 if complete else 2)
