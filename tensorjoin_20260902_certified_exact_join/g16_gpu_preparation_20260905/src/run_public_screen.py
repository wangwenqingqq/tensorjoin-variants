"""Two predeclared three-arm G16 processes blocks, no replacements."""
import fcntl,json,os,subprocess,math,statistics
from pathlib import Path
from g2b_public_common import atomic_json,sha256_file
HERE=Path(__file__).resolve().parents[1]; PROJECT=HERE.parent
OLD=PROJECT/'g15_strong_control_20260905'
PYTHON='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
ORDERS=(('gpu_g5','fp32_cpu','fp32_gpu'),('fp32_gpu','fp32_cpu','gpu_g5'))

def main():
    if (HERE/'results/public_start.json').exists(): raise FileExistsError('Screen already started')
    assert os.environ.get('NVIDIA_TF32_OVERRIDE')=='0'
    assert json.loads((HERE/'results/safety_gates_r1.json').read_text())['complete']
    assert json.loads((HERE/'results/precision_audit.json').read_text())['precision_gate_pass']
    frozen=json.loads((HERE/'artifacts/frozen_screen_sources.json').read_text())
    records=[]
    with open('/tmp/tensorjoin_gpu2_campaign.lock','a+') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      atomic_json(HERE/'results/public_start.json',dict(orders=ORDERS,pid=os.getpid(),frozen=frozen,orchestrator_sha256=sha256_file(Path(__file__))))
      for block,order in enumerate(ORDERS):
        for position,method in enumerate(order):
          for rel,h in frozen.items(): assert sha256_file(PROJECT/rel)==h,rel
          rid=f'g16_b{block}_p{position}_{method}_a0'
          if method=='gpu_g5': runner=HERE/'src/run_gpu_prepared_g5.py'; path=HERE/f'results/gpu_prepare_screen_{rid}.json'
          elif method=='fp32_gpu': runner=HERE/'src/run_fp32_gpu_norms.py'; path=HERE/f'results/fp32_gpu_screen_{rid}.json'
          else: runner=OLD/'src/run_fp32_first.py'; path=OLD/f'results/fp32_screen_{rid}.json'
          command=[PYTHON,str(PROJECT/'src/run_g5_guarded_process.py'),'--label',rid,'--expected-result',str(path.relative_to(PROJECT)),'--physical-gpu','2','--',PYTHON,str(runner),'--phase','screen','--record-id',rid]
          print('PUBLIC_START '+json.dumps(command),flush=True)
          done=subprocess.run(command,check=False); guard_path=PROJECT/f'results/g5_guard_{rid}.json'
          guard=json.loads(guard_path.read_text()) if guard_path.exists() else {}
          row=dict(block=block,position=position,method=method,command=command,returncode=done.returncode,
                   admitted=bool(guard.get('admitted')),guard_path=str(guard_path.relative_to(PROJECT)),result_path=str(path.relative_to(PROJECT)))
          if path.exists():
            r=json.loads(path.read_text());row.update(public_seconds=r['public_seconds'],correctness=r['correctness'],result_sha256=sha256_file(path))
          records.append(row);atomic_json(HERE/f'results/public_b{block}_p{position}.json',row)
          print('PUBLIC_COMPLETE '+json.dumps(row),flush=True)
          if done.returncode or not row['admitted']: break
        if done.returncode or not records[-1]['admitted']: break
    complete=len(records)==6 and all(r['admitted'] for r in records)
    pairs=[]
    if complete:
      for block in range(2):
        t={r['method']:r['public_seconds'] for r in records if r['block']==block}
        pairs.append(dict(block=block,times=t,fp32_cpu_over_tc=t['fp32_cpu']/t['gpu_g5'],
                     fp32_gpu_over_tc=t['fp32_gpu']/t['gpu_g5'],fastest_control_over_tc=min(t['fp32_cpu'],t['fp32_gpu'])/t['gpu_g5']))
    ratios=[p['fastest_control_over_tc'] for p in pairs]
    # Two-block descriptive statistics only. No confidence inference from n=2.
    stats={} if not complete else dict(process_wins=sum(r>1 for r in ratios),arithmetic_ratio_mean=statistics.mean(ratios),
          geometric_paired_ratio=math.exp(statistics.mean(map(math.log,ratios))),
          marginal_ratio=statistics.median(min(p['times']['fp32_cpu'],p['times']['fp32_gpu']) for p in pairs)/statistics.median(p['times']['gpu_g5'] for p in pairs),
          confidence_interval=None,reason='Only two independent blocks; no formal CI or tail admission')
    result=dict(complete=complete,records=records,pairs=pairs,descriptive_statistics=stats,
      narrow_engineering_screen_pass=complete and min(ratios)>=1.25,formal_or_sustained_promotion=False,novelty_pass=False)
    atomic_json(HERE/'results/public_screen.json',result);print('PUBLIC_SUMMARY '+json.dumps(result),flush=True)
    return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
