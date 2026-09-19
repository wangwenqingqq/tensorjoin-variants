"""Fixed fail-closed G16 safety sequence; one shared GPU lock."""
import fcntl,json,os,subprocess
from pathlib import Path
from g2b_public_common import atomic_json,sha256_file
HERE=Path(__file__).resolve().parents[1]; PROJECT=HERE.parent
PYTHON='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'

def main():
    if (HERE/'results/safety_gates_r1.json').exists(): raise FileExistsError('Already run')
    assert json.loads((HERE/'results/metadata_a0.json').read_text())['correctness']['exact_contract_pass']
    assert json.loads((PROJECT/'g15_strong_control_20260905/results/public_screen.json').read_text())['complete']
    assert os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    assert json.loads((HERE/'results/norm_metadata_a0.json').read_text())['correctness']['exact_contract_pass']
    fp=[PYTHON,str(HERE/'src/run_fp32_gpu_norms.py')]
    tc=[PYTHON,str(HERE/'src/run_gpu_prepared_g5.py')]
    md=[PYTHON,str(HERE/'src/validate_metadata.py')]
    mem=['compute-sanitizer','--tool','memcheck','--leak-check','no','--error-exitcode','99']
    sync=['compute-sanitizer','--tool','synccheck','--error-exitcode','99']
    slots=[
      ('fp32_validation_a1','fp32_gpu_validation_a1.json',fp+['--phase','validation','--record-id','a1']),
      ('tc_full_a0','gpu_prepare_compatibility_full_a0.json',tc+['--phase','compatibility','--record-id','full_a0']),
      ('fp32_full_a0','fp32_gpu_compatibility_full_a0.json',fp+['--phase','compatibility','--record-id','full_a0']),
      ('metadata_stress_a0','metadata_stress_a0.json',md+['--record-id','stress_a0','--stress']),
      ('fp32_stress_a0','fp32_gpu_stress_a0.json',fp+['--phase','stress','--record-id','a0']),
      ('tc_memcheck_a0','gpu_prepare_compatibility_memcheck_a0.json',mem+tc+['--phase','compatibility','--record-id','memcheck_a0']),
      ('fp32_memcheck_a0','fp32_gpu_compatibility_memcheck_a0.json',mem+fp+['--phase','compatibility','--record-id','memcheck_a0']),
      ('metadata_synccheck_a0','metadata_synccheck_a0.json',sync+md+['--record-id','synccheck_a0']),
      ('fp32_synccheck_a0','fp32_gpu_validation_synccheck_a0.json',sync+fp+['--phase','validation','--record-id','synccheck_a0']),
    ]
    frozen=json.loads((HERE/'artifacts/frozen_sources_r1.json').read_text())
    frozen.update(json.loads((PROJECT/'g15_strong_control_20260905/artifacts/frozen_b_sources.json').read_text()))
    records=[]
    with open('/tmp/tensorjoin_gpu2_campaign.lock','a+') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      atomic_json(HERE/'results/safety_start_r1.json',dict(slots=slots,frozen=frozen,pid=os.getpid(),orchestrator_sha256=sha256_file(Path(__file__))))
      for short,name,child in slots:
        for relative,digest in frozen.items(): assert sha256_file(PROJECT/relative)==digest,relative
        label='g16_'+short; result_path=HERE/'results'/name
        command=[PYTHON,str(PROJECT/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(result_path.relative_to(PROJECT)),'--physical-gpu','2','--']+child
        print('GATE_START '+json.dumps(command),flush=True)
        done=subprocess.run(command,check=False)
        guard_path=PROJECT/f'results/g5_guard_{label}.json'
        guard=json.loads(guard_path.read_text()) if guard_path.exists() else {}
        row=dict(label=label,command=command,returncode=done.returncode,admitted=bool(guard.get('admitted')),
                 result_path=str(result_path.relative_to(PROJECT)),guard_path=str(guard_path.relative_to(PROJECT)))
        if result_path.exists():
          result=json.loads(result_path.read_text()); row.update(result_sha256=sha256_file(result_path),correctness=result['correctness'],diagnostic_seconds=result.get('public_seconds'))
        records.append(row); atomic_json(HERE/f'results/safety_r1_{len(records):02d}.json',row)
        print('GATE_COMPLETE '+json.dumps(row),flush=True)
        if done.returncode or not row['admitted']: break
    complete=len(records)==len(slots) and all(r['admitted'] for r in records)
    atomic_json(HERE/'results/safety_gates_r1.json',dict(complete=complete,records=records))
    return 0 if complete else 2
if __name__=='__main__': raise SystemExit(main())
