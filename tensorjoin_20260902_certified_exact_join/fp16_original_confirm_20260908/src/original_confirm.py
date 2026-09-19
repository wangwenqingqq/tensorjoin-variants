"""One isolated original-GPU2 qualification suite; no performance comparison."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
    gate=['gpu0_fixture_a2','gpu0_public_a2','gpu0_full_a2','gpu0_stress_a0','gpu0_memcheck_a0','gpu0_synccheck_a0','gpu0_initcheck_a0','gpu0_racecheck_a0']
    for label in gate:
        assert json.loads((HERE/'results'/f'{label}.json').read_text())['pass']
        assert json.loads((HERE/'results'/f'{label}_guard.json').read_text())['pass']
    freeze=json.loads((HERE/'artifacts/original_confirmation_freeze.json').read_text())
    for path,h in freeze['files'].items():assert hashlib.sha256((HERE/path).read_bytes()).hexdigest()==h,path
    runs=[('gpu2_abi_a0',['src/audit_runtime_abi.py']),
          ('gpu2_fixture_a0',['src/validate_control_r2.py','--mode','fixture']),
          ('gpu2_public_a0',['src/validate_control_r2.py','--mode','public']),
          ('gpu2_full_a0',['src/validate_control_r2.py','--mode','full']),
          ('gpu2_stress_a0',['src/safety_probe.py','--mode','stress'])]
    for kind in ['memcheck','synccheck','initcheck','racecheck']:
        runs.append((f'gpu2_{kind}_a0',['SANITIZER',kind,'src/safety_probe.py','--mode','probe']))
    result={'pass':False,'started':time.time(),'runs':[],'scope':'same original GPU2, same frozen artifacts, functional/safety only','speed_claim':False}
    try:
        for label,args in runs:
            assert not (HERE/'results'/f'{label}.json').exists()
            if args[0]=='SANITIZER':command=['compute-sanitizer','--tool',args[1],'--error-exitcode','86',sys.executable]+args[2:]
            else:command=[sys.executable]+args
            command+=['--label',label];log=HERE/'raw'/f'{label}.log'
            with log.open('x') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=False)
            rec={'label':label,'exit_code':r.returncode,'log':str(log.relative_to(HERE))};result['runs'].append(rec);print(json.dumps(rec),flush=True)
            assert r.returncode==0 and json.loads((HERE/'results'/f'{label}.json').read_text())['pass'],rec
        result['pass']=True
    finally:
        result['ended']=time.time()
        with (HERE/'results/original_confirmation_a0.json').open('x') as f:json.dump(result,f,indent=2)
    return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
