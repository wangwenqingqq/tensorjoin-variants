"""One append-only guarded RT-HiSS export observation; never a timing claim."""
import argparse,datetime,hashlib,json,os,platform,re,subprocess,time
from pathlib import Path
from validate_export import validate
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--record-id',required=True)
    parser.add_argument('--sanitizer',choices=('none','memcheck','synccheck'),default='none');args=parser.parse_args()
    assert re.fullmatch('[a-zA-Z0-9_]+',args.record_id)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and platform.node()=='gpu-host-8'
    case=next(x for x in json.loads((HERE/'data/manifest.json').read_text())['inputs'] if x['name']==args.case)
    frozen=json.loads((HERE/'artifacts/frozen_execution.json').read_text())
    for relative,digest in frozen.items():assert sha(PROJECT/relative)==digest,relative
    target=HERE/'raw'/args.record_id;target.mkdir(exist_ok=False)
    result=HERE/f'results/case_{args.record_id}.json';assert not result.exists()
    binary=HERE/'build_a0/RT-HiSS';command=['timeout','-k','5s','180s']
    if args.sanitizer!='none':
        command+=['compute-sanitizer','--tool',args.sanitizer,'--error-exitcode','99']
        if args.sanitizer=='memcheck':command+=['--leak-check','no']
    command += [str(binary),str(PROJECT/case['path']),format(case['epsilon'],'.17g'),'highest']
    env=dict(os.environ,G17_N=str(case['n']),G17_OUTPUT_DIR=str(target),OMP_NUM_THREADS='8')
    start=time.monotonic()
    with (target/'program.log').open('x') as f:run=subprocess.run(command,cwd=target,env=env,stdout=f,stderr=subprocess.STDOUT)
    record=dict(experiment_id='tensorjoin_20260905_g17_rthiss_pair_contract',case=case,record_id=args.record_id,command=command,
        returncode=run.returncode,operational_seconds_not_performance=time.monotonic()-start,public_seconds=None,
        binary_sha256=sha(binary),libowl_sha256=sha(HERE/'build_a0/OWL/owl/libowl.so'),frozen_execution=frozen,
        runner_sha256=sha(Path(__file__)),sanitizer=args.sanitizer,started_scope='bounded full-output compatibility only')
    log=(target/'program.log').read_text(errors='replace')
    record['error_markers']=re.findall(r'[^\n]*(?:OUT OF BOUNDS|CUDA error|terminate called|ERROR SUMMARY: [1-9])[^\n]*',log)
    structural=False
    try:
        assert run.returncode==0 and not record['error_markers'],'Program failure'
        if args.sanitizer!='none':assert 'ERROR SUMMARY: 0 errors' in log,'Missing clean sanitizer summary'
        record['audit']=validate(target,case,PROJECT);structural=record['audit']['adapter_structure_pass']
    except Exception as exc:record['error']=repr(exc)
    record['correctness']=dict(exact_contract_pass=bool(structural and record['audit']['frozen_reference_pass']),adapter_structure_pass=structural,
        meaning='Frozen FP64 output equality; structural-only success never passes this exact gate')
    record['raw_files']={str(p.relative_to(PROJECT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.iterdir()) if p.is_file()}
    result.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['frozen_execution','raw_files']}),flush=True)
    return 0 if structural else 2
if __name__=='__main__':raise SystemExit(main())
