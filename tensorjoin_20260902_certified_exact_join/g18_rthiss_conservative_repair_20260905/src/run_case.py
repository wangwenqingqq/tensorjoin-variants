"""One complete repaired RT-HiSS correctness-only operator observation."""
import argparse,hashlib,json,os,platform,re,subprocess,sys,time
from pathlib import Path
import numpy as np
from bounds import cuts
H=Path(__file__).resolve().parents[1];P=H.parent;G17=P/'g17_rthiss_pair_contract_20260905'
sys.path.insert(0,str(G17/'src'));from validate_export import validate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--record-id',required=True);parser.add_argument('--sanitizer',choices=['none','memcheck','synccheck'],default='none');a=parser.parse_args()
 assert re.fullmatch('[A-Za-z0-9_]+',a.record_id) and os.environ['CUDA_VISIBLE_DEVICES']=='2' and platform.node()=='gpu-host-8'
 frozen=json.loads((H/'artifacts/frozen_execution.json').read_text())
 for path,digest in frozen.items():assert sha(P/path)==digest,path
 case=next(v for v in json.loads((G17/'data/manifest_r1.json').read_text())['inputs'] if v['name']==a.case)
 lo,hi=cuts(case['reference_threshold']);target=H/'raw'/a.record_id;target.mkdir(exist_ok=False);result=H/'results'/f'case_{a.record_id}.json';assert not result.exists()
 binary=H/'build_r1/RT-HiSS';cmd=['timeout','-k','5s','180s']
 if a.sanitizer!='none':
  cmd+=['compute-sanitizer','--tool',a.sanitizer,'--error-exitcode','99']
  if a.sanitizer=='memcheck':cmd+=['--leak-check','no']
 cmd += [str(binary),str(P/case['path']),format(case['epsilon'],'.17g'),'highest']
 env=dict(os.environ,G17_N=str(case['n']),G17_OUTPUT_DIR=str(target),G18_T=repr(case['reference_threshold']),G18_LOW=repr(float(lo)),G18_HIGH=repr(float(hi)),OMP_NUM_THREADS='8')
 begin=time.monotonic()
 with (target/'program.log').open('x') as f:run=subprocess.run(cmd,cwd=target,env=env,stdout=f,stderr=subprocess.STDOUT)
 record=dict(record_id=a.record_id,case=case,command=cmd,returncode=run.returncode,sanitizer=a.sanitizer,public_seconds=None,operational_seconds_not_performance=time.monotonic()-begin,
  binary_sha256=sha(binary),libowl_sha256=sha(H/'build_r1/OWL/owl/libowl.so'),low=float(lo),high=float(hi),threshold=case['reference_threshold'])
 passed=False
 try:
  log=(target/'program.log').read_text();assert run.returncode==0
  assert not re.search(r'OUT OF BOUNDS|CUDA error|terminate called|ERROR SUMMARY: [1-9]',log)
  if a.sanitizer!='none':assert 'ERROR SUMMARY: 0 errors' in log
  receipt=json.loads((target/'export.json').read_text());assert receipt['g18_repaired_refinement'] and not receipt['source_predicate_unchanged']
  audit=validate(target,case,P);record['audit']=audit
  assert audit['adapter_structure_pass'] and audit['frozen_reference_pass'] and audit['rt_candidate_missing_frozen_pairs']==0
  previous_id='none_boundary_zero32_a0' if a.case=='boundary_zero32' else 'native_'+a.case+'_a0'
  old=json.loads((G17/'results'/f'case_{previous_id}.json').read_text())
  assert audit['unique_rt_candidate_pairs']==old['audit']['unique_rt_candidate_pairs']==case['n']**2
  for name in ['dimension_map.u32','point_map.u32']:assert (target/name).read_bytes()==(G17/'raw'/previous_id/name).read_bytes(),name
  counts=np.fromfile(target/'g18_stats.u64',dtype='<u8');assert len(counts)==6
  assert int(counts[:4].sum())==audit['unique_rt_candidate_pairs']
  assert int(counts[1]+counts[2])==audit['output_pairs']
  assert int(counts[5])==int(counts[2]+counts[3])*512
  record['work']=dict(safe_reject=int(counts[0]),safe_accept=int(counts[1]),terminal_accept=int(counts[2]),terminal_reject=int(counts[3]),fp32_dimensions=int(counts[4]),fp64_dimensions=int(counts[5]),terminal_fraction=float((counts[2]+counts[3])/counts[:4].sum()),candidate_set_same_as_native=True)
  passed=True
 except Exception as exc:record['error']=repr(exc)
 record['correctness']=dict(exact_contract_pass=passed,meaning='complete frozen FP64-reference output plus candidate/permutation/counter gates')
 record['raw_files']={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in target.iterdir() if p.is_file()}
 result.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='raw_files'}));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
