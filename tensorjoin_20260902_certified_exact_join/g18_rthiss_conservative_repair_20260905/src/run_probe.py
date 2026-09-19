"""Validate the same-launch predicate A/B and exact-rational prefix witnesses."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from bounds import F,g,B
H=Path(__file__).resolve().parents[1];P=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--record-id',required=True);a.add_argument('--iterations',type=int,default=2);args=a.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
 freeze=json.loads((H/'artifacts/frozen_execution.json').read_text())
 for path,checksum in freeze.items():assert sha(P/path)==checksum,path
 out=H/'raw'/args.record_id;out.mkdir(exist_ok=False);result=H/'results'/f'{args.record_id}.json';assert not result.exists()
 cmd=['timeout','-k','5s','180s',str(H/'artifacts/predicate_probe'),str(H/'data'),'8192',str(args.iterations),str(out/'result.bin')]
 with (out/'program.log').open('x') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert run.returncode==0
 dtype=np.dtype([('reference','<f8'),('terminal','<f8'),('prefix','<f4'),('dims','<u4'),('stage','<u4'),('agree','<u4')]);assert dtype.itemsize==32
 r=np.fromfile(out/'result.bin',dtype=dtype);cpu=np.fromfile(H/'data/reference.f64',dtype='<f8');assert len(r)==8192
 assert np.array_equal(r['reference'].view('<u8'),cpu.view('<u8')),'GPU/CPU strict FP64 disagreement'
 assert r['agree'].all() and (r['stage']<=3).all() and ((r['dims']<=512)&(r['dims']>0)).all()
 terminal=r['stage']>=2;assert np.array_equal(r['terminal'][terminal],cpu[terminal])
 q=np.fromfile(H/'data/q.f32',dtype='<f4').reshape(8192,512);b=np.fromfile(H/'data/b.f32',dtype='<f4').reshape(8192,512)
 indices=sorted(set([0,1,2,3]+list(range(0,8192,16))))
 for i in indices:
  k=int(r['dims'][i]);S=sum((F(float(a))-F(float(b_)))**2 for a,b_ in zip(q[i,:k],b[i,:k]));s=F(float(r['prefix'][i]));assert (1-g)*S-B<=s<=(1+g)*S+B,('bound violation',i)
 record=dict(record_id=args.record_id,command=cmd,returncode=run.returncode,iterations=args.iterations,pairs=8192,
  stage_counts=np.bincount(r['stage'],minlength=4).tolist(),strict_fp64_bitwise_matches=8192,exact_rational_prefix_checks=len(indices),
  first_four_stages=r['stage'][:4].tolist(),full_result_sha256=sha(out/'result.bin'),binary_sha256=sha(H/'artifacts/predicate_probe'),
  correctness=dict(exact_contract_pass=True,meaning='8192-pair predicate A/B; not a complete RT operator or performance result'),public_seconds=None)
 result.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
