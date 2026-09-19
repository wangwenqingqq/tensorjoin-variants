"""Predeclared Williams-order complete 2x2 timing; no variant tuning."""
import argparse,gc,json,os,platform,time,traceback
import numpy as np
import torch
from g2_operator_r1 import Matrix,HERE,ROOT,full_source,output_check,sha,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',required=True,type=int);a=p.parse_args()
 spec=json.loads((HERE/'schedule_g2.json').read_text())['processes'][a.index]
 dest=HERE/'results'/f"{spec['label']}.json";assert not dest.exists()
 r=dict(spec,started=time.time(),pid=os.getpid(),host=platform.node(),torch=torch.__version__,torch_cuda=torch.version.cuda,numpy=np.__version__,samples=[],novelty_promotion=False);o=None
 try:
  assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='2' and os.environ['NVIDIA_TF32_OVERRIDE']=='0'
  for name,h in json.loads((HERE/'artifacts/g2_timing_freeze_a0.json').read_text()).items():assert sha(HERE/name)==h,name
  prereq=json.loads((HERE/'artifacts/g2_admission_a0.json').read_text());assert prereq['pass']
  for name,h in prereq['evidence_sha256'].items():assert sha(HERE/name)==h,name
  o=Matrix();r.update(identities=o.identities,runtime_abi=o.abi)
  x=full_source()
  def one(m,phase,i):
   out,rec=o.run(m,x);h=output_check(out)
   rec.update(method=m,phase=phase,iteration=i,hash=h,count=len(out));r['samples'].append(rec)
  r['compiled_after_warmups']={}
  for m in spec['order']:
   for i in range(2):one(m,'warmup',i)
   r['compiled_after_warmups'][m]=o.capture()
   for i in range(9):one(m,'retained',i)
  for m in reversed(spec['order']):
   for i in range(16):one(m,'repeat16',i)
  r['compiled_after']=o.capture();assert r['compiled_after']==r['compiled_after_warmups'][spec['order'][-1]]
  assert verify()==o.identities;r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['pass']=True
 except Exception:r['pass']=False;r['exception']=traceback.format_exc();print(r['exception'],flush=True)
 finally:
  if o is not None:o.close()
  o=None;gc.collect();torch.cuda.empty_cache();r['ended']=time.time();dest.write_text(json.dumps(r,indent=2))
 print(json.dumps(dict(label=spec['label'],pass_=r['pass'],calls=len(r['samples']))),flush=True)
 return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
