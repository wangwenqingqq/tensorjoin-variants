"""Frozen-code complete A/C timing. No predicate/algorithm changes."""
import argparse,gc,json,os,platform,time,traceback
import numpy as np
import torch
from control_r2 import Control,HERE,ROOT,full_source,output_check,sha,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--order',choices=['AC','CA'],required=True);a=p.parse_args()
 dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
 r={'label':a.label,'order':a.order,'pass':False,'started':time.time(),'pid':os.getpid(),'host':platform.node(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':np.__version__,'samples':[],'novelty_promotion':False,'scope':'warmed pageable FP32 host to sorted canonical uint64 host'};o=None
 try:
  assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='2' and os.environ['NVIDIA_TF32_OVERRIDE']=='0'
  for name,h in json.loads((HERE/'prerequisites.json').read_text()).items():assert sha(ROOT/name)==h,name
  for key in ['original_confirmation_a0','original_confirmation_a0_guard']:
   assert json.loads((ROOT/'fp16_original_confirm_20260908/results'/f'{key}.json').read_text())['pass']
  for name,h in json.loads((HERE/'artifacts/g1_source_freeze.json').read_text()).items():assert sha(HERE/name)==h,name
  o=Control();r.update(identities=o.identities,runtime_abi=o.abi,library=o.blas.record(),library_hashes=o.library_hashes)
  x=full_source()
  def one(m,phase,i):
   out,rec=o.run(m,x);h=output_check(out)
   seconds=rec.pop('seconds') if m=='A' else rec.pop('seconds_diagnostic_until_timing_admitted')
   rec.update(seconds=seconds,method=m,phase=phase,iteration=i,hash=h,count=len(out));r['samples'].append(rec)
  for m in a.order:
   for i in range(2):one(m,'warmup',i)
   if m=='C':r['compiled_before_retained']=o.capture()
   for i in range(9):one(m,'retained',i)
  for m in reversed(a.order):
   for i in range(16):one(m,'repeat16',i)
  r['compiled_after']=o.capture();assert r['compiled_before_retained']==r['compiled_after']
  assert verify()==o.identities;r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['pass']=True
 except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
 finally:
  if o is not None:o.close()
  o=None;gc.collect();torch.cuda.empty_cache();r['ended']=time.time()
  with dest.open('x') as f:json.dump(r,f,indent=2)
 print(json.dumps({'label':a.label,'pass':r['pass'],'calls':len(r['samples'])}),flush=True)
 return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
