"""Six frozen-order complete-operator processes, not a new planner."""
import argparse,gc,json,os,time,traceback
import numpy as np
import torch
from common import *
ORDERS=[['original','raw_tree','hadamard_tree'],['raw_tree','hadamard_tree','original'],['hadamard_tree','original','raw_tree'],['hadamard_tree','raw_tree','original'],['original','hadamard_tree','raw_tree'],['raw_tree','original','hadamard_tree']]

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();assert 0<=a.index<6
 result={'index':a.index,'layout_order':ORDERS[a.index],'method_order':PLANS if a.index%2==0 else PLANS[::-1],'started':time.time(),'pid':os.getpid(),'pass':False,'samples':[]}
 op=None
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='2' and os.environ['NVIDIA_TF32_OVERRIDE']=='0'
  freeze=json.loads((HERE/'artifacts/timing_freeze_a0.json').read_text())
  for rel,h in freeze.items():assert sha(HERE/rel)==h,rel
  admission=json.loads((HERE/'artifacts/admission_a0.json').read_text());assert admission['pass']
  for rel,h in admission['evidence'].items():assert sha(HERE/rel)==h,rel
  result['identities']=check_old();xs,orders=load();op=Matrix()
  def call(layout,method,phase,i):
   out,record=one(op,method,xs[layout],orders[layout]);record.update(layout=layout,method=method,phase=phase,iteration=i)
   result['samples'].append(record)
   print(json.dumps({'layout':layout,'method':method,'phase':phase,'i':i,'seconds':record['seconds'],'inner_seconds':record['inner_seconds']}),flush=True)
  for layout in result['layout_order']:
   for method in result['method_order']:
    for i in range(2):call(layout,method,'warmup',i)
    op.capture()
    for i in range(7):call(layout,method,'retained',i)
  for layout in result['layout_order'][::-1]:
   for method in result['method_order'][::-1]:
    for i in range(8):call(layout,method,'repeat8',i)
  result['compiled']=op.capture();assert verify()==op.identities
  result['peak_allocated_bytes']=torch.cuda.max_memory_allocated();result['pass']=True
 except Exception:result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 finally:
  if op is not None:op.close()
  op=None;gc.collect();torch.cuda.empty_cache();result['ended']=time.time()
  (HERE/'results'/f'p{a.index:02d}_a0.json').open('x').write(json.dumps(result,indent=2))
 print(json.dumps({'index':a.index,'pass':result['pass'],'calls':len(result['samples'])}),flush=True)
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
