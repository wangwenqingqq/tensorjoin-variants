"""Diagnose failed invariance, not a repaired admission or timing result."""
import gc,json,time,traceback
import numpy as np
import torch
from common import *

def classify_pair(a,b):
 x=np.zeros((N,D),dtype=np.float32);x[0]=a;x[64]=b
 op=Matrix();v=torch.from_numpy(x).to('cuda');prep=op.prepare(v,8)
 tr=torch.tensor([0],device='cuda',dtype=torch.int32);tc=torch.tensor([1],device='cuda',dtype=torch.int32)
 out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);c=torch.zeros(6,device='cuda',dtype=torch.int32)
 score=torch.empty(1,device='cuda',dtype=torch.int32);dump=score.view(torch.float32)
 op.stage1('F8',prep,tr,tc,out,amb,c,score,dump,dump)
 vals=c.cpu().numpy();accepted=out[:int(vals[0])].cpu().numpy();uncertain=amb[:int(vals[1])].cpu().numpy()
 label='accept' if np.any(accepted==64) else 'uncertain' if np.any(uncertain==64) else 'reject'
 q=prep[0][[0,64]].cpu().numpy();si,sj=prep[2][[0,64]].cpu().numpy();acc=int(q[0].astype(np.int64)@q[1].astype(np.int64))
 dot=np.float32(np.float32(np.float32(acc)*si)*sj)
 rec={'class':label,'scales':[float(si),float(sj)],'int_dot':acc,'scaled_dot_fp32':float(dot),'scaled_dot_bits':int(np.array(dot).view(np.uint32))}
 del v,prep,tr,tc,out,amb,c,score,dump;op.close();del op;gc.collect();torch.cuda.empty_cache();return rec

def main():
 result={'started':time.time(),'pass':False,'rows':[],'purpose':'diagnose failed A0; not performance evidence'}
 try:
  check_old();xs,orders=load();reference={};changes=[]
  for layout in LAYOUTS:
   for method in PLANS:
    op=Collector();a,rec=op.run(method,xs[layout]);logical=remap(a,orders[layout]);h=output_check(logical)
    u1=remap(np.concatenate(op.u1),orders[layout],True);u2=remap(np.concatenate(op.u2),orders[layout],True)
    if method not in reference:reference[method]=(u1.copy(),u2.copy())
    missing=np.setdiff1d(reference[method][0],u1,assume_unique=True);extra=np.setdiff1d(u1,reference[method][0],assume_unique=True)
    row={'layout':layout,'method':method,'stage_counts':rec['stage_counts'],'output_hash':h,'u1_hash':digest(u1),'u2_hash':digest(u2),'u1_missing':missing.tolist(),'u1_extra':extra.tolist(),'u2_set_equal':np.array_equal(u2,reference[method][1])}
    result['rows'].append(row);print(json.dumps(row),flush=True)
    if method=='F8':changes.extend(missing.tolist()+extra.tolist())
    np.savez_compressed(HERE/'artifacts'/f'diag_{layout}_{method}.npz',u1=u1,u2=u2)
    op.capture();op.close();del op,a,logical;gc.collect();torch.cuda.empty_cache()
  result['minimal_pairs']=[]
  x=full_source()
  for pair in sorted(set(changes))[:8]:
   i,j=divmod(pair,N);forward=classify_pair(x[i],x[j]);reverse=classify_pair(x[j],x[i])
   result['minimal_pairs'].append({'id':pair,'i':i,'j':j,'forward':forward,'reverse':reverse})
  result['pass']=True
 except Exception:result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 finally:
  result['ended']=time.time();(HERE/'results/diagnosis_a0.json').open('x').write(json.dumps(result,indent=2));print(json.dumps(result.get('minimal_pairs')),flush=True)
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
