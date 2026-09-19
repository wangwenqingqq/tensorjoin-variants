"""Changed-layout selected-tile composites, using only unchanged GPU kernels."""
import argparse,gc,json,os,time,traceback
import numpy as np
import torch
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);args=p.parse_args()
 result={'started':time.time(),'pid':os.getpid(),'label':args.label,'pass':False,'rows':[]}
 try:
  result['old_identities']=check_old();census=json.loads((HERE/'results/census_a0.json').read_text());assert census['pass']
  xs,orders=load();truth=np.fromfile(DGS/'artifacts/reference_ids_u64.bin',dtype='<u8');assert output_check(truth)
  tr,tc=np.triu_indices(M)
  for layout in LAYOUTS:
   order=orders[layout];inv=np.empty(N,dtype=np.int64);inv[order.astype(np.int64)]=np.arange(N)
   rr=inv[(truth//N).astype(np.int64)]//B;cc=inv[(truth%N).astype(np.int64)]//B
   rr,cc=np.minimum(rr,cc),np.maximum(rr,cc)
   for method in PLANS:
    hist=np.load(HERE/'artifacts'/f'{layout}_{method}_census.npz')['hist1']
    selected=np.unique(np.concatenate([np.linspace(0,len(tr)-1,128,dtype=np.int64),np.argsort(-hist,kind='stable')[:32],np.flatnonzero(tc==M-1)[-16:],np.array([0,len(tr)-1])]))
    schedule=xs[layout][1];selected=schedule[np.isin(schedule,selected)]
    tileset=np.zeros((M,M),bool);tileset[tr[selected],tc[selected]]=True
    want=truth[tileset[rr,cc]]
    op=Matrix();v=torch.from_numpy(xs[layout][0]).to('cuda');prep=op.prepare(v,8 if method=='F8' else 16)
    trg=torch.tensor(tr[selected],device='cuda',dtype=torch.int32);tcg=torch.tensor(tc[selected],device='cuda',dtype=torch.int32)
    out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out);c=torch.zeros(6,device='cuda',dtype=torch.int32)
    score=torch.empty(1,device='cuda',dtype=torch.int32 if method=='F8' else torch.float32);dump=score.view(torch.float32)
    op.stage1(method,prep,trg,tcg,out,amb,c,score,dump,dump);s1=c.cpu().numpy().copy();assert s1[2]==0
    op.stage2(v,amb,out,fp64,c,int(s1[1]));s2=c.cpu().numpy().copy();assert s2[2]==0
    op.terminal(v,fp64,out,c,int(s2[3]));last=c.cpu().numpy().copy();assert last[2]==0
    ids=out[:int(last[0])].cpu().numpy().astype(np.uint64,copy=True);r=ids//N;col=ids%N
    physical=np.concatenate([ids,col[r<col]*N+r[r<col]])
    got=remap(physical,order);assert np.array_equal(got,want),(layout,method,len(got),len(want))
    result['rows'].append({'layout':layout,'method':method,'tiles':len(selected),'u1':int(s1[1]),'u2':int(s2[3]),'count':len(got),'output_sha':digest(got),'pass':True,'compiled':op.capture()})
    print(json.dumps({k:v for k,v in result['rows'][-1].items() if k!='compiled'}),flush=True)
    del v,prep,trg,tcg,out,amb,fp64,c,score,dump;op.close();del op;gc.collect();torch.cuda.empty_cache()
  result['pass']=True
 except Exception:result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 finally:
  result['ended']=time.time();(HERE/'results'/f'{args.label}.json').open('x').write(json.dumps(result,indent=2))
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
