"""One fixed first full tile batch; NCU attribution only."""
import argparse,gc,json,time
import numpy as np
import torch
from g2_operator import Matrix,HERE,N,CAP,full_source

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['F8','S8','F16','S16']);p.add_argument('--label',required=True);a=p.parse_args();dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
 o=Matrix();x=full_source();v=torch.from_numpy(x).to('cuda');prep=o.prepare(v,8 if a.method.endswith('8') else 16)
 tr,tc=np.triu_indices(938);tr=torch.from_numpy(tr[:4096].astype(np.int32)).to('cuda');tc=torch.from_numpy(tc[:4096].astype(np.int32)).to('cuda')
 out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);c=torch.zeros(6,device='cuda',dtype=torch.int32)
 score=torch.empty(CAP if a.method.startswith('S') else 1,device='cuda',dtype=torch.int32 if a.method.endswith('8') else torch.float32)
 def body():o.stage1(a.method,prep,tr,tc,out,amb,c,score,score,score)
 for _ in range(2):c.zero_();body()
 torch.cuda.synchronize();expected=c.cpu().numpy().copy();c.zero_();torch.cuda.synchronize()
 torch.cuda.nvtx.range_push('G2_TILE_BATCH');body();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
 actual=c.cpu().numpy().copy();assert np.array_equal(expected,actual) and actual[2]==0
 r=dict(pass_=True,method=a.method,grid=4096,block=128,computed_cells=4096*4096,stage1_counts=actual.tolist(),compiled=o.capture(),speed_claim=False)
 dest.write_text(json.dumps(r,indent=2));o.close();print('PASS',a.method,flush=True)
if __name__=='__main__':main()
