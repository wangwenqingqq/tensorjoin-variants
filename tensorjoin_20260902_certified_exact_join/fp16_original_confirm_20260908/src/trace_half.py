"""Exact selected-call trace, not latency measurement or a full certificate."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
import platform
import time
import traceback
import numpy as np
import torch
from owned_half import OwnedCuBLAS
from frozen_driver import HERE, ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',type=int,choices=range(4),required=True);p.add_argument('--label',required=True);a=p.parse_args()
    dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    specs=[(0,0,512,512),(0,0,4096,4096),(0,57344,4096,2656),(57344,57344,2656,2656)]
    row,col,m,n=specs[a.shape]
    r={'label':a.label,'shape':[m,n,512],'offset':[row,col],'pass':False,'pid':os.getpid(),'host':platform.node(),'started':time.time(),'speed_claim':False}
    b=None
    try:
        assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='2'
        x=np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy',mmap_mode='r')
        assert sha(ROOT/'data/g2b_cifar60000/vectors_f32.npy')=='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
        if a.shape==0:
            y=np.load(ROOT/'fp16_dense_gate0_20260907/results/fixture_a0_fixture512.npy')
            hostq,hostb=y,y
        else:hostq,hostb=x[row:row+m],x[col:col+n]
        def cast(v):
            t=torch.from_numpy(np.array(v,copy=True)).to('cuda').to(torch.float16)
            return torch.where(t.abs()<2**-14,torch.zeros_like(t),t)
        q,base=cast(hostq),cast(hostb)
        out=torch.empty((m,n),device='cuda',dtype=torch.float32)
        b=OwnedCuBLAS();r['binding']=b.record()
        r['libraries']={name:sha(__import__('pathlib').Path(name)) for name in json.loads((HERE/'library_manifest.json').read_text())}
        assert r['libraries']==json.loads((HERE/'library_manifest.json').read_text())
        for _ in range(2):b.gemm(q,base,out)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('FP16_OWNED')
        b.gemm(q,base,out)
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
        dot=out.cpu().numpy();qc=q.cpu().numpy();bc=base.cpu().numpy()
        rng=np.random.default_rng(202609072)
        pairs=[(i,j) for i in range(min(16,m)) for j in range(min(16,n))]
        pairs += [(int(i),int(j)) for i,j in zip(rng.integers(m,size=128),rng.integers(n,size=128))]
        gamma=F(1026,2**23-1026);checks=[]
        for i,j in pairs:
            aa=[F(float(v)) for v in qc[i]];bb=[F(float(v)) for v in bc[j]]
            exact=sum(v*w for v,w in zip(aa,bb));error=abs(F(float(dot[i,j]))-exact)
            norm2=sum(v*v for v in aa)*sum(v*v for v in bb)
            ok=error*error <= gamma*gamma*norm2
            checks.append({'i':i,'j':j,'actual':float(dot[i,j]),'error':str(error),'pass':ok})
        r['dot_checks']=checks
        np.savez_compressed(HERE/'results'/f'{a.label}_dot_samples.npz',
                            q=qc[[i for i,j in pairs]],b=bc[[j for i,j in pairs]],
                            p=np.array([dot[i,j] for i,j in pairs],np.float32))
        assert all(x['pass'] for x in checks),'dot envelope failure'
        r['pass']=True
    except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        if b is not None:b.close()
        b=None
        import gc
        gc.collect();torch.cuda.empty_cache()
        r['ended']=time.time()
        with dest.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({'pass':r['pass'],'output':str(dest)}),flush=True)
    return 0 if r['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
