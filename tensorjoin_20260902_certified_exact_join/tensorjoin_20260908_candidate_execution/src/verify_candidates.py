"""Independent CPU direct-distance checks and a secondary layout diagnostic."""
import json,time
import numpy as np
from engine import *
def main():
    torch.set_num_threads(4);x,a,r=load();rng=np.random.default_rng(2026090819)
    q=np.load(ROOT/'tensorjoin_20260908_multiscale_probe/artifacts/features_f64.npy')[a['perm']]
    tr,tc=a['tr'],a['tc'];n=len(tr)
    picks=np.unique(np.r_[rng.integers(0,n,192),np.flatnonzero(tr==tc)[[0,1,500,-2,-1]],
                            np.flatnonzero(tc==937)[[0,1,500,-2,-1]]])
    checks=[]
    for cell in output.CELLS:
        all_modes=[]
        for mode in [0,1,2]:
            tg,cg,kg,fg,rg,clg,cnt=candidates(a,r,cell,mode)
            all_modes.append(dict(kind=kg.cpu().numpy(),fine=fg.cpu().numpy(),
                rows=rg.cpu().numpy(),cols=clg.cpu().numpy(),counts=cnt.cpu().numpy()))
        assert all(np.array_equal(all_modes[0]['counts'],v['counts']) for v in all_modes)
        checked_pairs=0;must_survive=0
        for p in picks:
            ri=np.arange(int(tr[p])*64,min((int(tr[p])+1)*64,N))
            ci=np.arange(int(tc[p])*64,min((int(tc[p])+1)*64,N))
            # Explicit differences, no Gram cancellation and no GPU computation.
            d=np.sum((q[ri,None,:]-q[ci][None,:,:])**2,axis=2)
            valid=ri[:,None]<=ci[None,:]
            # Every true-distance candidate in this projected screen must be covered.
            need=valid&(d<=(cell['T']+2**-20)*r['scale2']+2e-5)
            checked_pairs+=int(valid.sum());must_survive+=int(need.sum())
            if need.any():assert all_modes[0]['kind'][p]>0
            flags=all_modes[1]['fine'].reshape(n,4,4)[p]
            for ii in range(4):
                for jj in range(4):
                    if need[ii*16:(ii+1)*16,jj*16:(jj+1)*16].any():assert flags[ii,jj]>0
            v=all_modes[2];kind=int(v['kind'][p])
            if kind==0:assert not need.any()
            elif kind==2:
                chosen=v['rows'].reshape(n,16)[p];chosen=chosen[chosen<N]
                assert len(chosen)==len(np.unique(chosen)) and np.all(np.isin(chosen,ri))
                assert np.all(np.isin(ri[np.any(need,axis=1)],chosen))
            elif kind==3:
                chosen=v['cols'].reshape(n,16)[p];chosen=chosen[chosen<N]
                assert len(chosen)==len(np.unique(chosen)) and np.all(np.isin(chosen,ci))
                assert np.all(np.isin(ci[np.any(need,axis=0)],chosen))
            else:assert kind==1
        counts=all_modes[0]['counts']
        assert counts[tr==tc].min()>=32  # Including the 32-point tail diagonal.
        rec=dict(cell=cell['name'],sample_tiles=len(picks),checked_pairs=checked_pairs,
                 required_candidates=must_survive,survivor_pairs=int(counts.astype(np.int64).sum()),pass_=True)
        checks.append(rec);print(json.dumps(rec),flush=True)
        del all_modes;gc.collect();torch.cuda.empty_cache()
    # Input-only KD permutation, using exactly the same projected coordinates.
    kd=np.load(ROOT/'tensorjoin_20260908_multiscale_probe/artifacts/order_kd_variance.npy').astype(np.int64)
    inverse=np.argsort(a['perm']);newq=a['q'][inverse][kd].copy();newnorm=a['norm'][inverse][kd].copy()
    ka=dict(a,perm=kd,y=x[kd].copy(),q=newq,norm=newnorm)
    op=output.Matrix();radix=output.Radix();cell=next(c for c in output.CELLS if c['name']=='original')
    secondary=[]
    for m in ['layout_full','project64','subtile16','packed16']:
        call(op,radix,x,ka,r,cell,m,exact=True)
        s=call(op,radix,x,ka,r,cell,m,exact=True,diagnostic=True);s['layout']='kd_variance'
        secondary.append(s);print(json.dumps(s),flush=True)
        gc.collect();torch.cuda.empty_cache()
    record=dict(pass_=True,checks=checks,secondary=secondary,compiled_inherited=op.capture(),compiled_new=capture())
    with (HERE/'results/independent_validation.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
