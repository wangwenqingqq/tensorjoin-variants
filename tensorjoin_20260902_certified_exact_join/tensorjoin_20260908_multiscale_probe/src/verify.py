"""Independent direct-difference and reference-search checks on fixed blocks."""
import json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
N=60000
CELLS=['k4','k16','k64','original','k256','k1024']
LAYOUTS=['original','random','kd_variance','pca_sort']
RANKS=[16,32,64]

def tri_index(a,b,n):
    assert a<=b
    return a*n-a*(a+1)//2+b

def main():
    start=time.perf_counter()
    q=np.load(HERE/'artifacts/features_f64.npy')
    x=np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy')
    p=np.load(HERE/'artifacts/projector.npy')
    margin=json.loads((HERE/'artifacts/margins.json').read_text())
    cells=json.loads((ROOT/'precision_routing_20260908_threshold_sweep/artifacts/thresholds.json').read_text())['cells']
    refs=[np.load(ROOT/'precision_routing_20260908_threshold_sweep/artifacts'/f'reference_{c}.npy',mmap_mode='r') for c in CELLS]
    rng=np.random.Generator(np.random.PCG64(2026090837))
    ids=rng.choice(N,257,replace=False)
    direct_projection=np.empty((len(ids),64))
    for k in range(64):direct_projection[:,k]=np.sum(x[ids].astype(np.float64)*p[:,k],axis=1)
    max_proj_error=float(np.max(np.abs(direct_projection-q[ids])))
    assert max_proj_error<margin['projection_error']
    blocks=[]
    # Sample is fixed independently of outcomes; add diagonal and both tails.
    for a,b in rng.integers(0,(N+63)//64,size=(24,2)):
        a,b=sorted([int(a),int(b)]);blocks.append((a*64,b*64,64))
    blocks.extend([(0,0,64),(128,128,64),(59968,59968,64),(0,59968,64)])
    for a,b in rng.integers(0,N//256,size=(3,2)):
        a,b=sorted([int(a),int(b)]);blocks.append((a*256,b*256,256))
    blocks.extend([(59904,59904,256),(0,59904,256)])
    checks=0;reference_queries=0;support_checks=0;max_gram_delta=0.;bound_checks=0
    for layout in LAYOUTS:
        order=np.load(HERE/'artifacts'/f'order_{layout}.npy')
        archives={kind:np.load(HERE/'artifacts'/f'counts_{layout}_{kind}.npz') for kind in ['oracle','r16','r32','r64']}
        saved={kind:dict(z) for kind,z in archives.items()}
        for a,b,size in blocks:
            ia=order[a:min(a+size,N)];ib=order[b:min(b+size,N)]
            valid=np.arange(a,a+len(ia))[:,None]<=np.arange(b,b+len(ib))[None,:]
            z=q[ia,None,:]-q[None,ib,:]
            ds=np.cumsum(z*z,axis=2)
            keys=(ia[:,None].astype(np.uint64)*N+ib[None,:]).ravel()
            for ci,ref in enumerate(refs):
                at=np.searchsorted(ref,keys);hit=np.zeros(len(keys),bool)
                good=at<len(ref);hit[good]=ref[at[good]]==keys[good]
                hit=hit.reshape(len(ia),len(ib))&valid
                reference_queries+=len(keys)
                masks={'oracle':hit}
                cut=(cells[ci]['T']+margin['terminal_pad'])*margin['scale2']+margin['squared_error']
                for r in RANKS:
                    mask=(ds[:,:,r-1]<=cut)&valid
                    assert not np.any(hit&~mask)
                    masks[f'r{r}']=mask
                for kind,mask in masks.items():
                    for u in range(0,len(ia),16):
                        for v in range(0,len(ib),16):
                            if a+u>b+v:continue
                            idx=tri_index((a+u)//16,(b+v)//16,N//16)
                            got=int(mask[u:u+16,v:v+16].sum())
                            assert got==int(saved[kind]['counts16'][ci,idx]),(layout,a,b,kind,ci,u,v,got)
                            checks+=1
                    if kind!='oracle':
                        for u in range(0,len(ia),64):
                            for v in range(0,len(ib),64):
                                if a+u>b+v:continue
                                idx=tri_index((a+u)//64,(b+v)//64,(N+63)//64)
                                sub=mask[u:u+64,v:v+64]
                                nr=sum(any(row) for row in sub)
                                nc=sum(any(col) for col in sub.T)
                                assert nr==int(saved[kind]['active_rows64'][ci,idx])
                                assert nc==int(saved[kind]['active_cols64'][ci,idx])
                                support_checks+=2
            # Direct original distance verifies the projected inequality here,
            # independently of the reference membership lookup.
            flat_i=np.repeat(ia,len(ib));flat_j=np.tile(ib,len(ia))
            original=np.empty(len(flat_i))
            for off in range(0,len(flat_i),512):
                diff=x[flat_i[off:off+512]].astype(np.float64)-x[flat_j[off:off+512]].astype(np.float64)
                original[off:off+512]=np.sum(diff*diff,axis=1)
            lower=np.maximum(0.,ds[:,:,-1]-margin['squared_error'])/margin['scale2']
            assert np.all(lower.ravel()<=original+margin['terminal_pad'])
            bound_checks+=len(flat_i)
            gram=np.sum(q[ia]**2,axis=1)[:,None]+np.sum(q[ib]**2,axis=1)[None,:]-2*q[ia]@q[ib].T
            delta=float(np.max(np.abs(gram-ds[:,:,-1])))
            max_gram_delta=max(max_gram_delta,delta)
            assert delta<margin['arithmetic']+2.**-20
        for z in archives.values():z.close()
        print(json.dumps(dict(verified=layout,seconds=time.perf_counter()-start)),flush=True)
    result=dict(pass_=True,seed=2026090837,blocks_per_layout=len(blocks),layouts=LAYOUTS,
        independent_tile_count_checks=checks,active_row_col_checks=support_checks,
        exact_reference_pair_lookups=reference_queries,direct_full_dimension_bound_checks=bound_checks,
        max_projection_difference=max_proj_error,max_direct_vs_gram_difference=max_gram_delta,
        seconds=time.perf_counter()-start,
        scope='Independent fixed-block check complements exhaustive reference-positive survival checks; not a new universal numerical proof.')
    (HERE/'results/independent_verification.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
