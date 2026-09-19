"""CPU structural screen; no CUDA or empirical-recall filter is used here."""
import hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
N,D,B,T=60000,512,64,25921/65536
PREFIXES=(16,64,512)
U=2.0**-53
G=lambda k: (k*U)/(1-k*U)
ORACLE_ERROR=2.0**-28
assert 2048*G(4096)+2.0**-970 < ORACLE_ERROR
THRESHOLD=np.nextafter(T+ORACLE_ERROR,np.inf)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(8<<20),b''): h.update(a)
    return h.hexdigest()

def hadamard(x):
    y=x.astype(np.float64,copy=True)
    for width in (1,2,4,8,16,32,64,128,256):
        a=y.reshape(len(y),-1,2*width)
        left=a[:,:,:width].copy();right=a[:,:,width:].copy()
        a[:,:,:width]=left+right;a[:,:,width:]=left-right
    error=np.nextafter(G(9)*np.sum(np.abs(x),axis=1,dtype=np.float64)/(1-G(D))+2.0**-970,np.inf)
    return y,error

def tree_order(y):
    leaves=[];nodes=[]
    def visit(ids):
        if len(ids)<=B:leaves.append(ids);return
        z=y[ids];axis=int(np.argmax(np.var(z,axis=0,dtype=np.float64)))
        ids=ids[np.lexsort((ids,z[:,axis]))]
        split=((len(ids)+B-1)//B//2)*B
        nodes.append([len(ids),axis,split])
        visit(ids[:split]);visit(ids[split:])
    visit(np.arange(len(y),dtype=np.int32))
    return np.concatenate(leaves),nodes

def envelope(y,error,order,axes):
    z=y[order][:,axes]
    e=error[order] if np.ndim(error) else np.full(len(z),error)
    low=np.nextafter(z-e[:,None],-np.inf);high=np.nextafter(z+e[:,None],np.inf)
    starts=np.arange(0,len(z),B)
    lo=np.minimum.reduceat(low,starts,axis=0);hi=np.maximum.reduceat(high,starts,axis=0)
    idx=np.arange(len(z))//B
    assert np.all(lo[idx]<=low) and np.all(hi[idx]>=high)
    return lo,hi

def lower_values(lo,hi,scale,rows,cols):
    delta=np.maximum(lo[rows]-hi[cols],lo[cols]-hi[rows])
    gap=np.maximum(0,np.nextafter(delta,-np.inf))
    sq=np.nextafter(gap*gap,0)
    acc=np.cumsum(sq,axis=-1)
    # Any nonnegative reduction of at most D terms has this conservative bound.
    return np.nextafter(acc[...,np.array(PREFIXES)-1]/(1+G(D))/scale,0)

def test():
    rng=np.random.default_rng(20260908)
    x=rng.uniform(-1,1,(128,D)).astype(np.float32)
    x[:B]=0;x[B:]=np.float32(.2)
    y,e=hadamard(x)
    assert np.allclose(np.sum((y[:B]-y[B:])**2,axis=1)/D,np.sum((x[:B].astype(float)-x[B:])**2,axis=1),atol=1e-11,rtol=1e-12)
    for z,err,scale in [(x.astype(float),0,1),(y,e,D)]:
        lo,hi=envelope(z,err,np.arange(128),np.arange(D))
        lb=lower_values(lo,hi,scale,np.array([0]),np.array([1]))[0,-1]
        assert lb>THRESHOLD
        assert lb<=np.sum((x[0].astype(float)-x[B].astype(float))**2)
    x[:]=0;x[B:,0]=np.float32(161/256)
    for value in [np.float32(161/256),np.nextafter(np.float32(161/256),np.float32(0)),np.nextafter(np.float32(161/256),np.float32(np.inf))]:
        x[B:,0]=value
        for z,err,scale in [(x.astype(float),0,1),(*hadamard(x),D)]:
            lo,hi=envelope(z,err,np.arange(128),np.arange(D));lb=lower_values(lo,hi,scale,np.array([0]),np.array([1]))[0,-1]
            truth=float(value)**2
            assert lb<=truth+1e-15
            if truth<=T:assert lb<=THRESHOLD
    # High-entropy and wide-exponent transform consistency.
    wide=np.ldexp(rng.uniform(-1,1,(256,D)),rng.integers(-145,0,(256,D))).astype(np.float32)
    y,e=hadamard(wide)
    a=wide[:128].astype(float)-wide[128:].astype(float)
    b=y[:128]-y[128:]
    assert np.allclose(np.sum(a*a,axis=1),np.sum(b*b,axis=1)/D,atol=1e-11,rtol=1e-12)
    return {'pass':True,'boundary':True,'separable':True,'wide_exponent':True,'transform_consistency':True}

def main():
    result={'experiment':'tensorjoin_20260908_directional_envelopes_small_exact','started':time.time(),'pid':os.getpid(),'host':platform.node(),'python':platform.python_version(),'numpy':np.__version__,'mode':'CPU structural; not GPU timing','unit_tests':test(),'rows':[]}
    p=ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    assert sha(p)=='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
    x=np.load(p,allow_pickle=False);assert x.shape==(N,D) and x.dtype==np.float32 and np.isfinite(x).all() and np.max(np.abs(x))<=1
    x=x.astype(np.float64)
    t=time.perf_counter();h,he=hadamard(x);result['hadamard_build_seconds']=time.perf_counter()-t
    axes={key:np.argsort(-np.var(y,axis=0),kind='stable') for key,y in [('raw',x),('hadamard',h)]}
    np.savez(HERE/'artifacts/axes.npz',**axes)
    result['oracle_error_allowance']=ORACLE_ERROR
    result['preprocessing']={};M=(N+B-1)//B
    tr,tc=np.triu_indices(M);counts=np.minimum(B,N-np.arange(M)*B)
    pairweights=counts[tr]*counts[tc];diag=tr==tc
    pairweights[diag]=counts[tr[diag]]*(counts[tr[diag]]+1)//2
    assert int(pairweights.sum())==N*(N+1)//2
    orders={};trees={}
    for key,y in [('original',None),('raw_tree',x),('hadamard_tree',h)]:
        t=time.perf_counter()
        order,nodes=(np.arange(N,dtype=np.int32),[]) if y is None else tree_order(y)
        assert np.array_equal(np.sort(order),np.arange(N))
        orders[key]=order;trees[key]=nodes
        result['preprocessing'][key]={'tree_seconds':time.perf_counter()-t,'nodes':len(nodes)}
    np.savez(HERE/'artifacts/orders.npz',**orders)
    (HERE/'artifacts/trees.json').write_text(json.dumps(trees))
    masks={}; lower_summary={}
    rng=np.random.default_rng(20260908)
    for key,order in orders.items():
        components={}
        for space,y,error,scale in [('raw',x,0,1),('hadamard',h,he,D)]:
            t=time.perf_counter();lo,hi=envelope(y,error,order,axes[space]);summary_time=time.perf_counter()-t
            np.savez(HERE/'artifacts'/f'{key}_{space}_envelopes.npz',lo=lo,hi=hi)
            t=time.perf_counter();lb=np.empty((len(tr),3),dtype=np.float64)
            for off in range(0,len(tr),4096):
                sl=slice(off,off+4096);lb[sl]=lower_values(lo,hi,scale,tr[sl],tc[sl])
            filter_time=time.perf_counter()-t
            assert np.all(lb[:,1]>=lb[:,0]) and np.all(lb[:,2]>=lb[:,1])
            sample=rng.integers(0,len(tr),8192)
            ra=tr[sample]*B+rng.integers(0,counts[tr[sample]])
            rb=tc[sample]*B+rng.integers(0,counts[tc[sample]])
            delta=x[order[ra]]-x[order[rb]]
            dist=np.sum(delta*delta,axis=1)
            assert np.all(lb[sample,-1]<=dist+ORACLE_ERROR)
            components[space]=lb
            result['preprocessing'][key][space]={'summary_seconds':summary_time,'filter_seconds':filter_time,'summary_bytes':lo.nbytes+hi.nbytes,'sample_bound_checks':8192}
        combined=np.maximum(components['raw'],components['hadamard'])
        for space,lb in list(components.items())+[('combined',combined)]:
            for j,prefix in enumerate(PREFIXES):
                mask=lb[:,j]>THRESHOLD
                name=f'{key}__{space}__{prefix}';masks[name]=mask
                row={'order':key,'space':space,'prefix':prefix,'pruned_tiles':int(mask.sum()),'total_tiles':len(tr),'tile_skip_fraction':float(mask.mean()),'eliminated_valid_upper_pairs':int(pairweights[mask].sum()),'valid_upper_pairs':int(pairweights.sum()),'max_lower_d2':float(lb[:,j].max()),'lower_d2_quantiles':np.quantile(lb[:,j],[0,.5,.9,.99,1]).tolist(),'structural_20pct':bool(mask.mean()>=.2)}
                result['rows'].append(row);print(json.dumps(row),flush=True)
            lower_summary[f'{key}__{space}']=lb
    np.savez_compressed(HERE/'artifacts/masks.npz',tr=tr,tc=tc,**masks)
    np.savez_compressed(HERE/'artifacts/lower_bounds.npz',**lower_summary)
    result['any_count_gate_pass']=any(r['structural_20pct'] for r in result['rows'])
    result['correctness_status']='unit/envelope/random-pair passed; full output ID coverage pending'
    result['complete']=True;result['ended']=time.time()
    (HERE/'results/structural_a0.json').open('x').write(json.dumps(result,indent=2))
    print(json.dumps({'complete':True,'count_gate':result['any_count_gate_pass'],'elapsed_seconds':result['ended']-result['started']}),flush=True)
if __name__=='__main__':main()
