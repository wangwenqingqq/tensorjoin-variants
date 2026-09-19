"""Query-independent data summaries and exact offline tile labels."""
from common import *

N,D,BLOCK=60000,512,64
NB=(N+BLOCK-1)//BLOCK
BANDS=[(0,1),(1,4),(4,8),(8,16),(16,32)]
CONFIGS=['pca','all','heuristic25','random25','direct25','fm05','fm10','fm25']
METHODS=['F8','F16']
BUDGETS={'pca':0.,'all':1.,'heuristic25':.25,'random25':.25,'direct25':.25,'fm05':.05,'fm10':.10,'fm25':.25}


def tile_index(r,c):
    lo=np.minimum(r,c).astype(np.int64);hi=np.maximum(r,c).astype(np.int64)
    return lo*NB-lo*(lo+1)//2+hi


def summaries(x,q,perm,b):
    start=time.perf_counter()
    z=x[perm].astype(np.float64);qq=q[perm]
    gx=[z[i:i+64] for i in range(0,N,64)]
    gq=[qq[i:i+64] for i in range(0,N,64)]
    mx=np.asarray([a.mean(0) for a in gx]);mq=np.asarray([a.mean(0) for a in gq])
    vq=np.asarray([a.var(0) for a in gq])
    lo=np.asarray([a.min(0) for a in gq]);hi=np.asarray([a.max(0) for a in gq])
    norm=np.sum(z*z,axis=1)
    mn=np.asarray([norm[i:i+64].mean() for i in range(0,N,64)])
    nlo=np.asarray([np.sqrt(norm[i:i+64].min()) for i in range(0,N,64)])
    nhi=np.asarray([np.sqrt(norm[i:i+64].max()) for i in range(0,N,64)])
    radius=np.asarray([np.sqrt(np.sum((a-c)**2,axis=1).max()) for a,c in zip(gx,mx)])
    center_d2=np.maximum(0.,np.sum(mx*mx,axis=1)[:,None]+np.sum(mx*mx,axis=1)[None,:]-2*mx@mx.T)
    mean_pair=mn[:,None]+mn[None,:]-2*mx@mx.T
    tr,tc=b['tr'],b['tc'];m=len(tr)
    features=np.empty((m,20),np.float64);coarse=np.empty(m,np.float64)
    for off in range(0,m,16384):
        sl=slice(off,off+16384);i,j=tr[sl],tc[sl]
        delta2=(mq[i]-mq[j])**2;variance=vq[i]+vq[j]
        gaps=np.maximum(0.,np.maximum(lo[i]-hi[j],lo[j]-hi[i]))**2
        for k,(a,e) in enumerate(BANDS):
            features[sl,k]=delta2[:,a:e].sum(1)
            features[sl,5+k]=variance[:,a:e].sum(1)
            features[sl,10+k]=gaps[:,a:e].sum(1)
        features[sl,15]=center_d2[i,j]
        features[sl,16]=mean_pair[i,j]
        features[sl,17]=(radius[i]+radius[j])**2
        features[sl,18]=np.maximum(0.,np.maximum(nlo[i]-nhi[j],nlo[j]-nhi[i]))**2
        features[sl,19]=b['combined'][sl]
        # A cheap, untrusted minimum-distance heuristic, shared by all models.
        spread=np.sqrt(2*(variance**2).sum(1)+4*(delta2*variance).sum(1))
        coarse[sl]=mean_pair[i,j]-2*spread
    assert np.isfinite(features).all() and np.isfinite(coarse).all()
    return features.astype(np.float32),coarse.astype(np.float32),time.perf_counter()-start


def numerical_margin(q):
    f=q.astype(np.float32).astype(np.float64)
    delta=np.nextafter(np.max(np.abs(q-f),axis=0)+PROJ_PAD,np.inf)
    span=np.nextafter(np.ptp(f,axis=0),np.inf)
    unit=2.**-24;gamma=128*unit/(1-128*unit)
    conversion=float(4*np.sum(delta*(span+delta)))
    arithmetic=float(gamma*np.sum(span*span))
    error=np.nextafter(conversion+arithmetic+SQ_PAD,np.inf)
    return dict(error=float(error),conversion_error=conversion,fp32_arithmetic_error=arithmetic,
        gamma128=gamma,projection_coordinate_error=PROJ_PAD,delta=delta.tolist(),span=span.tolist(),
        scope='FP32 direct differences and FMA sums over <=32 coordinates; no dot-product cancellation; inherited projection/operator and FP64 predicate margins.')


def exact_counts(perm,allowed=None):
    btr,btc=np.triu_indices(NB);m=len(btr)
    if allowed is None:allowed=np.ones(m,bool)
    inv=np.empty(N,np.int64);inv[perm]=np.arange(N)
    result=np.full((len(CELLS),m),-1,np.int32)
    for ci,cell in enumerate(CELLS):
        counts=np.zeros(m,np.int64)
        ref=np.load(SWEEP/'artifacts'/f"reference_{cell['name']}.npy",mmap_mode='r')
        for off in range(0,len(ref),1<<20):
            ids=np.asarray(ref[off:off+(1<<20)],np.uint64)
            r=inv[(ids//N).astype(np.int64)]//64;c=inv[(ids%N).astype(np.int64)]//64
            index=tile_index(r,c);index=index[allowed[index]]
            counts+=np.bincount(index,minlength=m)
        assert counts.max()<=8192
        result[ci,allowed]=counts[allowed].astype(np.int32)
    return result


def initial_state(coarse,threshold):
    return np.tanh((np.float32(threshold)-coarse)/(np.float32(.25)*np.float32(threshold))).astype(np.float32)


def eligible(b,cell,subset=None):
    mask=(b['combined']<=cell['T']+SQ_PAD)&(b['tr']!=b['tc'])
    if subset is not None:mask &= subset
    return np.flatnonzero(mask).astype(np.int64)


def top_budget(scores,ids,fraction):
    n=int(np.ceil(len(ids)*fraction))
    if n==0:return np.empty(0,np.int64)
    # Ascending occupancy prediction; stable ID tie-breaking.
    return ids[np.argsort(scores,kind='stable')[:n]]
