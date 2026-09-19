"""CPU-only exhaustive projected-pair masks and nested block counts."""
import hashlib, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
SWEEP=ROOT/'precision_routing_20260908_threshold_sweep'
OLD=ROOT/'tensorjoin_20260908_block_probe'
N,D=60000,512
RANKS=[16,32,64]
LAYOUTS=['original','random','kd_variance','pca_sort']
PAD=2.**-20
PROJ_PAD=2.**-24
INPUT_HASH='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2)

def emit(**kw):print(json.dumps(kw),flush=True)

def tiled_counts(mask,size=16):
    # All scanned slabs have dimensions divisible by 16, including the tail.
    a,b=mask.shape
    return mask.reshape(a//size,size,b//size,size).sum(axis=(1,3),dtype=np.uint16)

def supports(mask):
    a,b=mask.shape
    rows=np.logical_or.reduceat(mask,np.arange(0,b,64),axis=1)
    nr=np.add.reduceat(rows.astype(np.uint8),np.arange(0,a,64),axis=0,dtype=np.uint16)
    cols=np.logical_or.reduceat(mask,np.arange(0,a,64),axis=0)
    nc=np.add.reduceat(cols.astype(np.uint8),np.arange(0,b,64),axis=1,dtype=np.uint16)
    assert nr.max()<=64 and nc.max()<=64
    return nr.astype(np.uint8),nc.astype(np.uint8)

def controls():
    a=np.zeros((64,64),bool);a[3,[1,2,5,7,9,11]]=True
    c=tiled_counts(a);r,k=supports(a)
    assert c.sum()==6 and np.count_nonzero(c)==1 and r[0,0]==1 and k[0,0]==6
    b=np.ones((96,160),bool)
    rr,cc=supports(b)
    assert np.array_equal(rr,np.array([[64,64,64],[32,32,32]]))
    assert np.array_equal(cc,np.array([[64,64,32],[64,64,32]]))
    assert tiled_counts(b).sum()==96*160
    # Threshold equality and duplicate/self cases must remain candidates.
    z=np.array([[0.,0.],[1.,0.],[1.,0.],[4.,0.]])
    ds=((z[:,None]-z[None])**2).sum(2)
    mask=ds<=1.+PAD
    assert mask[0,1] and mask[1,2] and np.diag(mask).all() and not mask[0,3]
    return dict(pass_=True,six_pair_control=True,ragged=True,duplicates=True,equality=True)

def prepare():
    started=time.perf_counter()
    paths=[ROOT/'data/g2b_cifar60000/vectors_f32.npy',SWEEP/'artifacts/thresholds.json',
           SWEEP/'artifacts/reference_manifest.json',HERE/'PROTOCOL.md',Path(__file__)]
    cells=json.loads(paths[1].read_text())['cells']
    manifest=json.loads(paths[2].read_text())
    frozen={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert frozen[str(paths[0].relative_to(ROOT))]==INPUT_HASH
    for cell in cells:
        p=SWEEP/'artifacts'/f"reference_{cell['name']}.npy"
        h=sha(p);assert h==manifest['cells'][cell['name']]['file_sha256']
        frozen[str(p.relative_to(ROOT))]=h
    for layout in LAYOUTS[:3]:
        p=OLD/'artifacts'/f'order_{layout}.npy'
        frozen[str(p.relative_to(ROOT))]=sha(p)
    write(HERE/'artifacts/input_freeze.json',frozen)
    hash_seconds=time.perf_counter()-started
    t=time.perf_counter();x=np.load(paths[0])
    assert x.shape==(N,D) and x.dtype==np.float32 and np.isfinite(x).all() and np.abs(x).max()<=1
    z=x.astype(np.float64);mu=z.mean(0);centered=z-mu
    cov=centered.T@centered
    vals,vecs=np.linalg.eigh(cov)
    p=np.ascontiguousarray(vecs[:,-64:][:,::-1])
    for j in range(64):
        if p[np.argmax(np.abs(p[:,j])),j]<0:p[:,j]*=-1
    assert np.max(np.abs(p))<=1
    pca_seconds=time.perf_counter()-t
    t=time.perf_counter();q=z@p;score=z@p[:,0]
    feature_seconds=time.perf_counter()-t
    residual=float(np.linalg.norm(p.T@p-np.eye(64),'fro'))
    scale2=1.+residual+PAD
    span=np.ptp(q,axis=0)
    conversion=float(4*PROJ_PAD*np.sum(span+PROJ_PAD))
    gamma=(4*64+32)*2.**-53/(1-(4*64+32)*2.**-53)
    arithmetic=float(gamma*8*64*np.max(np.abs(q))**2)
    error=float(np.nextafter(conversion+arithmetic+PAD,np.inf))
    margins=dict(scale2=scale2,projection_error=PROJ_PAD,conversion=conversion,
        arithmetic=arithmetic,squared_error=error,terminal_pad=PAD,gram_residual=residual)
    np.save(HERE/'artifacts/projector.npy',p)
    np.save(HERE/'artifacts/features_f64.npy',q)
    np.save(HERE/'artifacts/eigenvalues.npy',vals[::-1])
    write(HERE/'artifacts/margins.json',margins)
    meta=dict(hash_seconds=hash_seconds,pca_seconds=pca_seconds,feature_seconds=feature_seconds,
        variance_fractions={str(r):float(vals[-r:].sum()/vals.sum()) for r in RANKS},
        python=sys.version,numpy=np.__version__,platform=platform.platform(),cpu_only=True,
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(HERE/'results/preparation.json',meta)
    emit(stage='prepared',**meta)
    orders={k:np.load(OLD/'artifacts'/f'order_{k}.npy') for k in LAYOUTS[:3]}
    orders['pca_sort']=np.argsort(score,kind='stable').astype(np.int32)
    for name,order in orders.items():
        assert np.array_equal(np.sort(order),np.arange(N))
        np.save(HERE/'artifacts'/f'order_{name}.npy',order)
    return q,cells,manifest,margins,orders

def reference_data(order,cells,manifest):
    """Offline canonical upper-pair IDs and exact 16-tile positive counts."""
    inverse=np.empty(N,np.int32);inverse[order]=np.arange(N,dtype=np.int32)
    keys=[];counts=[]
    for cell in cells:
        ref=np.load(SWEEP/'artifacts'/f"reference_{cell['name']}.npy",mmap_mode='r')
        parts=[]
        for off in range(0,len(ref),1<<20):
            ids=np.asarray(ref[off:off+(1<<20)],np.uint64)
            a=(ids//N).astype(np.int64);b=(ids%N).astype(np.int64)
            keep=a<=b;a=inverse[a[keep]];b=inverse[b[keep]]
            lo=np.minimum(a,b).astype(np.int64);hi=np.maximum(a,b)
            parts.append(lo*N+hi)
        packed=np.concatenate(parts);del parts
        assert len(packed)==manifest['cells'][cell['name']]['upper_count']
        packed.sort();assert np.all(packed[1:]>packed[:-1])
        tile_key=(packed//N//16)*(N//16)+(packed%N//16)
        c=np.bincount(tile_key,minlength=(N//16)**2).reshape(N//16,N//16)
        assert c.max()<=256 and c.sum()==len(packed)
        counts.append(c.astype(np.uint16));keys.append(packed)
    return keys,np.stack(counts)

def save_counts(layout,kind,counts,rows=None,cols=None):
    tri=np.triu_indices(N//16)
    arrays={'counts16':counts[:,tri[0],tri[1]]}
    if rows is not None:
        t64=np.triu_indices((N+63)//64)
        arrays.update(active_rows64=rows[:,t64[0],t64[1]],active_cols64=cols[:,t64[0],t64[1]])
    np.savez_compressed(HERE/'artifacts'/f'counts_{layout}_{kind}.npz',**arrays)

def scan(layout,order,q,cells,manifest,margins):
    start=time.perf_counter();refs,truth=reference_data(order,cells,manifest)
    reference_seconds=time.perf_counter()-start
    t=time.perf_counter();save_counts(layout,'oracle',truth);del truth
    reference_save_seconds=time.perf_counter()-t
    z=q[order];nb=(N+63)//64
    counts=np.zeros((3,6,N//16,N//16),np.uint16)
    active_rows=np.zeros((3,6,nb,nb),np.uint8);active_cols=np.zeros_like(active_rows)
    checks=np.zeros((3,6),np.int64)
    thresholds=np.array([c['T'] for c in cells])
    cuts=(thresholds+PAD)*margins['scale2']+margins['squared_error']
    scan_start=time.perf_counter();matmul_seconds=0.;mask_seconds=0.
    bands=[(0,16),(16,32),(32,64)]
    for off in range(0,N,256):
        end=min(off+256,N);ni=end-off;nj=N-off
        valid=np.arange(off,end)[:,None]<=np.arange(off,N)[None,:]
        projected=np.zeros((ni,nj),np.float64)
        positives=[]
        for ref in refs:
            ids=ref[np.searchsorted(ref,off*N):np.searchsorted(ref,end*N)]
            positives.append((ids//N-off,ids%N-off))
        for ri,(a,b) in enumerate(bands):
            t=time.perf_counter()
            za=z[off:end,a:b];zb=z[off:,a:b]
            band=-2.*(za@zb.T)
            band+=np.sum(za*za,axis=1)[:,None]
            band+=np.sum(zb*zb,axis=1)[None,:]
            np.maximum(band,0.,out=band);projected+=band;del band
            matmul_seconds+=time.perf_counter()-t
            t=time.perf_counter()
            for ci,cut in enumerate(cuts):
                mask=(projected<=cut)&valid
                ia,ib=positives[ci]
                assert mask[ia,ib].all(),(layout,RANKS[ri],cells[ci]['name'],off,'false reject')
                checks[ri,ci]+=len(ia)
                c=tiled_counts(mask)
                counts[ri,ci,off//16:end//16,off//16:]=c
                nr,nc=supports(mask)
                active_rows[ri,ci,off//64:(end+63)//64,off//64:]=nr
                active_cols[ri,ci,off//64:(end+63)//64,off//64:]=nc
            mask_seconds+=time.perf_counter()-t
        if off==0 or (off//256+1)%32==0 or end==N:
            emit(stage='scan',layout=layout,rows_done=end,seconds=time.perf_counter()-scan_start)
    scan_seconds=time.perf_counter()-scan_start
    for ri in range(3):
        assert (checks[ri]==np.array([len(ref) for ref in refs])).all()
    assert np.all(counts[0]>=counts[1]) and np.all(counts[1]>=counts[2])
    assert np.all(counts[:,:-1]<=counts[:,1:])
    t=time.perf_counter()
    for ri,r in enumerate(RANKS):save_counts(layout,f'r{r}',counts[ri],active_rows[ri],active_cols[ri])
    record=dict(layout=layout,reference_seconds=reference_seconds,reference_save_seconds=reference_save_seconds,
        scan_seconds=scan_seconds,matmul_seconds=matmul_seconds,mask_and_reduction_seconds=mask_seconds,
        save_seconds=time.perf_counter()-t,positive_pair_checks=checks.tolist(),false_reject_pairs=0,
        monotonicity_pass=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'results'/f'{layout}.json',record);emit(stage='layout_complete',**record)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'results/probe.json').exists()
    started=time.time();write(HERE/'results/controls.json',controls())
    q,cells,manifest,margins,orders=prepare()
    for name in LAYOUTS:scan(name,orders[name],q,cells,manifest,margins)
    write(HERE/'results/probe.json',dict(pass_=True,started=started,ended=time.time(),
        layouts=LAYOUTS,ranks=RANKS,block_sizes=[16,32,64,256],cells=[c['name'] for c in cells],
        scanned_upper_pairs_per_layout_rank=N*(N+1)//2,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':main()
