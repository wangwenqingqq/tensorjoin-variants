import hashlib,json,os,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
OLD=ROOT/'tensorjoin_20260908_multiscale_probe'
GAMMA=0.00012232370499987155
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(x,save=False):
    t=time.perf_counter();z=x.astype(np.float64);centered=z-z.mean(0)
    vals,vecs=np.linalg.eigh(centered.T@centered)
    p=np.ascontiguousarray(vecs[:,-64:][:,::-1])
    for j in range(64):
        if p[np.argmax(np.abs(p[:,j])),j]<0:p[:,j]*=-1
    q=z@p;perm=np.argsort(z@p[:,0],kind='stable').astype(np.int64)
    projection_done=time.perf_counter()
    h=q.astype(np.float16);h[np.abs(h)<2**-14]=0
    hh=h.astype(np.float64)
    H=float(np.sqrt(np.max(np.sum(hh*hh,1)))+2**-24)
    E=float(np.sqrt(np.max(np.sum((q-hh)**2,1)))+8*2**-24)
    scale2=float(1+np.linalg.norm(p.T@p-np.eye(64),'fro')+2**-20)
    margin=float(8*H*E+4*E*E+(2*GAMMA+16*2**-24)*H*H+2**-20)
    norm=np.sum(hh*hh,1).astype(np.float32)
    y=np.ascontiguousarray(x[perm]);hs=np.ascontiguousarray(h[perm]);ns=norm[perm].copy()
    metadata_done=time.perf_counter()
    tr,tc=np.triu_indices((len(x)+63)//64);tr=tr.astype(np.int32);tc=tc.astype(np.int32)
    groups=[hs[i:i+64].astype(np.float64) for i in range(0,len(x),64)]
    lo=np.array([v.min(0) for v in groups]);hi=np.array([v.max(0) for v in groups])
    bbox=np.empty(len(tr),np.float64)
    for o in range(0,len(tr),4096):
        r,c=tr[o:o+4096],tc[o:o+4096]
        gap=np.maximum(0,np.maximum(lo[r]-hi[c],lo[c]-hi[r]))
        bbox[o:o+len(r)]=np.sum(gap*gap,1)
    rec=dict(seconds=time.perf_counter()-t,projection_order_seconds=projection_done-t,
        metadata_reorder_seconds=metadata_done-projection_done,
        bbox_seconds=time.perf_counter()-metadata_done,H=H,E=E,scale2=scale2,margin=margin)
    arrays=dict(y=y,q=hs,norm=ns,perm=perm,tr=tr,tc=tc,bbox=bbox)
    if save:
        for k,a in arrays.items():np.save(HERE/'artifacts'/f'{k}.npy',a)
        (HERE/'artifacts/preparation.json').write_text(json.dumps(rec,indent=2))
        paths=[ROOT/'data/g2b_cifar60000/vectors_f32.npy',OLD/'artifacts/order_pca_sort.npy',OLD/'artifacts/projector.npy']
        (HERE/'artifacts/input_freeze.json').write_text(json.dumps({str(p):digest(p) for p in paths},indent=2))
    return arrays,rec
if __name__=='__main__':
    x=np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy')
    assert digest(ROOT/'data/g2b_cifar60000/vectors_f32.npy')=='95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
    a,r=build(x,True)
    assert np.array_equal(a['perm'],np.load(OLD/'artifacts/order_pca_sort.npy'))
    print(json.dumps(r),flush=True)
