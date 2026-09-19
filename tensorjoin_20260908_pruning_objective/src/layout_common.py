import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join')
OLD = Path('@TENSORJOIN_ROOT@/tensorjoin_20260908_block_feasibility')
SWEEP = ROOT/'precision_routing_20260908_threshold_sweep'
sys.path.insert(0,str(OLD/'src'))
from geometry import projection, bbox_lower, SQ_PAD, PROJ_PAD, digest
from screen import occupied

CELLS = json.loads((SWEEP/'artifacts/thresholds.json').read_text())['cells']
FAMILIES = ['pca','proxy','direct']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,indent=2,default=str)

def source():
    p = ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    expected = json.loads((SWEEP/'artifacts/thresholds.json').read_text())['input_sha256']
    assert sha(p)==expected
    x = np.load(p)
    assert x.shape==(60000,512) and x.dtype==np.float32 and np.isfinite(x).all()
    return x

def make_features(x):
    start = time.perf_counter()
    p,scale2 = projection(x)
    assert np.max(np.abs(p))<=1
    projected = time.perf_counter()
    z = x.astype(np.float64)
    q = z @ p
    # Match the previous PCA sorting arithmetic exactly: a vector dot rather
    # than a selected column of a potentially differently reduced GEMM.
    score = z @ p[:,0]
    end = time.perf_counter()
    return p,q,score,scale2,dict(seconds=end-start,pca_seconds=projected-start,
                                feature_seconds=end-projected)

def projected_bounds(q,perm,scale2):
    start = time.perf_counter()
    z = q[perm]
    groups = [z[i:i+64] for i in range(0,len(z),64)]
    lo = np.asarray([a.min(axis=0) for a in groups])-PROJ_PAD
    hi = np.asarray([a.max(axis=0) for a in groups])+PROJ_PAD
    tr,tc = np.triu_indices(len(groups))
    lower = bbox_lower(lo,hi,tr,tc,scale2)
    size = np.asarray([len(a) for a in groups],dtype=np.int64)
    weights = size[tr]*size[tc]
    diag = tr==tc
    weights[diag] = size[tr[diag]]*(size[tr[diag]]+1)//2
    return dict(tr=tr.astype(np.int32),tc=tc.astype(np.int32),weights=weights,combined=lower),time.perf_counter()-start

def geometry_scores(q,scores,scale2):
    perm = np.argsort(scores,kind='stable').astype(np.int64)
    b,seconds = projected_bounds(q,perm,scale2)
    rows = []
    for cell in CELLS:
        reject = b['combined']>cell['T']+SQ_PAD
        assert not np.any(reject & (b['tr']==b['tc']))
        rows.append(dict(cell=cell['name'],pruned_tiles=int(reject.sum()),tiles=len(reject),
                         tile_fraction=float(reject.mean()),
                         pair_fraction=float(b['weights'][reject].sum()/b['weights'].sum())))
    return dict(mean_pair_pruning=float(np.mean([r['pair_fraction'] for r in rows])),
                cells=rows,bound_seconds=seconds,permutation_sha256=digest(perm)),perm,b

def reference_occupancy_check(perm,b,cell,subset=None):
    ref = np.load(SWEEP/'artifacts'/f"reference_{cell['name']}.npy",mmap_mode='r')
    if subset is not None:
        inverse = np.full(60000,-1,np.int64)
        inverse[subset] = np.arange(len(subset))
        parts = []
        for start in range(0,len(ref),1<<20):
            ids = np.asarray(ref[start:start+(1<<20)],np.uint64)
            ri,ci = inverse[(ids//60000).astype(np.int64)],inverse[(ids%60000).astype(np.int64)]
            mask = (ri>=0)&(ci>=0)
            parts.append((ri[mask]*len(subset)+ci[mask]).astype(np.uint64))
        ref = np.concatenate(parts)
    hit = occupied(ref,perm,b['tr'],b['tc'])
    reject = b['combined']>cell['T']+SQ_PAD
    assert not np.any(hit & reject),(cell['name'],int(np.sum(hit & reject)))
    return dict(reference_entries=len(ref),occupied_tiles=int(hit.sum()),
                pruned_tiles=int(reject.sum()),pass_=True)
