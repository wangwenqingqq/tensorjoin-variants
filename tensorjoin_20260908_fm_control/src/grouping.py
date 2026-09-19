from layout_common import *

RULES=['first','balanced']


def permutation(coordinates,rule,first_scores=None):
    assert rule in RULES
    z=np.asarray(coordinates,dtype=np.float64)
    assert z.ndim==2 and z.shape[1]==32 and np.isfinite(z).all()
    if rule=='first':
        score=z[:,0] if first_scores is None else first_scores
        return np.argsort(score,kind='stable').astype(np.int64)
    def split(ids,leaves):
        if leaves==1:return ids
        rows=z[ids]
        axis=int(np.argmax(rows.var(axis=0)))
        left_leaves=leaves//2
        left_rows=left_leaves*64
        order=np.argsort(rows[:,axis],kind='stable')
        return np.concatenate([split(ids[order[:left_rows]],left_leaves),
            split(ids[order[left_rows:]],leaves-left_leaves)])
    return split(np.arange(len(z),dtype=np.int64),(len(z)+63)//64)


def layout_geometry(q,coordinates,scale2,rule,first_scores=None):
    started=time.perf_counter()
    perm=permutation(coordinates,rule,first_scores)
    grouping_seconds=time.perf_counter()-started
    assert np.array_equal(np.sort(perm),np.arange(len(q)))
    b,bound_seconds=projected_bounds(q,perm,scale2)
    cells=[]
    for cell in CELLS:
        reject=b['combined']>cell['T']+SQ_PAD
        assert not np.any(reject & (b['tr']==b['tc']))
        cells.append(dict(cell=cell['name'],pruned_tiles=int(reject.sum()),tiles=len(reject),
            tile_fraction=float(reject.mean()),
            pair_fraction=float(b['weights'][reject].sum()/b['weights'].sum())))
    return dict(rule=rule,mean_pair_pruning=float(np.mean([c['pair_fraction'] for c in cells])),
        cells=cells,grouping_seconds=grouping_seconds,bound_seconds=bound_seconds,
        permutation_sha256=digest(perm)),perm,b


def distribution_diagnostics(coordinates,source,target_sorted,directions,target_std):
    z=np.asarray(coordinates,dtype=np.float64)
    projected=np.sort(z@directions,axis=0)
    return dict(sliced_w2_squared_normalized=float(np.mean((projected-target_sorted)**2)/(target_std**2)),
        mean_point_squared_displacement=float(np.mean(np.sum((z-source)**2,axis=1))),
        mean_coordinate_variance=float(np.mean(z.var(axis=0))),
        norm_of_mean=float(np.linalg.norm(z.mean(axis=0))),
        finite=bool(np.isfinite(z).all()))
