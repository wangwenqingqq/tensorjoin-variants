"""CPU layouts and conservative squared block-distance lower bounds."""
import hashlib
import time
import numpy as np

BLOCK = 64
LAYOUTS = ['original', 'pca_sort', 'balanced_axis', 'balanced_2means']
SQ_PAD = 2.0 ** -20
DIST_PAD = 2.0 ** -30
PROJ_PAD = 2.0 ** -24

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def projection(x):
    z = x.astype(np.float64)
    mu = z.mean(axis=0)
    z -= mu
    cov = z.T @ z
    val, vec = np.linalg.eigh(cov)
    p = np.ascontiguousarray(vec[:, -min(32, x.shape[1]):][:, ::-1])
    for j in range(p.shape[1]):
        if p[np.argmax(np.abs(p[:, j])), j] < 0:
            p[:, j] *= -1
    gram = p.T @ p
    # ||P||_2^2 <= 1 + ||P^T P-I||_F. Generous verified slack also
    # covers CPU dot/reduction rounding for D <= 1024, |P_ij| <= 1.
    residual = float(np.sqrt(np.sum((gram - np.eye(len(gram))) ** 2)))
    scale2 = 1.0 + residual + SQ_PAD
    return p, scale2

def layout(x, method, p=None):
    n = len(x)
    if method == 'original':
        return np.arange(n, dtype=np.int64)
    z = x.astype(np.float64)
    if method == 'pca_sort':
        assert p is not None
        return np.argsort(z @ p[:, 0], kind='stable').astype(np.int64)
    assert method in ['balanced_axis', 'balanced_2means']
    def split(ids, leaves):
        if leaves == 1:
            return ids
        a = z[ids]
        left_leaves = leaves // 2
        nleft = left_leaves * BLOCK
        if method == 'balanced_axis':
            v = a.var(axis=0)
            score = a[:, int(np.argmax(v))]
        else:
            center = a.mean(axis=0)
            c0 = a[np.argmax(np.sum((a-center) ** 2, axis=1))]
            c1 = a[np.argmax(np.sum((a-c0) ** 2, axis=1))]
            for _ in range(4):
                score = a @ (c1-c0)
                order = np.argsort(score, kind='stable')
                c0 = a[order[:nleft]].mean(axis=0)
                c1 = a[order[nleft:]].mean(axis=0)
            score = a @ (c1-c0)
        order = np.argsort(score, kind='stable')
        return np.concatenate([split(ids[order[:nleft]], left_leaves),
                               split(ids[order[nleft:]], leaves-left_leaves)])
    return split(np.arange(n, dtype=np.int64), (n+BLOCK-1)//BLOCK)

def sq_dist(a, b):
    """Direct FP64 differences/squares, no cancellation-prone Gram formula."""
    v = a-b
    return np.sum(v*v, axis=-1)

def groups(a):
    return [a[i:i+BLOCK] for i in range(0, len(a), BLOCK)]

def bbox_lower(lo, hi, tr, tc, scale2=1.0):
    ans = np.empty(len(tr), np.float64)
    for off in range(0, len(tr), 4096):
        i, j = tr[off:off+4096], tc[off:off+4096]
        gap = np.maximum(0., np.maximum(lo[i]-hi[j], lo[j]-hi[i]))
        ans[off:off+len(i)] = np.maximum(0., np.sum(gap*gap, axis=1)-SQ_PAD)/scale2
    return ans

def bounds(x, perm, p, scale2):
    assert x.dtype == np.float32 and x.shape[1] <= 1024
    assert np.isfinite(x).all() and np.max(np.abs(x)) <= 1
    z = x[perm].astype(np.float64)
    g = groups(z)
    nb = len(g)
    tr, tc = np.triu_indices(nb)
    center = np.asarray([a.mean(axis=0) for a in g], np.float32).astype(np.float64)
    radius = np.asarray([np.nextafter(np.sqrt(np.max(sq_dist(a, c))+SQ_PAD), np.inf)
                         for a, c in zip(g, center)])
    ball = np.empty(len(tr), np.float64)
    for off in range(0, len(tr), 4096):
        i, j = tr[off:off+4096], tc[off:off+4096]
        dist = np.nextafter(np.sqrt(np.maximum(0., sq_dist(center[i], center[j])-SQ_PAD)), -np.inf)
        lower = np.maximum(0., dist-radius[i]-radius[j]-DIST_PAD)
        ball[off:off+len(i)] = np.maximum(0., lower*lower-SQ_PAD)
    lo = np.asarray([a.min(axis=0) for a in g])
    hi = np.asarray([a.max(axis=0) for a in g])
    axis = bbox_lower(lo, hi, tr, tc)
    # Pivots depend only on original row order, not the tested permutation.
    orig = x.astype(np.float64)
    pivot_ids = []
    closest = np.full(len(x), np.inf)
    dist_cols = []
    pivot = 0
    for _ in range(16):
        pivot_ids.append(int(pivot))
        ds = sq_dist(orig, orig[pivot])
        dist_cols.append(ds)
        closest = np.minimum(closest, ds)
        pivot = int(np.argmax(closest))
    ds = np.asarray(dist_cols).T[perm]
    dlo = np.nextafter(np.sqrt(np.maximum(0., ds-SQ_PAD)), -np.inf)
    dhi = np.nextafter(np.sqrt(ds+SQ_PAD), np.inf)
    pl = np.asarray([a.min(axis=0) for a in groups(dlo)])
    pu = np.asarray([a.max(axis=0) for a in groups(dhi)])
    gap = np.maximum(0., np.maximum(pl[tr]-pu[tc], pl[tc]-pu[tr])-DIST_PAD)
    pivot_bound = np.maximum(0., np.max(gap, axis=1)**2-SQ_PAD)
    projected = z @ p
    qlo = np.asarray([a.min(axis=0) for a in groups(projected)])-PROJ_PAD
    qhi = np.asarray([a.max(axis=0) for a in groups(projected)])+PROJ_PAD
    proj_bound = bbox_lower(qlo, qhi, tr, tc, scale2)
    all_bounds = dict(ball=ball, aabb=axis, pivots=pivot_bound, pca32=proj_bound)
    all_bounds['combined'] = np.maximum.reduce(list(all_bounds.values()))
    size = np.asarray([len(a) for a in g], dtype=np.int64)
    pair_weight = size[tr]*size[tc]
    diag = tr == tc
    pair_weight[diag] = size[tr[diag]]*(size[tr[diag]]+1)//2
    return dict(tr=tr.astype(np.int32), tc=tc.astype(np.int32), weights=pair_weight,
                **all_bounds), dict(pivot_ids=pivot_ids, projection_scale2=scale2,
                                   radii_quantiles=np.quantile(radius, [0,.25,.5,.75,1]).tolist())

def build(x, method):
    start = time.perf_counter()
    p, scale2 = projection(x)
    projected_at = time.perf_counter()
    perm = layout(x, method, p)
    grouped_at = time.perf_counter()
    b, diag = bounds(x, perm, p, scale2)
    bounded_at = time.perf_counter()
    reordered = np.ascontiguousarray(x[perm])
    end = time.perf_counter()
    return perm, reordered, b, dict(seconds=end-start, projection_seconds=projected_at-start,
        grouping_seconds=grouped_at-projected_at, bound_seconds=bounded_at-grouped_at,
        reorder_seconds=end-bounded_at, permutation_sha256=digest(perm), **diag)

def retained_mask(b, threshold):
    # The positive absolute squared-distance slack also encloses the frozen
    # direct FP64 terminal's roundoff in the admitted input range.
    return b['combined'] <= float(threshold)+SQ_PAD
