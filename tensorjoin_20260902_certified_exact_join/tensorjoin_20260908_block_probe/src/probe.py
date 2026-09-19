"""CPU-only full-reference block census and conventional geometric bounds."""
import hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SWEEP = ROOT / 'precision_routing_20260908_threshold_sweep'
PAD = 2.0**-20
SIZES = [32, 64, 128]
LAYOUTS = ['original', 'random', 'pivot_sort', 'kd_variance']
INPUT_HASH = '95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, a):
    with open(p, 'x') as f: json.dump(a, f, indent=2)

def emit(**kw): print(json.dumps(kw), flush=True)

def kd_order(x):
    def rec(ids):
        if len(ids) <= 32: return ids
        v = x[ids]
        axis = int(np.argmax(np.var(v, axis=0)))
        order = np.argsort(v[:, axis], kind='stable')
        ids = ids[order]
        del v
        # Split at a multiple of 32, so all sizes use one nested ordering.
        cut = ((len(ids) + 31) // 32 // 2) * 32
        assert 0 < cut < len(ids)
        return np.concatenate((rec(ids[:cut]), rec(ids[cut:])))
    return rec(np.arange(len(x), dtype=np.int32))

def pivot_data(x):
    t = time.perf_counter()
    first = int(np.random.Generator(np.random.PCG64(2026090815)).integers(len(x)))
    selected, distances = [], np.empty((len(x), 16), dtype=np.float64)
    nearest = np.full(len(x), np.inf)
    pid = first
    for k in range(16):
        selected.append(pid)
        for off in range(0, len(x), 4096):
            z = x[off:off+4096] - x[pid]
            d2 = np.sum(z*z, axis=1)
            distances[off:off+4096, k] = np.sqrt(d2)
            nearest[off:off+4096] = np.minimum(nearest[off:off+4096], d2)
        nearest[selected] = -np.inf
        pid = int(np.argmax(nearest))
    return distances, selected, time.perf_counter()-t

def metadata(x, pd, order, size):
    t = time.perf_counter()
    starts = np.arange(0, len(x), size)
    y = x[order]
    lo = np.minimum.reduceat(y, starts, axis=0)
    hi = np.maximum.reduceat(y, starts, axis=0)
    ns = np.minimum(size, len(x)-starts)
    centers = np.add.reduceat(y, starts, axis=0) / ns[:, None]
    d = y - np.repeat(centers, ns, axis=0)
    radii = np.sqrt(np.maximum.reduceat(np.sum(d*d, axis=1), starts))
    pp = pd[order]
    pl = np.minimum.reduceat(pp, starts, axis=0)
    pu = np.maximum.reduceat(pp, starts, axis=0)
    return (lo, hi, centers, radii, pl, pu, ns), time.perf_counter()-t

def bound_arrays(meta):
    lo, hi, centers, radii, pl, pu, ns = meta
    rows, cols = np.triu_indices(len(ns))
    result, timings = {}, {}
    for family in ['box', 'sphere', 'pivot']:
        t = time.perf_counter()
        lower, upper = np.empty(len(rows)), np.empty(len(rows))
        for off in range(0, len(rows), 2048):
            a, b = rows[off:off+2048], cols[off:off+2048]
            if family == 'box':
                gap = np.maximum(np.maximum(lo[a]-hi[b], lo[b]-hi[a]), 0)
                far = np.maximum(np.abs(hi[a]-lo[b]), np.abs(hi[b]-lo[a]))
                ll, uu = np.sum(gap*gap, axis=1), np.sum(far*far, axis=1)
            elif family == 'sphere':
                diff = centers[a]-centers[b]
                distance = np.sqrt(np.sum(diff*diff, axis=1))
                rad = radii[a]+radii[b]
                ll, uu = np.maximum(0, distance-rad)**2, (distance+rad)**2
            else:
                gap = np.maximum(np.maximum(pl[a]-pu[b], pl[b]-pu[a]), 0)
                ll = np.max(gap, axis=1)**2
                uu = np.min(pu[a]+pu[b], axis=1)**2
            lower[off:off+2048] = np.maximum(0, ll-PAD)
            upper[off:off+2048] = uu+PAD
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        assert (lower <= upper).all()
        timings[family] = time.perf_counter()-t
        result[family] = (lower, upper)
    t = time.perf_counter()
    result['combined'] = (np.maximum.reduce([v[0] for v in result.values()]),
                          np.minimum.reduce([v[1] for v in result.values()]))
    timings['combine_only'] = time.perf_counter()-t
    timings['combined'] = sum(timings.values())
    caps = ns[rows]*ns[cols]
    diag = rows == cols
    caps[diag] = ns[rows[diag]]*(ns[rows[diag]]+1)//2
    assert int(caps.sum()) == int(ns.sum())*(int(ns.sum())+1)//2
    return rows, cols, caps, result, timings

def count_references(order, cells, manifest):
    n, size = len(order), 32
    groups = (n+size-1)//size
    inverse = np.empty(n, dtype=np.int32)
    inverse[order] = np.arange(n, dtype=np.int32)//size
    result = {}
    t = time.perf_counter()
    for cell in cells:
        name = cell['name']
        ref = np.load(SWEEP/'artifacts'/f'reference_{name}.npy', mmap_mode='r')
        hist = np.zeros(groups*groups, dtype=np.int64)
        for off in range(0, len(ref), 2_000_000):
            ids = np.asarray(ref[off:off+2_000_000])
            r, c = ids//n, ids % n
            keep = r <= c
            a, b = inverse[r[keep]], inverse[c[keep]]
            key = np.minimum(a,b).astype(np.int64)*groups+np.maximum(a,b)
            hist += np.bincount(key, minlength=len(hist))
        assert int(hist.sum()) == manifest['cells'][name]['upper_count']
        result[name] = hist.reshape(groups, groups)
    return result, time.perf_counter()-t

def aggregate_counts(small, size):
    if size == 32: return small
    factor = size//32
    rows, cols = np.nonzero(small)
    groups = (len(small)+factor-1)//factor
    out = np.zeros(groups*groups, dtype=np.int64)
    np.add.at(out, (rows//factor)*groups+cols//factor, small[rows, cols])
    assert out.sum() == small.sum()
    return out.reshape(groups, groups)

def controls():
    rng = np.random.Generator(np.random.PCG64(2026090816))
    x = np.zeros((137, 512), dtype=np.float64)
    x[:, 0] = np.repeat([-.75, -.25, .25, .75, .95], [32,32,32,32,9])
    x[:, 1:5] = rng.uniform(-.001, .001, (137,4))
    x[1] = x[0]
    pd = np.sqrt(np.sum((x[:, None, :]-x[None, :16, :])**2, axis=2))
    meta, _ = metadata(x, pd, np.arange(len(x)), 32)
    r, c, caps, bounds, _ = bound_arrays(meta)
    distances = np.sum((x[:, None, :]-x[None, :, :])**2, axis=2)
    tests = []
    for threshold in [0., .01, float(distances[0,32]), 4.]:
        counts = []
        for a, b in zip(r,c):
            z = distances[a*32:min((a+1)*32,len(x)),b*32:min((b+1)*32,len(x))] <= threshold
            counts.append(int(np.triu(z).sum()) if a==b else int(z.sum()))
        counts = np.array(counts)
        for family, (lower, upper) in bounds.items():
            reject, accept = lower > threshold, upper <= threshold
            assert not np.any(reject & (counts != 0))
            assert not np.any(accept & (counts != caps))
            tests.append(dict(threshold=threshold, family=family, rejects=int(reject.sum()), accepts=int(accept.sum())))
    assert any(z['rejects'] and z['accepts'] for z in tests if z['threshold']==.01)
    return tests

def main():
    start = time.time()
    assert not (HERE/'results/probe.json').exists()
    cell_path = SWEEP/'artifacts/thresholds.json'
    cells = json.loads(cell_path.read_text())['cells']
    ref_manifest_path = SWEEP/'artifacts/reference_manifest.json'
    manifest = json.loads(ref_manifest_path.read_text())
    input_path = ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    hashes = {str(input_path.relative_to(ROOT)):sha(input_path),
              str(cell_path.relative_to(ROOT)):sha(cell_path),
              str(ref_manifest_path.relative_to(ROOT)):sha(ref_manifest_path),
              str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),
              str((HERE/'PROTOCOL.md').relative_to(ROOT)):sha(HERE/'PROTOCOL.md')}
    assert hashes[str(input_path.relative_to(ROOT))] == INPUT_HASH
    for cell in cells:
        p = SWEEP/'artifacts'/f"reference_{cell['name']}.npy"
        hashes[str(p.relative_to(ROOT))] = sha(p)
        assert hashes[str(p.relative_to(ROOT))] == manifest['cells'][cell['name']]['file_sha256']
    write(HERE/'artifacts/input_freeze.json', hashes)
    checks = controls()
    write(HERE/'results/controls.json', dict(pass_=True, tests=checks))
    x32 = np.load(input_path, allow_pickle=False)
    assert x32.shape==(60000,512) and x32.dtype==np.float32 and np.isfinite(x32).all() and np.abs(x32).max()<=1
    x = x32.astype(np.float64); del x32
    pd, pivots, pivot_seconds = pivot_data(x)
    write(HERE/'artifacts/pivots.json', dict(ids=pivots, seconds=pivot_seconds))
    rows_out, layout_info = [], {}
    for layout in LAYOUTS:
        t = time.perf_counter()
        if layout == 'original': order = np.arange(len(x), dtype=np.int32)
        elif layout == 'random': order = np.random.Generator(np.random.PCG64(2026090814)).permutation(len(x)).astype(np.int32)
        elif layout == 'pivot_sort': order = np.argsort(pd[:,0], kind='stable').astype(np.int32)
        else: order = kd_order(x)
        ordering_seconds = time.perf_counter()-t
        assert np.array_equal(np.sort(order), np.arange(len(x)))
        np.save(HERE/'artifacts'/f'order_{layout}.npy', order, allow_pickle=False)
        counts32, census_seconds = count_references(order, cells, manifest)
        layout_info[layout] = dict(ordering_seconds=ordering_seconds, census_seconds=census_seconds)
        emit(layout=layout, order_seconds=ordering_seconds, census_seconds=census_seconds)
        for size in SIZES:
            meta, metadata_seconds = metadata(x,pd,order,size)
            rr, cc, caps, bounds, times = bound_arrays(meta)
            records = []
            np.savez_compressed(HERE/'artifacts'/f'bounds_{layout}_{size}.npz',
                **{f'{k}_{side}': v[i] for k,v in bounds.items() for i,side in enumerate(['lower','upper'])})
            for cell in cells:
                counts = aggregate_counts(counts32[cell['name']], size)[rr,cc]
                assert (counts <= caps).all()
                empty, full = counts==0, counts==caps
                np.savez_compressed(HERE/'artifacts'/f'counts_{layout}_{size}_{cell["name"]}.npz', accepted=counts)
                for family,(lower,upper) in bounds.items():
                    reject, accept = lower > cell['T'], upper <= cell['T']
                    assert not (reject & accept).any()
                    assert not (reject & ~empty).any(), (layout,size,cell,family,'false reject')
                    assert not (accept & ~full).any(), (layout,size,cell,family,'false accept')
                    resolved = reject | accept
                    rec = dict(layout=layout, size=size, cell=cell['name'], threshold=cell['T'], family=family,
                        blocks=len(rr), pairs=int(caps.sum()), positive_pairs=int(counts.sum()),
                        oracle_empty_blocks=int(empty.sum()), oracle_full_blocks=int(full.sum()),
                        oracle_resolved_fraction=float(np.mean(empty|full)),
                        rejected_blocks=int(reject.sum()), accepted_blocks=int(accept.sum()),
                        resolved_fraction=float(resolved.mean()), resolved_pair_fraction=float(caps[resolved].sum()/caps.sum()),
                        rejected_pairs=int(caps[reject].sum()), accepted_pairs=int(caps[accept].sum()),
                        empty_block_recall=float(reject.sum()/max(1,empty.sum())),
                        ordering_seconds=ordering_seconds, pivot_preprocessing_seconds=pivot_seconds,
                        metadata_seconds=metadata_seconds, bound_seconds=times[family],
                        false_reject_blocks=0, false_accept_blocks=0)
                    records.append(rec)
            rows_out.extend(records)
            best = max(z['resolved_fraction'] for z in records if z['family']=='combined')
            emit(layout=layout, size=size, best_combined_resolved=best, bounds_seconds=times, metadata_seconds=metadata_seconds)
            write(HERE/'results'/f'{layout}_{size}.json', records)
        del counts32
    native = [z for z in rows_out if z['size']==64 and z['family']=='combined']
    best = max(native, key=lambda z:z['resolved_fraction'])
    result = dict(pass_=True, started=start, ended=time.time(), seconds=time.time()-start,
        host=platform.node(), python=sys.version, numpy=np.__version__, cpu=platform.processor(),
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        input_shape=list(x.shape), pad=PAD, cells=cells, pivots=pivots, pivot_seconds=pivot_seconds,
        layouts=layout_info, records=rows_out, native_best=best,
        gpu_stage_admitted=best['resolved_fraction']>=.20, configuration_count=len(rows_out),
        total_false_reject_blocks=0, total_false_accept_blocks=0)
    for rel, h in hashes.items(): assert sha(ROOT/rel)==h,rel
    write(HERE/'results/probe.json', result)
    emit(finished=True, configurations=len(rows_out), seconds=result['seconds'], native_best=best, gpu_stage_admitted=result['gpu_stage_admitted'])

if __name__=='__main__': main()
