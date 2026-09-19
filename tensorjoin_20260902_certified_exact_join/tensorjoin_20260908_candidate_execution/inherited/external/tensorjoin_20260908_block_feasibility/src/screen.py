import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from geometry import BLOCK, LAYOUTS, SQ_PAD, build, digest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join')
SWEEP = ROOT/'precision_routing_20260908_threshold_sweep'

def sha(p):
    import hashlib
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def save(p, r):
    with Path(p).open('x') as f:
        json.dump(r, f, indent=2)

def occupied(ids, perm, tr, tc):
    n = len(perm)
    nb = (n+BLOCK-1)//BLOCK
    inv = np.empty(n, np.int64)
    inv[perm] = np.arange(n)
    hit = np.zeros(nb*nb, bool)
    for start in range(0, len(ids), 1<<20):
        a = np.asarray(ids[start:start+(1<<20)], dtype=np.uint64)
        i, j = inv[(a//n).astype(np.int64)]//BLOCK, inv[(a%n).astype(np.int64)]//BLOCK
        lo, hi = np.minimum(i,j), np.maximum(i,j)
        hit[lo*nb+hi] = True
    return hit[tr.astype(np.int64)*nb+tc]

def one_dataset(name, x, cells, references):
    results = []
    for method in LAYOUTS:
        print(json.dumps(dict(building=name, layout=method, time=time.time())), flush=True)
        perm, _, b, timing = build(x, method)
        assert np.array_equal(np.sort(perm), np.arange(len(x)))
        prefix = HERE/'artifacts'/f'{name}_{method}'
        np.save(str(prefix)+'_permutation.npy', perm, allow_pickle=False)
        np.savez_compressed(str(prefix)+'_bounds.npz', **b)
        ideal = {}
        rows = []
        for c in cells:
            ids = references(c)
            hit = occupied(ids, perm, b['tr'], b['tc'])
            ideal[c['name']] = hit
            for bound in ['ball', 'aabb', 'pivots', 'pca32', 'combined']:
                reject = b[bound] > c['T']+SQ_PAD
                assert not np.any(reject & hit), (name, method, c['name'], bound)
                assert not np.any(reject & (b['tr'] == b['tc']))
                rec = dict(dataset=name, layout=method, cell=c['name'], threshold=c['T'],
                    bound=bound, tiles=len(hit), pruned_tiles=int(reject.sum()),
                    pruned_tile_fraction=float(reject.mean()),
                    pruned_pairs=int(b['weights'][reject].sum()),
                    pruned_pair_fraction=float(b['weights'][reject].sum()/b['weights'].sum()),
                    ideal_empty_tiles=int((~hit).sum()),
                    ideal_empty_tile_fraction=float((~hit).mean()),
                    ideal_empty_pair_fraction=float(b['weights'][~hit].sum()/b['weights'].sum()),
                    reference_entries=len(ids), reference_occupancy_check=True)
                rows.append(rec)
                if bound == 'combined': print(json.dumps(rec), flush=True)
        np.savez_compressed(str(prefix)+'_ideal.npz', **ideal)
        rec = dict(dataset=name, layout=method, shape=list(x.shape), timing=timing, cells=rows,
                   input_raw_sha256=digest(x), bounds_file_sha256=sha(str(prefix)+'_bounds.npz'))
        save(HERE/'results'/f'screen_{name}_{method}.json', rec)
        results.append(rec)
    return results

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    freeze = {str(p.relative_to(HERE)): sha(p) for p in
              [HERE/'PROTOCOL.md', HERE/'NUMERICAL_SCOPE.md', *sorted((HERE/'src').glob('*.py'))]}
    save(HERE/'artifacts/screen_freeze.json', dict(created=time.time(), files=freeze))
    meta = json.loads((SWEEP/'artifacts/thresholds.json').read_text())
    data = ROOT/'data/g2b_cifar60000/vectors_f32.npy'
    assert sha(data) == meta['input_sha256']
    x = np.load(data, allow_pickle=False)
    records = one_dataset('cifar60000', x, meta['cells'],
                         lambda c: np.load(SWEEP/'artifacts'/f"reference_{c['name']}.npy", mmap_mode='r'))
    scores = {r['layout']: float(np.mean([c['pruned_pair_fraction'] for c in r['cells']
                if c['bound'] == 'combined'])) for r in records if r['layout'] != 'original'}
    selected = max(scores, key=scores.get) if max(scores.values()) > 0 else 'balanced_2means'
    selection = dict(created=time.time(), selected=selected, scores=scores,
                     rule='Highest six-cell mean certified pair pruning; no GPU timings or ideal occupancy used.')
    save(HERE/'artifacts/selection.json', selection)
    print(json.dumps(selection), flush=True)
    del x
    for name in ['sift128', 'fashion784']:
        base = ROOT/'data/g4b_public'/name
        x = np.load(base/'vectors_f32.npy')*np.float32(2**-8)
        cells = []
        for k in [1,16,64]:
            m = json.loads((base/f'n4096/k{k}/metadata.json').read_text())
            cells.append(dict(name=f'k{k}', T=m['threshold_d2']*2**-16))
        records += one_dataset(name, x, cells,
             lambda c: np.load(base/f"n4096/{c['name']}/oracle_upper_ids_u64.npy", mmap_mode='r'))
    rng = np.random.default_rng(2026090809)
    labels = np.repeat(np.arange(64),64)
    x = rng.uniform(-1e-4,1e-4,(4096,512)).astype(np.float32)
    x[np.arange(4096), labels] += np.float32(.5)
    order = rng.permutation(4096)
    x, labels = x[order], labels[order]
    # Guaranteed separated clusters: within distance <0.01; between >0.69.
    assert 2*np.sqrt(512)*1.01e-4 < .1
    assert np.sqrt(.5)-2*np.sqrt(512)*1.01e-4 > .1
    rr, cc = np.where(labels[:,None] == labels[None,:])
    ids = (rr*4096+cc).astype(np.uint64)
    records += one_dataset('positive_control', x, [dict(name='separated', T=.01)], lambda c: ids)
    save(HERE/'results/screen_all.json', dict(pass_=True, selected=selection,
        records=records, host=platform.node(), numpy_version=np.__version__, ended=time.time()))
    print(json.dumps(dict(screen_complete=True, layouts=len(records))), flush=True)

if __name__ == '__main__':
    main()
