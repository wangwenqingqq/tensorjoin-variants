"""CPU-only replay of the stopped G0 mismatch; does not waive or rerun G0."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def terminal_order_distances(x):
    """Emulate the measured SM120 terminal's explicit FP64 PTX reduction graph.

    Two 256-element chunks, adjacent pairs per thread, XOR 16/8/4/2/1
    across 32 lanes, then XOR 2/1 across four warps, then add the chunks.
    Re-derive this graph if the compiled terminal, layout or toolkit changes.
    """
    delta=x.astype(np.float64)[:,None,:]-x.astype(np.float64)[None,:,:]
    sq=delta*delta
    local=sq.reshape(len(x),len(x),2,4,32,2)
    lanes=local[...,0]+local[...,1]
    for shift in (16,8,4,2,1):
        lanes=lanes+lanes[...,np.arange(32)^shift]
    warps=lanes[...,0]
    for shift in (2,1):
        warps=warps+warps[...,np.arange(4)^shift]
    chunks=warps[...,0]
    return (np.float64(0)+chunks[...,0])+chunks[...,1]


def main():
    d=json.loads((ROOT/'results/g0_failure_diagnosis_v1.json').read_text())
    x=np.load(ROOT/'results/g0_failure_subset_v1.npy',allow_pickle=False)
    assert hashlib.sha256(x.tobytes()).hexdigest()==d['subset_input_sha256']
    t=float.fromhex(d['config']['threshold_hex'])
    distance=terminal_order_distances(x)
    ids=np.flatnonzero(distance<=t).astype(np.uint64)
    gpu=np.array(d['gpu_fp64_ids'],np.uint64); cpu=np.array(d['cpu_ids'],np.uint64)
    assert np.array_equal(ids,gpu)
    assert not np.array_equal(cpu,gpu) and len(np.setxor1d(cpu,gpu))==2
    assert d['all_methods_match_gpu_fp64'] and len(d['results'])==24
    assert all(r['output_sha256']==hashlib.sha256(gpu.tobytes()).hexdigest() for r in d['results'])
    rows=[]
    for r in d['differing_pairs']:
        i,j=r['subset_rows'];v=float(distance[i,j])
        rows.append({'parent_rows':r['parent_rows'],'threshold_hex':t.hex(),
                     'emulated_gpu_distance_hex':v.hex(),'inside':v<=t})
    print(json.dumps({'pass':True,'scope':'CPU emulation matches the recorded GPU IDs, not a fresh GPU validation',
                      'cpu_count':len(cpu),'gpu_count':len(gpu),'differences':rows},indent=2))


if __name__=='__main__': main()
