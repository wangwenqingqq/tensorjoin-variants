"""Read-only reproduction of the first G0 CPU/GPU reference mismatch."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from run import Engine, cpu_oracle, environment, hashes
import experiment as keeper
from workloads import records, load
import numpy as np
import torch


def main():
    rec = next(r for r in records() if r['config_id'] == 'clustered_102_n4096_k8')
    x = load(rec); t = rec['threshold_d2']
    idx = np.sort(np.random.Generator(np.random.PCG64(31337)).choice(len(x),32,replace=False))
    x = x[idx].copy(); cpu = cpu_oracle(x,t)
    engine = Engine(); old = keeper.Engine()
    results = []; sets = {}
    for name, impl in [('current',engine),('keeper',old)]:
        for method in ['fp64','int8','e3m4','fp16']:
            for repeat in range(3):
                ids, row = impl.run(x,t,method)
                results.append({'implementation':name,'method':method,'repeat':repeat,
                    'count':len(ids),'output_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                    'cpu_equal':bool(np.array_equal(ids,cpu)),
                    'gpu_extra_ids':np.setdiff1d(ids,cpu).tolist(),'gpu_missing_ids':np.setdiff1d(cpu,ids).tolist()})
                sets[name,method] = ids
    gpu = sets['current','fp64']; differing = np.union1d(np.setdiff1d(gpu,cpu),np.setdiff1d(cpu,gpu))
    rows=[]
    for pair in differing:
        i,j=divmod(int(pair),len(x))
        delta=x[i].astype(np.float64)-x[j].astype(np.float64); sq=delta*delta
        exact = sum((Fraction(float(a))-Fraction(float(b)))**2 for a,b in zip(x[i],x[j]))
        sequential = np.float64(0.)
        for value in sq: sequential=np.float64(sequential+value)
        vals={'numpy_sum512':float(np.sum(sq)), 'numpy_two_sum256':float(np.sum(sq[:256])+np.sum(sq[256:])),
              'sequential512':float(sequential),'math_fsum_rounded_squares':math.fsum(map(float,sq)),
              'exact_rational_rounded_once':float(exact)}
        rows.append({'pair_id':int(pair),'subset_rows':[i,j],'parent_rows':[int(idx[i]),int(idx[j])],
            'threshold_hex':t.hex(),'values':{k:{'hex':v.hex(),'decimal':v,'inside':v<=t,'threshold_ulps':(v-t)/math.ulp(t)} for k,v in vals.items()},
            'exact_rational_inside':exact<=Fraction(t),'exact_minus_threshold_numerator':str((exact-Fraction(t)).numerator),
            'exact_minus_threshold_denominator':str((exact-Fraction(t)).denominator)})
    result={'config':rec,'environment':environment(),'source_sha256':hashes(),'subset_indices':idx.tolist(),
            'subset_input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'cpu_ids':cpu.tolist(),'gpu_fp64_ids':gpu.tolist(),
            'results':results,'differing_pairs':rows,
            'all_methods_match_gpu_fp64':all(np.array_equal(a,gpu) for a in sets.values()),
            'scope':'Numerical diagnosis only; no performance estimator or gate promotion.'}
    p=ROOT/'results/g0_failure_diagnosis_v1.json'
    if p.exists():raise FileExistsError(p.name)
    p.write_text(json.dumps(result,indent=2)+'\n')
    np.save(ROOT/'results/g0_failure_subset_v1.npy',x,allow_pickle=False)
    engine.capture(ROOT/'results/g0_failure_diagnosis_v1.compiled.json')
    for name,k in engine.kernels.items():
        if name=='fp64':
            (ROOT/'artifacts'/'failure_terminal.ptx').write_text(k.asm['ptx'])
    print(json.dumps({'all_methods_match_gpu_fp64':result['all_methods_match_gpu_fp64'],
                      'comparisons':len(results),'differing_pairs':rows},indent=2))


if __name__=='__main__':main()
