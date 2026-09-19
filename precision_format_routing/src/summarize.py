"""Rebuild the process-paired screen summary from immutable raw JSON records."""
import hashlib
import json
from pathlib import Path
import statistics
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[ROOT/'results'/f'v1b_bench_{i}.json' for i in range(4)]
    data=[json.loads(f.read_text()) for f in files]
    identities=[d['source_sha256'] for d in data]
    assert all(d['ptxas_version']=='13.1' for d in data)
    assert all(h==identities[0] for h in identities)
    records=[d['records'] for d in data]
    cases=list(dict.fromkeys(r['case'] for r in records[0]))
    output={'estimator':'paired process median ratios INT8/candidate; 4 processes x 5 measured repetitions',
            'timing_scope':'pageable host FP32 input to canonical directed host uint64 IDs; warm JIT, cold charged selector',
            'raw_files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'cases':{}}
    for case in cases:
        rows=[]
        for recs in records:
            rows.append([r for r in recs if r['case']==case and r.get('exact_ids') and not r.get('warmup')])
        formats=list(dict.fromkeys(r['format'] for r in rows[0]));per_case={}
        base=[np.median([r['e2e_ms'] for r in rr if r['format']=='int8']) for rr in rows]
        hashes={r['output_sha256'] for rr in rows for r in rr};assert len(hashes)==1
        for fmt in formats:
            grouped=[[r for r in rr if r['format']==fmt] for rr in rows]
            assert all(len(g)==5 for g in grouped),(case,fmt)
            med=[float(np.median([r['e2e_ms'] for r in g])) for g in grouped]
            raw=[r for g in grouped for r in g];times=[r['e2e_ms'] for r in raw]
            ratios=[float(a/b) for a,b in zip(base,med)];logs=list(map(math.log,ratios))
            center=statistics.mean(logs);half=3.182446305*statistics.stdev(logs)/2
            rec={'n':raw[0]['n'],'threshold_d2':raw[0]['threshold_d2'],'p10_p50_p90_ms':np.percentile(times,[10,50,90]).tolist(),
                 'process_medians_ms':med,'paired_ratios_int8_over_candidate':ratios,
                 'geomean_paired_ratio':math.exp(center),'paired_log_t95_ci':[math.exp(center-half),math.exp(center+half)],
                 'process_wins':sum(r>1 for r in ratios),'screen_gate_all_processes_gt_1_10':all(r>1.1 for r in ratios),
                 'fp32_pairs':sorted({r['fp32_pairs'] for r in raw}),'fp64_pairs':sorted({r['fp64_pairs'] for r in raw}),
                 'output_count':raw[0]['output_count'],'output_sha256':raw[0]['output_sha256']}
            for key in ['transfer_allocate_prepare_ms','stage1_ms','host_counts_fp32_ms','fp64_ms']:
                rec[key+'_median']=float(np.median([r[key] for r in raw]))
            if fmt=='route':
                rec['selected']={m:sum(r['selected']==m for r in raw) for m in sorted({r['selected'] for r in raw})}
                rec['routing_ms_median']=float(np.median([r['routing_ms'] for r in raw]))
            per_case[fmt]=rec
        output['cases'][case]=per_case
    path=ROOT/'results'/'SUMMARY.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({name:{f:{'e2e_ms':r['p10_p50_p90_ms'][1],'fp32':r['fp32_pairs'],'fp64':r['fp64_pairs'],'ratio':r['geomean_paired_ratio'],'gate':r['screen_gate_all_processes_gt_1_10']} for f,r in methods.items()} for name,methods in output['cases'].items() if name in ['cifar4096','synthetic_clustered','synthetic_outlier']},indent=2))

if __name__=='__main__':main()
