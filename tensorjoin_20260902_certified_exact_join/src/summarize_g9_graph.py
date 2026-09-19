#!/usr/bin/env python3
"""Frozen two-process G9 summary, without favorable-order/result selection."""

import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LABELS=('g9_graph_p0_a0','g9_graph_p1_a0')
METHODS=('P16','TC16physical','TC16diagonal')


def main():
    all_labels=('g9_graph_check_a0','g9_graph_memcheck_a0','g9_graph_synccheck_a0')+LABELS
    records={}
    for label in all_labels:
        data=json.loads((ROOT/f'results/{label}.json').read_text())
        manifest=json.loads((ROOT/f'results/{label}_manifest.json').read_text())
        assert data['correctness_pass'] and manifest['admitted']
        assert data['gpu_uuid']=='GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603'
        if 'memcheck' in label or 'synccheck' in label:
            assert 'ERROR SUMMARY: 0 errors' in (ROOT/f'raw/{label}.log').read_text()
        records[label]=data
    keeper=records['g9_graph_check_a0']
    original=json.loads((ROOT/'results/g9_check_a3.json').read_text())
    assert set(keeper['compiled'])==set(original['compiled'])
    assert keeper['canonical_final_state_hashes']==original['canonical_final_state_hashes']
    for data in records.values():
        assert data['graph_mode']
        assert data['source_hashes']==keeper['source_hashes']
        assert set(data['compiled'])==set(keeper['compiled'])
        assert data['canonical_final_state_hashes']==keeper['canonical_final_state_hashes']
    cases=[]
    for origin in keeper['cases']:
        if origin['fixture']:
            continue
        name=origin['name']
        process_rows=[]
        for label in LABELS:
            data=records[label]
            c=next(c for c in data['cases'] if c['name']==name)
            rounds={i:{} for i in range(40)}
            for sample in c['samples']:
                assert sample['method'] not in rounds[sample['round']]
                rounds[sample['round']][sample['method']]=sample['ms']
            assert all(set(row)==set(METHODS) for row in rounds.values())
            values={m:np.array([r[m] for r in rounds.values()]) for m in METHODS}
            row=dict(label=label,reverse=data['reverse'],methods={},sustained_ms=c['sustained_ms'],
                     stage_diagnostic_ms=c['stage_diagnostic_ms'],ratios={})
            for m in METHODS:
                row['methods'][m]=dict(zip(('p10_ms','median_ms','p90_ms'),np.quantile(values[m],[.1,.5,.9]).tolist()))
                row['methods'][m]['position_medians_ms']={str(i):float(np.median([s['ms'] for s in c['samples'] if s['method']==m and s['position']==i])) for i in range(3)}
            for m in METHODS[1:]:
                ratio=values['P16']/values[m]
                logs=np.log(ratio)
                row['ratios'][m]=dict(paired_log_median=float(np.median(logs)),
                    primary_p16_over_candidate=float(np.exp(np.median(logs))),
                    marginal_median_ratio=float(np.median(values['P16'])/np.median(values[m])),
                    paired_arithmetic_mean=float(np.mean(ratio)),paired_geometric_mean=float(np.exp(np.mean(logs))),
                    round_wins=int(np.sum(ratio>1)),
                    sustained_ratio=c['sustained_ms']['P16']/c['sustained_ms'][m])
            process_rows.append(row)
        aggregate={}
        for m in METHODS[1:]:
            r=[p['ratios'][m] for p in process_rows]
            logs=np.array([v['paired_log_median'] for v in r])
            mean=float(logs.mean())
            half=12.706204736432095*float(logs.std(ddof=1)/math.sqrt(2))
            aggregate[m]=dict(primary_p16_over_candidate=math.exp(mean),
                diagnostic_t95_interval=[math.exp(mean-half),math.exp(mean+half)],
                process_wins=sum(v['primary_p16_over_candidate']>1 for v in r),
                qualifies=all(v['primary_p16_over_candidate']>=1.15 and v['sustained_ratio']>1 for v in r),
                interval_limit='Two processes; log-normal assumption untested; directional screen only.')
        cases.append(dict(name=name,synthetic=origin['synthetic'],methods=origin['methods'],
                          processes=process_rows,aggregate=aggregate))
    qualifiers=[dict(case=c['name'],method=m,ratio=v['primary_p16_over_candidate']) for c in cases if not c['synthetic'] for m,v in c['aggregate'].items() if v['qualifies']]
    result=dict(experiment_id=keeper['experiment_id'],status='partial_optimistic_graph_refinement_evidence',same_cubins_as_eager_a0=True,graph_mode=True,
        gpu_uuid=keeper['gpu_uuid'],cases=cases,actual_qualifiers=qualifiers,
        advance_to_fully_costed_protocol=bool(qualifiers),
        all_main_outputs_fully_resolved=True,canonical_final_state_hashes=keeper['canonical_final_state_hashes'],
        identical_cubin_set_across_correctness_sanitizers_and_timing=True,
        identical_source_hashes_across_admitted_processes=True,
        compiled_kernel_count=len(keeper['compiled']),
        denominator='CUDA-event complete captured refinement, 20 Graph replays per observation; capture excluded',
        included='TC dot, directed certificate and endpoint writes, actual FP32 repair, actual FP64 repair, aligned output updates',
        excluded='initial filter, encoding, layout/tile metadata, intermediate queue construction/count discovery, Graph capture, final ID export, ingest',
        claim_boundary='Not an online implementation, full join speedup, novelty result, or production admission.')
    out=ROOT/'results/g9_graph_summary_a0.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('ADVANCE',result['advance_to_fully_costed_protocol'])
    for c in cases:
        print(c['name'])
        for m,v in c['aggregate'].items():
            print(m,'ratio',v['primary_p16_over_candidate'],'processes',
                [p['ratios'][m]['primary_p16_over_candidate'] for p in c['processes']],
                'sustained',[p['ratios'][m]['sustained_ratio'] for p in c['processes']], 'qualifies',v['qualifies'])


if __name__=='__main__':
    main()
