#!/usr/bin/env python3
"""Report all retained observations with frozen marginal and paired estimators."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

A='baseline'; B='chunked_nobubble'
METRICS=['public_seconds','join_excluding_shared_sort_seconds','shared_host_canonical_seconds']

def distribution(values):
    a=np.asarray(values,dtype=np.float64)
    return {'n':len(a),'median':float(np.median(a)),'p10':float(np.percentile(a,10)),
            'p90':float(np.percentile(a,90)),'arithmetic_mean':float(np.mean(a)),
            'geometric_mean':float(np.exp(np.mean(np.log(a)))),'raw':a.tolist()}


def summarize(paths):
    documents=[json.loads(Path(p).read_text()) for p in paths]
    samples=[s for d in documents for s in d['samples']]
    assert all(s['eligible'] and s['correctness']['exact_contract_pass'] for s in samples)
    pairs=[]
    for process,d in enumerate(documents):
        assert not d.get('invalid') and not d.get('interrupted_by_contention')
        for i in range(0,len(d['samples']),2):
            ab={s['variant']:s for s in d['samples'][i:i+2]}
            assert set(ab)=={A,B}
            pairs.append({'process':process,'order':d['samples'][i]['variant'],**ab})
    rng=np.random.default_rng(20261028)
    bootstrap=rng.integers(0,len(pairs),size=(10000,len(pairs)))
    metrics={}
    for metric in METRICS:
        av=np.asarray([p[A][metric] for p in pairs]);bv=np.asarray([p[B][metric] for p in pairs])
        marginal=np.median(av)/np.median(bv)
        boot=np.median(av[bootstrap],axis=1)/np.median(bv[bootstrap],axis=1)
        log=np.log(av/bv)
        paired=np.exp(np.mean(log))
        pairedboot=np.exp(np.mean(log[bootstrap],axis=1))
        order={}
        for first in [A,B]:
            idx=[i for i,p in enumerate(pairs) if p['order']==first]
            if idx:order[first+'_first']={'n':len(idx),
                'median_A':float(np.median(av[idx])),'median_B':float(np.median(bv[idx])),
                'marginal_ratio':float(np.median(av[idx])/np.median(bv[idx]))}
        metrics[metric]={'A':distribution(av),'B':distribution(bv),
            'marginal_median_speedup':float(marginal),
            'marginal_paired_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist(),
            'paired_geometric_speedup':float(paired),
            'paired_geometric_95_ci':np.percentile(pairedboot,[2.5,97.5]).tolist(),
            'B_wins':int(np.count_nonzero(bv<av)),'pairs':len(pairs),'order_split':order,
            'raw_pair_ratios':(av/bv).tolist()}
    passed=all(metrics[m]['marginal_median_speedup']>=1.10 for m in METRICS[:2])
    return {'A':A,'B':B,'processes':len(documents),'pairs':len(pairs),
            'source_records':paths,'metrics':metrics,'screen_gate_pass':passed,
            'all_correct':True,'all_isolated':True,'bootstrap_seed':20261028,
            'bootstrap_resamples':10000,'samples':samples}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('paths',nargs='+');a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    r=summarize(a.paths)
    path.write_text(json.dumps(r,indent=2)+'\n')
    for k,v in r['metrics'].items():print(k,v['A']['median'],v['B']['median'],v['marginal_median_speedup'],v['marginal_paired_bootstrap_95_ci'],v['B_wins'])
    print('SCREEN_GATE_PASS',r['screen_gate_pass'])
