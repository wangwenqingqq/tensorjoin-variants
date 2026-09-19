"""Aggregate all retained repetitions, with process medians as the unit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
CELLS=['k4','k16','k64','original','k256','k1024']
CONTROLS=['original_full','project64','packed16']
PRIMARY=['sequential','pipeline','draft_serial','speculative']
METHODS=CONTROLS+PRIMARY

def main():
    processes=[];raw=[]
    for i in range(3):
        path=HERE/'results'/f'confirm_{i}.json'
        record=json.loads(path.read_text());assert record['pass_']
        assert len(record['samples'])==len(CELLS)*len(METHODS)*4
        assert len(record['warmups'])==len(CELLS)*len(METHODS)
        grid=np.empty((len(CELLS),len(METHODS)))
        for ci,cell in enumerate(CELLS):
            for mi,method in enumerate(METHODS):
                vals=[v for v in record['samples'] if v['cell']==cell and v['method']==method]
                assert sorted(v['repetition'] for v in vals)==[0,1,2,3]
                grid[ci,mi]=np.median([v['query_seconds'] for v in vals])*1000
                for v in vals:
                    raw.append({k:v[k] for k in ['cell','method','process','repetition',
                        'query_seconds','output_count','output_sha256','peak_allocated_bytes',
                        'peak_reserved_bytes','output_finalize_seconds','batches']})
        processes.append(grid)
    proc=np.asarray(processes);means=proc.mean(0);mix=proc.mean(1)
    winner=CONTROLS[int(np.argmin(mix[:,:len(CONTROLS)].mean(0)))]
    si=METHODS.index('speculative');pi=METHODS.index('pipeline');wi=METHODS.index(winner)
    rows=[]
    for ci,cell in enumerate(CELLS):
        row=dict(cell=cell)
        for mi,m in enumerate(METHODS):row[m+'_ms']=float(means[ci,mi])
        for base in ['sequential','pipeline','draft_serial',winner]:
            bi=METHODS.index(base);ratios=proc[:,ci,bi]/proc[:,ci,si]
            row['spec_speedup_vs_'+base]=float(means[ci,bi]/means[ci,si])
            row['spec_speedup_vs_'+base+'_min']=float(ratios.min())
            row['spec_speedup_vs_'+base+'_max']=float(ratios.max())
        rows.append(row)
    mixtures={m:dict(mean_ms=float(mix[:,mi].mean()),process_ms=mix[:,mi].tolist())
        for mi,m in enumerate(METHODS)}
    comparisons={}
    for base in ['sequential','pipeline','draft_serial']+CONTROLS:
        bi=METHODS.index(base)
        comparisons[base]=dict(speedup=float(mix[:,bi].mean()/mix[:,si].mean()),
            saving_percent=float(100*(1-mix[:,si].mean()/mix[:,bi].mean())),
            process_saving_percent=(100*(1-mix[:,si]/mix[:,bi])).tolist())
    passed=bool(np.all(mix[:,si]<=.95*mix[:,pi]) and np.all(mix[:,si]<=.95*mix[:,wi]))
    adm=json.loads((HERE/'results/admission_a0.json').read_text());assert adm['pass_']
    coverage=[];cold=[]
    for v in adm['diagnostics']:
        if v['cold']:cold.append({k:v[k] for k in ['method','cell','build_seconds','query_seconds','seconds']})
        if v['method']=='speculative' and not v['cold'] and v.get('force',0)==0:
            coverage.append({k:v[k] for k in ['cell','verified_tiles','draft_tiles','repair_tiles',
                'execution_tiles','unnecessary_draft_tiles','drafted_verified_tiles','candidate_pairs',
                'execution_capacity']})
    guards=[];ncall=0;exact=0
    for p in sorted((HERE/'results').glob('*_guard.json')):
        g=json.loads(p.read_text());guards.append(dict(label=g['label'],pass_=g['pass'],
            max_device_used_mib=g.get('max_device_used_mib'),gpu_uuid=g.get('gpu_uuid')))
    for p in sorted((HERE/'results').glob('*.json')):
        rr=json.loads(p.read_text())
        if 'phase' not in rr or not rr.get('pass_'):continue
        calls=sum((rr.get(k,[]) for k in ['samples','warmups','diagnostics']),[])
        ncall+=len(calls)
        if rr['phase'] in ['preflight','admit','profile']:exact+=len(calls)
    summary=dict(cells=CELLS,methods=METHODS,rows=rows,mixtures=mixtures,
        process_medians_ms=proc.tolist(),strongest_fixed_control=winner,
        comparisons=comparisons,pass_gate=passed,coverage=coverage,cold=cold,
        retained_measurements=len(raw),full_calls=ncall,exact_array_calls=exact,
        guards=guards,aggregation='mean of per-process medians; six thresholds equal query weight',
        uncertainty='observed ranges across three processes; not confidence intervals',
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((HERE/'results').glob('confirm_?.json'))})
    (HERE/'analysis/summary.json').write_text(json.dumps(summary,indent=2))
    for name,data in [('summary.csv',rows),('raw_times.csv',raw),('coverage.csv',coverage),('cold.csv',cold)]:
        with (HERE/'analysis'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(pass_gate=passed,strongest_fixed_control=winner,
        mixtures=mixtures,comparisons=comparisons,retained_measurements=len(raw),full_calls=ncall)),flush=True)

if __name__=='__main__':main()
