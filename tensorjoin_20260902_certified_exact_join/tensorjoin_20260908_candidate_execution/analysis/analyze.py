import csv,json,statistics
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
METHODS=['original_full','layout_full','bbox64','project64','subtile16','packed16']
CELLS=['k4','k16','k64','original','k256','k1024']
def main():
    processes=[json.loads((HERE/'results'/f'confirm_{p}.json').read_text()) for p in range(3)]
    for p,r in enumerate(processes):
        assert r['pass_'] and len(r['samples'])==108 and len(r['warmups'])==36
        g=json.loads((HERE/'results'/f'confirm_{p}_guard.json').read_text());assert g['pass']
    admission=json.loads((HERE/'results/admission_a1.json').read_text());assert admission['pass_']
    independent=json.loads((HERE/'results/independent_validation.json').read_text());assert independent['pass_']
    truth={s['cell']:(s['output_sha256'],s['output_count'],s['stage_counts']) for s in admission['warmups']}
    for r in processes+[admission]:
        for key in ['warmups','samples','diagnostics']:
            for s in r.get(key,[]):assert (s['output_sha256'],s['output_count'],s['stage_counts'])==truth[s['cell']]
    med=np.empty((3,6,6));allmed=np.empty((6,6))
    for ci,c in enumerate(CELLS):
        for mi,m in enumerate(METHODS):
            allvalues=[]
            for pi,r in enumerate(processes):
                v=[s['seconds'] for s in r['samples'] if s['cell']==c and s['method']==m]
                assert len(v)==3;med[pi,ci,mi]=np.median(v);allvalues+=v
            allmed[ci,mi]=np.median(allvalues)
    means=med.mean(1);poolmean=allmed.mean(0);fixed=int(np.argmin(poolmean[:4]))
    rows=[]
    for ci,c in enumerate(CELLS):
        row=dict(cell=c)
        row.update({m:allmed[ci,mi]*1000 for mi,m in enumerate(METHODS)})
        ratio=med[:,ci,fixed]/med[:,ci,5]
        row.update(packed_vs_control_speedup=float(allmed[ci,fixed]/allmed[ci,5]),
                   process_ratio_min=float(ratio.min()),process_ratio_max=float(ratio.max()))
        rows.append(row)
    with (HERE/'analysis/summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    process_details=[]
    for pi in range(3):
        ctrl=means[pi,:4].min()
        process_details.append(dict(process=pi,mean_ms={m:float(means[pi,mi]*1000) for mi,m in enumerate(METHODS)},
            best_fixed_control=METHODS[int(np.argmin(means[pi,:4]))],
            subtile_time_saving=float(1-means[pi,4]/ctrl),packed_time_saving=float(1-means[pi,5]/ctrl)))
    passed=all(p['packed_time_saving']>=.05 for p in process_details)
    result=dict(rows=rows,pooled_mean_ms={m:float(poolmean[i]*1000) for i,m in enumerate(METHODS)},
        fixed_control=METHODS[fixed],process_details=process_details,packed_gate_pass=passed,
        pooled_packed_time_saving_vs_control=float(1-poolmean[5]/poolmean[fixed]),
        pooled_packed_speedup_vs_original=float(poolmean[0]/poolmean[5]),
        retained_measurements=324,exact_admission_calls=77,confirmation_warmups=108,
        independent_exact_calls=8,total_verified_calls=517,
        peak_device_mib=max(json.loads(p.read_text()).get('max_device_used_mib',0) for p in (HERE/'results').glob('*_guard.json')),
        direct_sample_checks=independent['checks'])
    (HERE/'analysis/summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for i,m in enumerate(METHODS):
        axes[0].plot(CELLS,allmed[:,i]*1000,'o-',label=m,linewidth=1.6)
    axes[0].set_yscale('log');axes[0].set_ylabel('Complete query time (ms, log scale)')
    axes[0].set_title('Nine retained calls per point');axes[0].legend(fontsize=8,ncol=2)
    for mi,m in [(4,'subtile16'),(5,'packed16')]:
        ratio=med[:,:,fixed]/med[:,:,mi]
        mid=np.median(ratio,axis=0)
        axes[1].plot(CELLS,mid,'o-',label=m)
        axes[1].vlines(np.arange(6),ratio.min(0),ratio.max(0),alpha=.5)
    axes[1].axhline(1,color='gray',linewidth=1)
    axes[1].set_ylabel('Speedup over '+METHODS[fixed]);axes[1].set_title('Median and range of three process ratios')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'analysis/candidate_execution.png',dpi=180)
if __name__=='__main__':main()
