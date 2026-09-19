"""Predeclared process-level paired estimates; never delete rejected samples."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
def j(p):return json.loads(p.read_text())
def gm(x):return float(np.exp(np.mean(np.log(x))))
def stats(records,phase):
    rows=[];pools={'A':[],'C':[]}
    for r in records:
        samples={m:[s['seconds'] for s in r['samples'] if s['method']==m and s['phase']==phase] for m in ['A','C']}
        expected=9 if phase=='retained' else 16
        assert all(len(x)==expected for x in samples.values())
        totals={m:float(np.median(x)) if phase=='retained' else float(np.sum(x)) for m,x in samples.items()}
        ratio=totals['C']/totals['A']
        rows.append({'label':r['label'],'order':r['order'],'A':totals['A'],'C':totals['C'],'ratio_C_over_A':ratio})
        for m in pools:pools[m].extend(samples[m])
    a=np.array([x['ratio_C_over_A'] for x in rows]);idx={o:np.array([i for i,x in enumerate(rows) if x['order']==o]) for o in ['AC','CA']}
    assert all(len(v)==4 for v in idx.values())
    rng=np.random.default_rng(20260908);draws=[]
    for _ in range(10000):
        picks=np.r_[rng.choice(idx['AC'],4,replace=True),rng.choice(idx['CA'],4,replace=True)]
        draws.append(gm(a[picks]))
    ci=np.quantile(draws,[.025,.975]).tolist()
    return {'phase':phase,'process_denominator':'per-method sample median' if phase=='retained' else 'sum of16 complete-call latencies','processes':rows,'process_wins':int(np.count_nonzero(a>1)),'paired_geometric_ratio':gm(a),'paired_arithmetic_ratio':float(np.mean(a)),'paired_stratified_bootstrap_95ci':ci,'marginal_sample_median_ratio':float(np.median(pools['C'])/np.median(pools['A'])),'order_geometric_ratios':{o:gm(a[i]) for o,i in idx.items()},'sample_p10_median_p90_seconds':{m:np.quantile(x,[.1,.5,.9]).tolist() for m,x in pools.items()}}

def main():
    schedule=j(HERE/'schedule_g1.json');records=[];failures=[];csvrows=[]
    for spec in schedule['processes']:
        path=HERE/'results'/f"{spec['label']}.json";gp=HERE/'results'/f"{spec['label']}_guard.json"
        if not path.exists() or not gp.exists():failures.append({'label':spec['label'],'reason':'missing result/guard'});continue
        r,g=j(path),j(gp)
        if not r['pass'] or not g['pass'] or g.get('exit_code')!=0:failures.append({'label':spec['label'],'reason':'failed gate','result':r.get('exception'),'guard':g.get('exception')});continue
        assert r['order']==spec['order'];assert len(r['samples'])==54
        for m in ['A','C']:
            assert [s['phase'] for s in r['samples'] if s['method']==m]==['warmup']*2+['retained']*9+['repeat16']*16
        for s in r['samples']:
            assert s['count']==3926078 and s['hash']=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495'
            assert s['seconds']>0 and math.isfinite(s['seconds'])
            csvrows.append({k:s[k] for k in ['method','phase','iteration','seconds','count','hash']}|{'process':spec['index'],'order':spec['order'],'label':r['label']})
        records.append(r)
    out={'complete':len(records)==8 and not failures,'valid_processes':len(records),'failures':failures,'no_replacements':True,'novelty_promotion':False,'old_headline_promoted':False}
    if out['complete']:
        primary=stats(records,'retained');repeat=stats(records,'repeat16')
        tail=[]
        for r in records:
            for m in ['A','C']:
                a=[s['seconds'] for s in r['samples'] if s['method']==m and s['phase']=='repeat16']
                tail.append({'label':r['label'],'method':m,'last4_over_first4':float(np.mean(a[-4:])/np.mean(a[:4]))})
        signatures={m:sorted(set(tuple(s['stage_counts']) for r in records for s in r['samples'] if s['method']==m)) for m in ['A','C']}
        assert all(len(v)==1 for v in signatures.values())
        win=primary['paired_stratified_bootstrap_95ci'][0]>1 and primary['process_wins']>=7 and min(primary['order_geometric_ratios'].values())>1 and repeat['paired_stratified_bootstrap_95ci'][0]>1 and min(repeat['order_geometric_ratios'].values())>1
        out['material_geomean_10pct']=primary['paired_geometric_ratio']>=1.10
        out.update(primary=primary,repeat16=repeat,repeat_tail=tail,stage_count_signatures=signatures,narrow_screen_win=win,decision='narrow_screen_pass' if win else 'screen_criterion_not_met',limits=['warmed repeated complete calls, not old one-shot absolute times','one public input and one threshold','new diagnostic host adapter with original GPU code objects','owned cuBLAS FP16 + directed FP32 interval + original stage2/terminal; not strongest possible fused FP16','same compiled identities as admitted control; no novelty claim'])
    else:out.update(narrow_screen_win=False,decision='incomplete_no_promotion')
    p=HERE/'results/g1_analysis.json';assert not p.exists()
    with p.open('x') as f:json.dump(out,f,indent=2)
    with (HERE/'results/g1_samples.csv').open('x',newline='') as f:
        fields=['process','order','label','method','phase','iteration','seconds','count','hash'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csvrows)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
