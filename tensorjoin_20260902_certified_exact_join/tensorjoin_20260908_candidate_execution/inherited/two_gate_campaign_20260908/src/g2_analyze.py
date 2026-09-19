"""Predeclared paired-process estimators; all raw samples and all orders."""
import csv,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
METHODS=['F8','S8','F16','S16']
def j(p):return json.loads(p.read_text())
def gm(x):return float(np.exp(np.mean(np.log(x))))

def phase(records,phase):
 rows=[];pools={m:[] for m in METHODS}
 for r in records:
  samples={m:[s['seconds'] for s in r['samples'] if s['method']==m and s['phase']==phase] for m in METHODS}
  assert all(len(x)==(9 if phase=='retained' else 16) for x in samples.values())
  t={m:float(np.median(x) if phase=='retained' else np.sum(x)) for m,x in samples.items()}
  ratios=dict(R_fused=t['F16']/t['F8'],R_split=t['S16']/t['S8'],G8=t['S8']/t['F8'],G16=t['S16']/t['F16'])
  ratios['interaction']=ratios['G8']/ratios['G16'];assert np.isclose(ratios['interaction'],ratios['R_fused']/ratios['R_split'])
  rows.append(dict(label=r['label'],order=r['order'],order_id=r['order_id'],seconds=t,ratios=ratios))
  for m in METHODS:pools[m].extend(samples[m])
 idx={o:np.array([i for i,x in enumerate(rows) if x['order_id']==o]) for o in range(4)};assert all(len(v)==2 for v in idx.values())
 rng=np.random.default_rng(20260908);picks=np.array([np.concatenate([rng.choice(idx[o],2,replace=True) for o in range(4)]) for _ in range(10000)])
 estimates={}
 for k in rows[0]['ratios']:
  a=np.array([x['ratios'][k] for x in rows]);draws=np.exp(np.log(a[picks]).mean(axis=1))
  estimates[k]=dict(geometric_ratio=gm(a),arithmetic_ratio=float(a.mean()),bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),wins=int(np.count_nonzero(a>1)),order_geometric_ratios={o:gm(a[i]) for o,i in idx.items()})
 return dict(phase=phase,process_denominator='method median' if phase=='retained' else 'sum of 16 complete calls',processes=rows,estimates=estimates,sample_p10_median_p90_seconds={m:np.quantile(x,[.1,.5,.9]).tolist() for m,x in pools.items()},marginal_medians={m:float(np.median(x)) for m,x in pools.items()})

def main():
 records=[];failures=[];csvrows=[]
 for spec in j(HERE/'schedule_g2.json')['processes']:
  p=HERE/'results'/f"{spec['label']}.json";g=HERE/'results'/f"{spec['label']}_guard.json"
  if not p.exists() or not g.exists():failures.append(dict(label=spec['label'],reason='missing'));continue
  r,g=j(p),j(g)
  if not r['pass'] or not g['pass'] or g.get('exit_code')!=0:failures.append(dict(label=spec['label'],reason='failed',result=r.get('exception'),guard=g.get('exception')));continue
  assert r['order']==spec['order'] and len(r['samples'])==108
  for m in METHODS:assert [s['phase'] for s in r['samples'] if s['method']==m]==['warmup']*2+['retained']*9+['repeat16']*16
  for s in r['samples']:
   assert s['count']==3926078 and s['hash']=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495' and s['seconds']>0 and math.isfinite(s['seconds'])
   csvrows.append({k:s[k] for k in ['method','phase','iteration','seconds','count','hash']}|dict(process=spec['index'],order=','.join(spec['order']),label=r['label']))
  records.append(r)
 out=dict(complete=len(records)==8 and not failures,failures=failures,valid_processes=len(records),no_replacements=True,novelty_promotion=False)
 if out['complete']:
  pri,rep=phase(records,'retained'),phase(records,'repeat16')
  def material(r):
   x=r['estimates']['R_fused'];return x['bootstrap_95ci'][0]>1.10 and x['wins']>=7 and min(x['order_geometric_ratios'].values())>1
  win=material(pri) and material(rep)
  interaction=all(x['estimates']['interaction']['bootstrap_95ci'][0]>1.10 for x in [pri,rep])
  signatures={m:sorted(set(tuple(s['stage_counts']) for r in records for s in r['samples'] if s['method']==m)) for m in METHODS}
  assert all(len(x)==1 for x in signatures.values()) and signatures['F8']==signatures['S8'] and signatures['F16']==signatures['S16']
  tails=[]
  for r in records:
   for m in METHODS:
    x=[s['seconds'] for s in r['samples'] if s['method']==m and s['phase']=='repeat16'];tails.append(dict(label=r['label'],method=m,last4_over_first4=float(np.mean(x[-4:])/np.mean(x[:4]))))
  out.update(primary=pri,repeat16=rep,repeat_tail=tails,stage_count_signatures=signatures,material_matched_fusion_advantage=win,int8_specific_fusion_interaction=interaction,decision='material_cost_hypothesis_retained_not_novelty' if win else 'material_cost_criterion_not_met',limits=['one workload, shape, threshold, GPU and fixed FP64 reference','native dtypes have different metadata, storage, MMA geometry and refinement counts','F8 is original cubin, F16 conventional custom fused control not fastest possible kernel','C-to-F16 differences are not fusion-only','experiments alone do not establish a non-incremental prior-art distinction'])
 else:out['decision']='incomplete_no_promotion'
 with (HERE/'results/g2_analysis.json').open('x') as f:json.dump(out,f,indent=2)
 with (HERE/'results/g2_samples.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['process','order','label','method','phase','iteration','seconds','count','hash']);w.writeheader();w.writerows(csvrows)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
