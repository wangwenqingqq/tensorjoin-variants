"""Predeclared process-level, order-stratified analysis; no sample removal."""
import csv,json
import numpy as np
from common import HERE,LAYOUTS,PLANS,sha

def gm(x):return float(np.exp(np.mean(np.log(x))))
def main():
 runs=[];rows=[]
 for i in range(6):
  a=json.loads((HERE/'results'/f'p{i:02d}_a0.json').read_text());g=json.loads((HERE/'results'/f'p{i:02d}_a0_guard.json').read_text())
  assert a['pass'] and g['pass'] and g['exit_code']==0 and len(a['samples'])==102
  runs.append(a)
  for s in a['samples']:rows.append(dict(process=i,method_order='8_16' if i%2==0 else '16_8',**{k:v for k,v in s.items() if k in ['layout','method','phase','iteration','seconds','inner_seconds','hash','count']}))
 with (HERE/'results/samples.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rng=np.random.default_rng(20260908);groups=[np.array([0,2,4]),np.array([1,3,5])]
 ix=np.concatenate([rng.choice(g,(10000,3),replace=True) for g in groups],axis=1)
 result={'complete':True,'processes':6,'calls':len(rows),'primary_observations':252,'repeat_observations':288,'warmups':72,'replacement_runs':False,'phases':{},'scope':'complete prearranged-input operator, same logical IDs; only two fixed plans'}
 for phase,num in [('retained',7),('repeat8',8)]:
  t=np.empty((6,3,2));inner=np.empty_like(t);marginal={}
  for i,a in enumerate(runs):
   for j,layout in enumerate(LAYOUTS):
    for k,m in enumerate(PLANS):
     ss=[s for s in a['samples'] if s['phase']==phase and s['layout']==layout and s['method']==m]
     assert len(ss)==num
     fn=np.median if phase=='retained' else np.sum
     t[i,j,k]=fn([s['seconds'] for s in ss]);inner[i,j,k]=fn([s['inner_seconds'] for s in ss])
  ratios=t[:,:,1]/t[:,:,0];layout_results=[]
  for j,layout in enumerate(LAYOUTS):
   rs=ratios[:,j];boots=np.exp(np.mean(np.log(rs[ix]),axis=1));ci=np.quantile(boots,[.025,.975]).tolist();groupgm=[gm(rs[g]) for g in groups]
   int8win=bool(ci[0]>1.05 and np.sum(rs>1)>=5 and min(groupgm)>1)
   fp16win=bool(ci[1]<1/1.05 and np.sum(rs<1)>=5 and max(groupgm)<1)
   v={'layout':layout,'F16_over_F8_geometric':gm(rs),'arithmetic':float(np.mean(rs)),'CI95':ci,'process_ratios':rs.tolist(),'INT8_wins':int(np.sum(rs>1)),'FP16_wins':int(np.sum(rs<1)),'order_geomeans':{'8_16':groupgm[0],'16_8':groupgm[1]},'material_INT8':int8win,'material_FP16':fp16win,'inner_diagnostic_ratio':gm(inner[:,j,1]/inner[:,j,0]),'process_seconds':t[:,j,:].tolist()}
   for method in PLANS:
    vals=[r['seconds'] for r in rows if r['phase']==phase and r['layout']==layout and r['method']==method]
    v[method+'_marginal_p10_median_p90']=np.quantile(vals,[.1,.5,.9]).tolist()
   layout_results.append(v)
  h=np.min(t.sum(axis=1),axis=1)/np.min(t,axis=2).sum(axis=1)
  hci=np.quantile(np.exp(np.mean(np.log(h[ix]),axis=1)),[.025,.975]).tolist()
  result['phases'][phase]={'layouts':layout_results,'material_reversal':any(x['material_INT8'] for x in layout_results) and any(x['material_FP16'] for x in layout_results),'oracle_headroom':{'process_values':h.tolist(),'geometric':gm(h),'CI95':hci,'material':bool(hci[0]>1.05),'warning':'post-hoc zero-cost empirical selector over two fixed plans; no deployable or held-out planner claim'}}
 a=result['phases']['retained'];b=result['phases']['repeat8']
 common8={x['layout'] for x in a['layouts'] if x['material_INT8']} & {x['layout'] for x in b['layouts'] if x['material_INT8']}
 common16={x['layout'] for x in a['layouts'] if x['material_FP16']} & {x['layout'] for x in b['layouts'] if x['material_FP16']}
 c=json.loads((HERE/'results/census_a0.json').read_text());result['manipulation_pass']=c['manipulation_pass'];result['census_manipulation']=c['manipulation']
 result['existence_gate_pass']=bool(common8 and common16 and c['pass'] and c['manipulation_pass'])
 result['practical_selector_gate_pass']=result['existence_gate_pass'] and a['oracle_headroom']['material'] and b['oracle_headroom']['material']
 result['decision']='BOUNDED_PLAN_REVERSAL_ESTABLISHED' if result['existence_gate_pass'] else 'NO_MATERIAL_PLAN_SWITCH_EVIDENCE_IN_FROZEN_SCOPE'
 result['novelty_promotion']=False
 (HERE/'results/analysis_a0.json').open('x').write(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
