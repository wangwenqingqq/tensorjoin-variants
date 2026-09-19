"""Exact uncertainty membership and locality census, excluded from timing."""
import gc,json,os,time,traceback
import numpy as np
import torch
from common import *

def main():
 result={'started':time.time(),'pid':os.getpid(),'pass':False,'rows':[]}
 try:
  result['old_identities']=check_old();xs,orders=load();hashes={}
  for layout in LAYOUTS:
   for method in PLANS:
    op=Collector();out,rec=one(op,method,xs[layout],orders[layout]);u1=np.concatenate(op.u1);u2=np.concatenate(op.u2)
    logical1=remap(u1,orders[layout],True);logical2=remap(u2,orders[layout],True)
    hist1,d1=distribution(u1);hist2,d2=distribution(u2)
    h=[digest(logical1),digest(logical2)]
    if method in hashes:assert hashes[method]==h,('uncertainty membership changed',layout,method,hashes[method],h)
    else:hashes[method]=h
    np.savez_compressed(HERE/'artifacts'/f'{layout}_{method}_census.npz',u1_physical=u1,u2_physical=u2,u1_logical=logical1,u2_logical=logical2,hist1=hist1,hist2=hist2,batch_u1=op.batch_u1,batch_u2=op.batch_u2)
    row={'layout':layout,'method':method,'output_hash':rec['hash'],'stage_counts':rec['stage_counts'],'uncertainty_hashes':h,'u1':d1,'u2':d2,'batch_u1_quantiles':np.quantile(op.batch_u1,[0,.5,.9,1]).tolist(),'batch_u2_quantiles':np.quantile(op.batch_u2,[0,.5,.9,1]).tolist(),'compiled':op.capture()}
    result['rows'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='compiled'}),flush=True)
    op.close();del op,out,u1,u2,logical1,logical2;gc.collect();torch.cuda.empty_cache()
  changes={}
  for m in PLANS:
   ds=[r['u1'] for r in result['rows'] if r['method']==m]
   occ=[d['occupied_tiles'] for d in ds];top=[d['top10pct_mass'] for d in ds]
   batch=[]
   for layout in LAYOUTS:
    z=np.load(HERE/'artifacts'/f'{layout}_{m}_census.npz')['batch_u1'];full=z[:-1]
    batch.append({'schedule':layout,'CV':float(np.std(full)/np.mean(full)), 'top10_mass':float(np.sort(z)[-int(np.ceil(len(z)*.1)):].sum()/z.sum())})
   cv=[v['CV'] for v in batch];tm=[v['top10_mass'] for v in batch]
   changes[m]={'batch_metrics':batch,'CV_ratio':max(cv)/max(min(cv),1e-12),'CV_range':max(cv)-min(cv),'top10_mass_range':max(tm)-min(tm)}
  result['manipulation']=changes;result['manipulation_pass']=any((v['CV_ratio']>=2 and v['CV_range']>=.10) or v['top10_mass_range']>=.10 for v in changes.values());result['pass']=True
 except Exception:result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 finally:
  result['ended']=time.time();(HERE/'results/census_a0.json').open('x').write(json.dumps(result,indent=2))
 print(json.dumps({'pass':result['pass'],'manipulation':result.get('manipulation'),'manipulation_pass':result.get('manipulation_pass')}),flush=True)
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
