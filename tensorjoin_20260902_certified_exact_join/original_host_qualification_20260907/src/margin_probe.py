"""Additional same-cubin composition probe; original Engine methods unchanged."""
import json,os,platform,time,traceback
import numpy as np
import torch
from frozen_driver import Driver,HERE,verify
from validate import Engine,N,D,EPS,selected_ids,digest,write_new
label='p8_margin_a0';path=HERE/'results'/f'{label}.json';assert not path.exists()
r={'label':label,'pass':False,'timing_claim':False,'pid':os.getpid(),'host':platform.node(),'started':time.time()}
try:
    assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='2'
    rng=np.random.default_rng(71907);x=np.zeros((N,D),np.float32)
    directions=rng.normal(size=(64,D));directions/=np.linalg.norm(directions,axis=1)[:,None]
    margins=np.tile(np.array([-.002,-.001,-.0005,-.0001,0,.0001,.0005,.001]),8)
    x[64:128]=(directions*(EPS+margins[:,None])).astype(np.float32);x[128:192]=x[64:128]
    tr,tc=np.triu_indices(3);tr=tr.astype(np.int32);tc=tc.astype(np.int32);ii=selected_ids(tr,tc)
    d=Driver();e=Engine(d,x);truth=e.direct(ii);out,stats=e.cascade(tr,tc,truth,True)
    r.update(loaded_identity=d.identities,identity_after=verify(),pairs=len(ii),direct_hash=digest(truth),cascade=stats)
    np.savez_compressed(HERE/'results'/f'{label}_fixture.npz',vectors=x[:192],tile_rows=tr,tile_columns=tc,margins=margins)
    assert all(stats[k]>0 for k in ['stage1_accept','stage1_ambiguous','stage2_accept','stage2_reject'])
    assert r['loaded_identity']==r['identity_after'];e.canaries();r['pass']=True
except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
finally:r['ended']=time.time();write_new(path,r)
print(json.dumps(r,indent=2),flush=True)
raise SystemExit(0 if r['pass'] else 1)
