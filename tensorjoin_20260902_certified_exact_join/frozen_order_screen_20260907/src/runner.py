"""Run one predeclared frozen-code paired process or a diagnostic gate."""
import argparse,gc,hashlib,json,os,platform,sys,time,traceback
import numpy as np
import torch
from operators import Operators,full_source,output_check,BASE
from frozen_driver import HERE,ROOT,verify,sha

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['validate','probes','paired'],required=True);p.add_argument('--order',choices=['AB','BA'],default='AB');a=p.parse_args();path=HERE/'results'/f'{a.label}.json';assert not path.exists()
    r={'label':a.label,'mode':a.mode,'order':a.order,'pass':False,'started':time.time(),'pid':os.getpid(),'host':platform.node(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':np.__version__,'samples':[],'scope':'warmed pageable FP32 host to sorted canonical uint64 host','novelty_promotion':False}
    try:
        assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='2' and os.environ['NVIDIA_TF32_OVERRIDE']=='0'
        o=Operators();r.update(identities=o.identities,runtime_abi=o.abi,library=o.library_record,library_hashes=o.library_hashes)
        print(json.dumps({'phase':'initialized','mode':a.mode,'order':a.order}),flush=True)
        if a.mode in ['validate','probes']:r['prefix_probes']=o.prefix_probe()
        if a.mode!='probes':
            x=full_source()
            def one(m,phase,i):
                out,rec=o.run(m,x);h=output_check(out);rec.update(method=m,phase=phase,iteration=i,hash=h,count=len(out));r['samples'].append(rec)
                print(json.dumps({k:v for k,v in rec.items() if k!='stage_counts'}),flush=True)
            if a.mode=='validate':
                for m in a.order:one(m,'diagnostic',0)
            else:
                gate=json.loads((HERE/'results/validate_a0.json').read_text());assert gate['pass']
                for m in a.order:
                    for i in range(2):one(m,'warmup',i)
                    for i in range(9):one(m,'retained',i)
                for m in reversed(a.order):
                    for i in range(16):one(m,'repeat16',i)
        after=verify()
        for spec in BASE.values():
            for ext,h in spec['hashes'].items():
                key=spec['stem']+ext;assert sha(ROOT/key)==h;after[key]=h
        assert after==o.identities;r['identity_after']=after;r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['pass']=True
    except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:r['ended']=time.time();save(path,r)
    print(json.dumps({'result':str(path),'pass':r['pass']}),flush=True)
    return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
