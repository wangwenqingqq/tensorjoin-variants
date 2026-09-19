import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from spec_engine import *

def write(path,record):
    with Path(path).open('x') as f:json.dump(record,f,indent=2)

def release():
    gc.collect();torch.cuda.empty_cache()

def freeze():
    deps=json.loads((BASE/'artifacts/timing_freeze.json').read_text())
    paths=list((HERE/'src').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'targets_r2.json']
    paths += [BASE/'artifacts/timing_freeze.json',BASE/'REPORT.md']
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and filename.endswith('.py') and filename.startswith('@TENSORJOIN_ROOT@/tensorjoin'):
            paths.append(Path(filename))
    deps.update({str(p):output.sha(p) for p in paths})
    write(HERE/'artifacts/timing_freeze.json',deps)

def verify_queues(d):
    vf=d['verified']>0;df=d['draft']>0
    for flags,counts,jobs in [(df,d['draft_counts'],d['draft_jobs']),
             (vf&~df,d['verified_counts'],d['verified_jobs'])]:
        if jobs is None:
            assert not df.any()
            continue
        for g,cnt in enumerate(counts):
            want=np.flatnonzero(flags[g*CHUNK:(g+1)*CHUNK])+g*CHUNK
            got=jobs[g*CHUNK:g*CHUNK+int(cnt)]
            assert np.array_equal(np.sort(got),want),(g,len(got),len(want))
    assert np.all(vf <= (df|(vf&~df)))
    return dict(verified_tiles=int(vf.sum()),draft_tiles=int(df.sum()),
        repair_tiles=int((vf&~df).sum()),queue_groups=len(d['verified_counts']),pass_=True)

def check_cpu_draft(a,r,cell,d):
    rng=np.random.default_rng(2026090837)
    tr,tc=a['tr'],a['tc'];nt=len(tr)
    picks=np.unique(np.r_[rng.integers(0,nt,96),np.flatnonzero(tr==tc)[[0,1,500,-2,-1]],
        np.flatnonzero(tc==937)[[0,1,500,-2,-1]]])
    near=0;checked=0
    for p in picks:
        ri=int(tr[p])*64+np.arange(0,64,4);ci=int(tc[p])*64+np.arange(0,64,4)
        ri=ri[ri<N];ci=ci[ci<N]
        aa=a['q'][ri].astype(np.float64);bb=a['q'][ci].astype(np.float64)
        # Use independent direct differences, allowing a small dot/norm rounding gap.
        dist=np.sum((aa[:,None,:]-bb[None,:,:])**2,axis=2)
        valid=ri[:,None]<=ci[None,:]
        lo=float(np.min(np.where(valid,dist,np.inf)))
        lim=legacy.cutoff(cell,r)
        if abs(lo-lim)<=1e-5:near+=1
        else:assert bool(d['draft'][p])==(lo<lim),(p,lo,lim)
        checked+=int(valid.sum())
    return dict(tiles=len(picks),pairs=checked,threshold_near_tiles=near,pass_=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True)
    p.add_argument('--phase',choices=['preflight','admit','confirm','profile'],required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--profile-method',choices=PRIMARY,default='speculative')
    args=p.parse_args();torch.set_num_threads(4)
    x,a,r=load();op=output.Matrix();radix=output.Radix()
    record=dict(label=args.label,phase=args.phase,started=time.time(),samples=[],warmups=[],
        diagnostics=[],validation=[],pass_=False,environment=dict(torch=torch.__version__,
        triton=legacy.triton.__version__,device=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,python=sys.version))
    try:
        if args.phase in ['confirm','profile']:
            for path,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
                assert output.sha(Path(path))==h,path
        cell0=next(c for c in output.CELLS if c['name']=='original')
        if args.phase=='preflight':
            for m in PRIMARY:
                rec=call(op,radix,x,a,r,cell0,m,exact=True,diagnostic=True,decisions=True,canary=True)
                d=rec.pop('_decisions');record['validation'].append(verify_queues(d))
                del d;record['diagnostics'].append(rec)
                print(json.dumps(rec),flush=True);release()
        elif args.phase=='admit':
            for cell in output.CELLS:
                tg,cg,kg,fg,rg,clg,cnt=legacy.candidates(a,r,cell,0)
                want_flags=kg.cpu().numpy().copy();want_counts=cnt.cpu().numpy().copy()
                del tg,cg,kg,fg,rg,clg,cnt;release()
                for m in METHODS:
                    rec=call(op,radix,x,a,r,cell,m,exact=True,diagnostic=True,
                        decisions=m in PRIMARY,canary=m in PRIMARY)
                    if m in PRIMARY:
                        d=rec.pop('_decisions')
                        assert np.array_equal(d['verified'],want_flags),(cell['name'],m,'flags')
                        assert np.array_equal(d['pair_counts'],want_counts),(cell['name'],m,'counts')
                        check=verify_queues(d);check.update(cell=cell['name'],method=m)
                        if m=='speculative':check['cpu_draft']=check_cpu_draft(a,r,cell,d)
                        record['validation'].append(check);del d
                    record['diagnostics'].append(rec);print(json.dumps(rec),flush=True);release()
            for force in [1,2,3]:
                rec=call(op,radix,x,a,r,cell0,'speculative',exact=True,force=force,
                    diagnostic=True,decisions=True,canary=True)
                d=rec.pop('_decisions');check=verify_queues(d);check['force']=force
                record['validation'].append(check);del d
                record['diagnostics'].append(rec);print(json.dumps(rec),flush=True);release()
            for m in PRIMARY:
                rec=call(op,radix,x,a,r,cell0,m,exact=True,cold=True,canary=True)
                record['diagnostics'].append(rec);print(json.dumps(rec),flush=True);release()
            record['compiled_new']=capture();record['compiled_inherited']=op.capture()
            freeze()
        elif args.phase=='profile':
            m=args.profile_method
            rec=call(op,radix,x,a,r,cell0,m,exact=True)
            record['warmups'].append(rec);release()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('tensorjoin_'+m)
            rec=call(op,radix,x,a,r,cell0,m,exact=True,diagnostic=True)
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            record['diagnostics'].append(rec);print(json.dumps(rec),flush=True)
        else:
            configs=[(c,m) for c in output.CELLS for m in METHODS]
            for cell,m in configs:
                rec=call(op,radix,x,a,r,cell,m)
                record['warmups'].append(rec);print(json.dumps(dict(warmup=True,**rec)),flush=True);release()
            for rep in range(4):
                seq=configs if (args.index+rep)%2==0 else list(reversed(configs))
                shift=(args.index*11+rep*7)%len(seq);seq=seq[shift:]+seq[:shift]
                for cell,m in seq:
                    rec=call(op,radix,x,a,r,cell,m)
                    rec.update(process=args.index,repetition=rep)
                    record['samples'].append(rec);print(json.dumps(rec),flush=True);release()
            record['compiled_new']=capture();record['compiled_inherited']=op.capture()
        record.update(pass_=True,ended=time.time())
    except Exception:
        record['exception']=traceback.format_exc();raise
    finally:write(HERE/'results'/f'{args.label}.json',record)

if __name__=='__main__':main()
