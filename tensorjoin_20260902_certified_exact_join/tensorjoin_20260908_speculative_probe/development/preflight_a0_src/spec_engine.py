"""Four matched scheduling arms and unchanged strong controls."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
BASE=ROOT/'tensorjoin_20260908_candidate_execution'
sys.path.insert(0,str(BASE/'src'))
import engine as legacy
from spec_kernels import plan_tiles,scan_queue

output=legacy.output
N,D,CAP,GAMMA=legacy.N,legacy.D,legacy.CAP,legacy.GAMMA
CHUNK=4096
PRIMARY=['sequential','pipeline','draft_serial','speculative']
CONTROLS=['original_full','project64','packed16']
METHODS=CONTROLS+PRIMARY
NEW={}

def load():
    return legacy.load()

def capture():
    target=HERE/'artifacts/compiled';target.mkdir(exist_ok=True)
    records={}
    for name,k in dict(legacy.NEW,**NEW).items():
        h=hashlib.sha256(k.asm['cubin']).hexdigest()
        records[name]=dict(cubin_sha256=h,n_regs=k.n_regs,n_spills=k.n_spills)
        for ext in ['cubin','ptx','llir','ttgir','ttir']:
            v=k.asm[ext];data=v if isinstance(v,bytes) else v.encode()
            p=target/(h+'.'+ext)
            if p.exists():assert p.read_bytes()==data
            else:p.write_bytes(data)
    return records

def call(op,radix,x,a,r,cell,method,exact=False,diagnostic=False,
         cold=False,force=0,decisions=False,canary=False):
    if method in CONTROLS:
        return legacy.call(op,radix,x,a,r,cell,method,exact=exact,diagnostic=diagnostic,cold=cold)
    assert method in PRIMARY
    isdraft=method in ['draft_serial','speculative']
    concurrent=method in ['pipeline','speculative']
    assert not force or isdraft
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    if cold:a,r=legacy.build(x)
    built=time.perf_counter()
    op.set_cell(cell)
    v=torch.from_numpy(a['y']).cuda();prep=op.prepare(v,16)
    perm=torch.from_numpy(a['perm']).cuda()
    q=torch.from_numpy(a['q']).cuda();norm=torch.from_numpy(a['norm']).cuda()
    tr=torch.from_numpy(a['tr']).cuda();tc=torch.from_numpy(a['tc']).cuda()
    nt=len(tr);ng=(nt+CHUNK-1)//CHUNK
    vf=torch.empty(nt,device='cuda',dtype=torch.uint8)
    df=torch.empty_like(vf)
    vc=torch.zeros(ng,device='cuda',dtype=torch.int32)
    dc=torch.zeros_like(vc)
    vj=torch.empty(ng*CHUNK,device='cuda',dtype=torch.int32)
    dj=torch.empty_like(vj)
    vp=torch.empty(nt,device='cuda',dtype=torch.int32)
    dp=torch.empty_like(vp)
    storage=[torch.empty(CAP+64,device='cuda',dtype=torch.int64) for _ in range(3)]
    if canary:
        for t in storage:t[:32].fill_(-197);t[-32:].fill_(-197)
    out,amb,fp64=[t[32:-32] for t in storage]
    sink=legacy.Sink(radix,perm,diagnostic)
    compute=torch.cuda.current_stream()
    verifier=torch.cuda.Stream() if concurrent else compute
    ready=[torch.cuda.Event() for _ in range(ng)] if concurrent else []
    origin=torch.cuda.Event(enable_timing=True) if diagnostic else None
    intervals=[]
    if origin:origin.record()
    def begin(kind,group):
        if not diagnostic:return None
        ev=torch.cuda.Event(enable_timing=True);ev.record()
        return (kind,group,ev)
    def end(item):
        if item is not None:
            ev=torch.cuda.Event(enable_timing=True);ev.record()
            intervals.append((*item,ev))
    def plan(offset,n,sample,repair):
        item=begin('draft' if sample else 'verify',offset//CHUNK)
        k=plan_tiles[(n,)](q,norm,tr,tc,df if sample else vf,df,
            dj if sample else vj,dc if sample else vc,dp if sample else vp,
            offset,TOTAL=nt,N=N,CHUNK=CHUNK,LIMIT=legacy.cutoff(cell,r),
            SAMPLE=sample,REPAIR=repair,FORCE=force if sample else 0,
            num_warps=4,num_stages=3,enable_fp_fusion=False)
        NEW[f'plan_{sample}_{repair}_{force if sample else 0}_{cell["name"]}']=k
        end(item)
    def scan(group,jobs,counts,c,kind):
        item=begin(kind,group)
        k=scan_queue[(min(CHUNK,nt-group*CHUNK),)](*prep,tr,tc,jobs,counts,out,amb,c,
            group,N=N,D=D,CAP=CAP,CHUNK=CHUNK,GAMMA=GAMMA,T=cell['T'],
            num_warps=4,num_stages=3,enable_fp_fusion=False)
        NEW[f'scan_queue_{cell["name"]}']=k
        end(item)
    prepared=time.perf_counter()
    if isdraft:plan(0,nt,True,False)
    if not concurrent:
        plan(0,nt,False,isdraft)
    else:
        # Inputs/count zeroing and, for speculation, all draft flags must be ready.
        data_ready=torch.cuda.Event();data_ready.record()
        verifier.wait_event(data_ready)
    scheduled=0
    records=[]
    for group in range(ng):
        if concurrent:
            while scheduled<min(ng,group+2):
                with torch.cuda.stream(verifier):
                    off=scheduled*CHUNK
                    plan(off,min(CHUNK,nt-off),False,isdraft)
                    ready[scheduled].record()
                scheduled+=1
        c=torch.zeros(6,device='cuda',dtype=torch.int32)
        if isdraft:scan(group,dj,dc,c,'draft_scan')
        if concurrent:compute.wait_event(ready[group])
        scan(group,vj,vc,c,'repair_scan' if isdraft else 'verified_scan')
        item=begin('refine',group)
        s1=c.cpu().numpy().copy();n1=int(s1[1])
        assert s1[2]==0 and 0<=n1<=CAP
        op.stage2(v,amb,out,fp64,c,n1)
        s2=c.cpu().numpy().copy();n2=int(s2[3])
        assert s2[2]==0 and 0<=n2<=CAP
        op.terminal(v,fp64,out,c,n2)
        last=c.cpu().numpy().copy();n=int(last[0])
        assert last[2]==0 and 0<=n<=CAP
        end(item)
        sink.push(out,n)
        records.append([int(s1[0]),n1,n2,n])
    torch.cuda.synchronize();computed=time.perf_counter()
    result,sr=sink.finish();finish=time.perf_counter()
    rec=dict(method=method,cell=cell['name'],seconds=finish-start,
        query_seconds=finish-built,build_seconds=built-start,cold=cold,
        diagnostic=diagnostic,force=force,chunk_tiles=CHUNK,batches=ng,
        stage_counts=np.sum(records,axis=0).tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),**sr)
    if diagnostic or decisions:
        vfa=vf.cpu().numpy().copy();vca=vc.cpu().numpy().copy()
        dfa=df.cpu().numpy().copy() if isdraft else np.zeros(nt,np.uint8)
        dca=dc.cpu().numpy().copy() if isdraft else np.zeros(ng,np.int32)
        rec.update(verified_tiles=int(vfa.sum()),draft_tiles=int(dca.sum()),
            repair_tiles=int(vca.sum()) if isdraft else 0,
            execution_tiles=int(vca.sum()+dca.sum()),
            unnecessary_draft_tiles=int(np.count_nonzero((dfa>0)&(vfa==0))),
            drafted_verified_tiles=int(np.count_nonzero((dfa>0)&(vfa>0))),
            candidate_pairs=int(vp.to(torch.int64).sum().item()),
            preparation_seconds=prepared-built,compute_seconds=computed-prepared)
        rec['execution_capacity']=rec['execution_tiles']*4096
        if decisions:
            rec['_decisions']=dict(verified=vfa,draft=dfa,verified_counts=vca,
                draft_counts=dca,verified_jobs=vj.cpu().numpy().copy(),
                draft_jobs=dj.cpu().numpy().copy() if isdraft else None,
                pair_counts=vp.cpu().numpy().copy())
    if diagnostic:
        rec['gpu_intervals_ms']=[dict(kind=k,group=g,start=origin.elapsed_time(s),
            end=origin.elapsed_time(e)) for k,g,s,e in intervals]
        # Event envelopes can contain dispatch gaps; trace separately checks kernels.
        bykind={}
        for v0 in rec['gpu_intervals_ms']:
            bykind[v0['kind']]=bykind.get(v0['kind'],0)+v0['end']-v0['start']
        rec['gpu_event_envelope_ms']=bykind
    if canary:
        assert all(torch.all(t[:32]==-197).item() and torch.all(t[-32:]==-197).item() for t in storage)
        rec['canaries_pass']=True
    rec['output_sha256']=output.check_output(result,cell,exact)
    if sink.remap_kernel is not None:NEW['remap']=sink.remap_kernel
    return rec
