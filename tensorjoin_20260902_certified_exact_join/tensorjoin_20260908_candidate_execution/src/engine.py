import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import triton
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
sys.path.insert(0,'@TENSORJOIN_ROOT@/tensorjoin_20260908_block_feasibility/src')
from executor import Sink,output
from kernels import filter_tiles,scan_rect
from prepare import build
N,D,CAP,GAMMA=output.N,output.D,output.CAP,output.GAMMA
METHODS=['original_full','layout_full','bbox64','project64','subtile16','packed16']
NEW={}
def save_kernel(key,k):NEW[key]=k
def load():
    x=output.full_source()
    a={k:np.load(HERE/'artifacts'/f'{k}.npy') for k in ['y','q','norm','perm','tr','tc','bbox']}
    r=json.loads((HERE/'artifacts/preparation.json').read_text())
    return x,a,r
def cutoff(cell,r):
    limit=(cell['T']+2**-20)*r['scale2']+r['margin']
    return float(np.nextafter(np.float32(limit),np.float32(np.inf)))
def candidates(a,r,cell,mode):
    tr=torch.from_numpy(a['tr']).cuda();tc=torch.from_numpy(a['tc']).cuda()
    q=torch.from_numpy(a['q']).cuda();norm=torch.from_numpy(a['norm']).cuda()
    n=len(tr);kind=torch.empty(n,device='cuda',dtype=torch.uint8)
    fine=torch.empty(n*16 if mode==1 else 1,device='cuda',dtype=torch.uint8)
    rows=torch.full((n*16 if mode==2 else 1,),N,device='cuda',dtype=torch.int32)
    cols=torch.full_like(rows,N);counts=torch.empty(n,device='cuda',dtype=torch.int32)
    k=filter_tiles[(n,)](q,norm,tr,tc,kind,fine,rows,cols,counts,N=N,LIMIT=cutoff(cell,r),MODE=mode,
                        num_warps=4,num_stages=3,enable_fp_fusion=False)
    save_kernel(f'filter_{mode}_{cell["name"]}',k)
    return tr,tc,kind,fine,rows,cols,counts
def call(op,radix,x,a,r,cell,method,exact=False,diagnostic=False,cold=False):
    assert method in METHODS
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    if cold:
        assert method!='original_full'
        a,r=build(x)
    built=time.perf_counter();op.set_cell(cell)
    y=x if method=='original_full' else a['y']
    v=torch.from_numpy(y).cuda();prep=op.prepare(v,16)
    pg=None if method=='original_full' else torch.from_numpy(a['perm']).cuda()
    sink=Sink(radix,pg,diagnostic)
    out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
    score=torch.empty(1,device='cuda',dtype=torch.float32)
    mode={'project64':0,'subtile16':1,'packed16':2}.get(method)
    if diagnostic:torch.cuda.synchronize()
    prepared=time.perf_counter();filter_events=[]
    if mode is None:
        mask=a['bbox']<=cutoff(cell,r) if method=='bbox64' else np.ones(len(a['tr']),bool)
        tr=torch.from_numpy(a['tr'][mask]).cuda();tc=torch.from_numpy(a['tc'][mask]).cuda()
        rows=cols=score
        plans=[(0,torch.arange(len(tr),device='cuda',dtype=torch.int64),64,64)]
        count_tensor=None
    else:
        if diagnostic:
            filter_events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            filter_events[0].record()
        tr,tc,kind,fine,rows,cols,count_tensor=candidates(a,r,cell,mode)
        if diagnostic:filter_events[1].record()
        if mode==0:plans=[(0,torch.nonzero(kind,as_tuple=True)[0],64,64)]
        elif mode==1:plans=[(1,torch.nonzero(fine,as_tuple=True)[0],16,16)]
        else:plans=[(s,torch.nonzero(kind==k,as_tuple=True)[0],m,b) for s,k,m,b in [(0,1,64,64),(2,2,16,64),(3,3,64,16)]]
        if diagnostic:filter_events[2].record()
    if diagnostic:torch.cuda.synchronize()
    filtered=time.perf_counter()
    records=[];events=[];work=[]
    for shape,jobs,m,b in plans:
        work.append(dict(shape=[m,b],tiles=len(jobs),capacity=len(jobs)*m*b))
        if shape==0:
            rt=tr[jobs];ct=tc[jobs]
        batch=CAP//(m*b)
        for off in range(0,len(jobs),batch):
            c=torch.zeros(6,device='cuda',dtype=torch.int32)
            if diagnostic:
                ev=[torch.cuda.Event(enable_timing=True) for _ in range(6)];ev[0].record()
            if shape==0:op.stage1('F16',prep,rt[off:off+batch],ct[off:off+batch],out,amb,c,score,score,score)
            else:
                k=scan_rect[(min(batch,len(jobs)-off),)](*prep,tr,tc,jobs[off:off+batch],rows,cols,out,amb,c,
                    N=N,D=D,CAP=CAP,GAMMA=GAMMA,T=cell['T'],M=m,B=b,MODE=shape,
                    num_warps=4,num_stages=3,enable_fp_fusion=False)
                save_kernel(f'scan_{shape}_{cell["name"]}',k)
            if diagnostic:ev[1].record()
            s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=CAP
            if diagnostic:ev[2].record()
            op.stage2(v,amb,out,fp64,c,n1)
            if diagnostic:ev[3].record()
            s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=CAP
            if diagnostic:ev[4].record()
            op.terminal(v,fp64,out,c,n2)
            if diagnostic:ev[5].record();events.append(ev)
            last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=CAP
            sink.push(out,n);records.append([int(s1[0]),n1,n2,n])
    torch.cuda.synchronize();computed=time.perf_counter()
    result,sr=sink.finish();end=time.perf_counter()
    rec=dict(method=method,cell=cell['name'],seconds=end-t,query_seconds=end-built,
        build_seconds=built-t,cold=cold,diagnostic=diagnostic,work=work,batches=len(records),
        stage_counts=np.sum(np.asarray(records,dtype=np.int64),axis=0).tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**sr)
    if diagnostic:
        rec.update(preparation_seconds=prepared-built,filter_list_seconds=filtered-prepared,
            execution_seconds=computed-filtered,
            filter_gpu_ms=[filter_events[i].elapsed_time(filter_events[i+1]) for i in [0,1]] if filter_events else [],
            stage_gpu_ms=np.sum([[e[i].elapsed_time(e[i+1]) for i in [0,2,4]] for e in events],axis=0).tolist(),
            survivor_pairs=int(count_tensor.to(torch.int64).sum().item()) if count_tensor is not None else None)
    rec['output_sha256']=output.check_output(result,cell,exact)
    if sink.remap_kernel is not None:save_kernel('remap',sink.remap_kernel)
    return rec
def capture():
    directory=HERE/'artifacts/compiled';directory.mkdir(exist_ok=True)
    rec={}
    for key,k in NEW.items():
        h=hashlib.sha256(k.asm['cubin']).hexdigest();rec[key]=dict(cubin_sha256=h,n_spills=k.n_spills,n_regs=k.n_regs)
        for ext in ['cubin','ptx','llir','ttgir','ttir']:
            data=k.asm[ext];data=data if isinstance(data,bytes) else data.encode()
            p=directory/(h+'.'+ext)
            if p.exists():assert p.read_bytes()==data
            else:p.write_bytes(data)
    return rec
