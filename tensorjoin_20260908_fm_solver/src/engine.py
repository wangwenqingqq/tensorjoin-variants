"""Full solver execution; neural scores never directly remove a candidate."""
from routing import *
import executor as retained


def capture_remap(kernel):
    if kernel is None:return None
    import hashlib
    stem='remap_'+hashlib.sha256(kernel.asm['cubin']).hexdigest()[:16];record={}
    for ext in ['cubin','ptx','llir','ttgir','ttir']:
        data=kernel.asm[ext];data=data if isinstance(data,bytes) else data.encode()
        path=HERE/'artifacts'/(stem+'.'+ext)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        record[ext]=sha(path)
    assert kernel.n_spills==0
    return record


def run(op,radix,x,cell,method,configuration,plan,models,exact=False,build=False,diagnostic=False):
    op.set_cell(cell);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();construction=None
    if build:prepared,construction=rebuild_plan(x,configuration)
    else:prepared=plan
    built=time.perf_counter()
    mask,routing,certificate_kernel,route_audit=route(prepared,models,cell,configuration,diagnostic)
    tr,tc=prepared['bounds']['tr'][mask],prepared['bounds']['tc'][mask]
    listed=time.perf_counter()
    v=torch.from_numpy(prepared['y']).to('cuda')
    arithmetic=op.prepare(v,8 if method=='F8' else 16)
    pg=torch.from_numpy(prepared['perm']).to('cuda');sink=retained.Sink(radix,pg,diagnostic)
    trg,tcg=torch.from_numpy(tr).to('cuda'),torch.from_numpy(tc).to('cuda')
    out=torch.empty(retained.CAP,device='cuda',dtype=torch.int64)
    amb,fp64=torch.empty_like(out),torch.empty_like(out)
    score=torch.empty(1,device='cuda',dtype=torch.int32 if method=='F8' else torch.float32);dummy=score.view(torch.float32)
    records=[]
    for off in range(0,len(tr),4096):
        c=torch.zeros(6,device='cuda',dtype=torch.int32)
        op.stage1(method,arithmetic,trg[off:off+4096],tcg[off:off+4096],out,amb,c,score,dummy,dummy)
        s1=c.cpu().numpy().copy();n1=int(s1[1]);assert s1[2]==0 and 0<=n1<=retained.CAP
        op.stage2(v,amb,out,fp64,c,n1)
        s2=c.cpu().numpy().copy();n2=int(s2[3]);assert s2[2]==0 and 0<=n2<=retained.CAP
        op.terminal(v,fp64,out,c,n2)
        last=c.cpu().numpy().copy();n=int(last[0]);assert last[2]==0 and 0<=n<=retained.CAP
        sink.push(out,n)
        records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
    torch.cuda.synchronize();output,finalized=sink.finish();end=time.perf_counter()
    record=dict(seconds=end-start,query_seconds=end-built,build_wall_seconds=built-start,build=construction,
        cell=cell['name'],method=method,configuration=configuration,includes_build=build,includes_training=False,
        includes_model_inference=configuration.startswith(('direct','fm')),includes_ode=configuration.startswith('fm'),
        executed_tiles=len(tr),batches=len(records),stage_counts=np.sum(np.asarray(records,dtype=np.int64),axis=0).tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        list_and_routing_seconds=listed-built,routing=routing,**finalized)
    record['output_sha256']=retained.output.check_output(output,cell,exact)
    record['tile_mask_sha256']=digest(mask);record['certificate_selection_sha256']=digest(route_audit['chosen'])
    if build:
        assert np.array_equal(prepared['perm'],plan['perm'])
        assert np.array_equal(prepared['bounds']['combined'],plan['bounds']['combined'])
        if prepared['features'] is not None:
            assert np.array_equal(prepared['features'],plan['features'])
            assert np.array_equal(prepared['coarse'],plan['coarse'])
    del output
    return record,sink.remap_kernel,certificate_kernel,mask
