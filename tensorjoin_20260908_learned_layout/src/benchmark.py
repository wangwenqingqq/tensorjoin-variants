import gc
import torch
from layout_common import *
from models import configure,load_model,mapped_scores,parameter_count
import executor as retained

METHODS=['F8','F16']
CONFIGS=['original_full','pca_geo','linear_geo','mlp_geo','mlp_full']

def selected_models(device='cuda'):
    selection=json.loads((HERE/'artifacts/model_selection.json').read_text())
    out={}
    for family in ['linear','mlp']:
        spec=selection['models'][family]
        p=HERE/'artifacts'/(spec['name']+'.pt')
        assert sha(p)==spec['checkpoint_sha256']
        out[family]=load_model(p,device)
    return out

def from_scores(x,q,scores,scale2,family):
    start=time.perf_counter()
    perm=np.argsort(scores,kind='stable').astype(np.int64)
    sorted_at=time.perf_counter()
    b,_=projected_bounds(q,perm,scale2)
    bounded_at=time.perf_counter()
    y=np.ascontiguousarray(x[perm])
    end=time.perf_counter()
    return dict(layout=family,perm=perm,y=y,bounds=b),dict(sort_seconds=sorted_at-start,
        bound_seconds=bounded_at-sorted_at,reorder_seconds=end-bounded_at)

def rebuild(x,family,model=None):
    start=time.perf_counter()
    _,q,pca,scale2,features=make_features(x)
    mapped_at=time.perf_counter()
    scores=pca if family=='pca' else mapped_scores(model,q)
    mapped_end=time.perf_counter()
    plan,components=from_scores(x,q,scores,scale2,family)
    end=time.perf_counter()
    rec=dict(seconds=end-start,features=features,mapping_seconds=mapped_end-mapped_at,
        **components,permutation_sha256=digest(plan['perm']))
    return plan,rec

def saved_plans(x):
    plans={}
    for family in FAMILIES:
        perm=np.load(HERE/'artifacts'/f'{family}_permutation.npy')
        b=dict(np.load(HERE/'artifacts'/f'{family}_bounds.npz'))
        plans[family]=dict(layout=family,perm=perm,y=np.ascontiguousarray(x[perm]),bounds=b)
    return plans

def run(op,radix,x,cell,method,configuration,plans,models,exact=False,build=False,diagnostic=False):
    if configuration=='original_full':
        family,mode='pca','original_full'
    else:
        family,tail=configuration.split('_')
        mode='geometric' if tail=='geo' else 'layout_full'
    construction=None
    if build:
        assert mode=='geometric'
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        total_start=time.perf_counter()
        prepared,construction=rebuild(x,family,models.get(family))
        build_peak=torch.cuda.max_memory_allocated()
    else:
        prepared=plans[family]
        build_peak=0
    output,record,kernel=retained.execute(op,radix,x,cell,method,mode,prepared,diagnostic=diagnostic)
    total_end=time.perf_counter()
    if build:
        record['query_seconds']=record['seconds']
        record['seconds']=total_end-total_start
        record['build_wall_seconds']=record['seconds']-record['query_seconds']
        record['build']=construction
        assert np.array_equal(prepared['perm'],plans[family]['perm'])
        assert np.array_equal(prepared['bounds']['combined'],plans[family]['bounds']['combined'])
    record.update(configuration=configuration,family=family,includes_build=build,
        includes_training=False,build_peak_allocated_bytes=build_peak)
    record['output_sha256']=retained.output.check_output(output,cell,exact)
    del output
    return record,kernel

def capture_remap(kernel):
    if kernel is None:return None
    stem='remap_'+hashlib.sha256(kernel.asm['cubin']).hexdigest()[:16]
    rec={}
    for ext in ['cubin','ptx','llir','ttgir','ttir']:
        data=kernel.asm[ext]
        data=data if isinstance(data,bytes) else data.encode()
        p=HERE/'artifacts'/(stem+'.'+ext)
        h=hashlib.sha256(data).hexdigest()
        if p.exists():assert sha(p)==h
        else:p.write_bytes(data)
        rec[ext]=h
    assert kernel.n_spills==0
    return rec

def verify_timing_freeze():
    for p,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
        assert sha(p)==h,p

def mapping_benchmark(q,models):
    cpu=selected_models('cpu')
    rows=[]
    for family in ['linear','mlp']:
        for device in ['cuda','cpu']:
            model=models[family] if device=='cuda' else cpu[family]
            mapped_scores(model,q,device)
            values=[]
            for _ in range(7):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                scores=mapped_scores(model,q,device)
                end=time.perf_counter()
                assert np.isfinite(scores).all()
                values.append(end-start)
            # Dense layer multiply-add estimate; normalization/activation,
            # copies and launches are included in times but not this estimate.
            macs=32 if family=='linear' else 32*64+64*64+64
            rows.append(dict(family=family,device=device,parameters=parameter_count(model),
                host_to_host_seconds=values,median_seconds=float(np.median(values)),
                dense_layer_flops=2*len(q)*macs,rows=len(q),input_features=32,
                scope='One-pass scalar map only; not FM or an ODE solver.'))
    return rows
