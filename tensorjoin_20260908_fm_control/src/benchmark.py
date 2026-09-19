import gc
import torch
from grouping import *
from models import configure,load_model,mapped_coordinates,parameter_count
import executor as retained

METHODS=['F8','F16']
CONFIGS=['original_full','pca_geo','direct_geo','fm_geo','fm_full']


def selected_models(device='cuda'):
    selection=json.loads((HERE/'artifacts/model_selection.json').read_text())
    models={}
    for family in ['direct','fm']:
        spec=selection['models'][family]
        path=HERE/'artifacts'/(spec['name']+'.pt')
        assert sha(path)==spec['checkpoint_sha256'] and spec['trained']
        model=load_model(path,device)
        model.solver_steps=spec['solver_steps']
        model.rule=spec['rule']
        models[family]=model
    return models


def from_coordinates(x,q,coords,scale2,family,rule,first_scores=None):
    start=time.perf_counter()
    perm=permutation(coords,rule,first_scores)
    grouped=time.perf_counter()
    bounds,_=projected_bounds(q,perm,scale2)
    bounded=time.perf_counter()
    y=np.ascontiguousarray(x[perm])
    end=time.perf_counter()
    return dict(layout=family,rule=rule,perm=perm,y=y,bounds=bounds),dict(
        group_seconds=grouped-start,bound_seconds=bounded-grouped,reorder_seconds=end-bounded)


def rebuild(x,family,rule,model=None):
    start=time.perf_counter()
    _,q,pca,scale2,features=make_features(x)
    mapping_started=time.perf_counter()
    coords=q if family=='pca' else mapped_coordinates(model,q)
    mapped=time.perf_counter()
    plan,parts=from_coordinates(x,q,coords,scale2,family,rule,pca if family=='pca' and rule=='first' else None)
    end=time.perf_counter()
    return plan,dict(seconds=end-start,features=features,mapping_seconds=mapped-mapping_started,
        **parts,rule=rule,solver_steps=model.solver_steps if family=='fm' else 0,
        nfe=2*model.solver_steps if family=='fm' else (1 if family=='direct' else 0),
        permutation_sha256=digest(plan['perm']))


def saved_plans(x):
    selection=json.loads((HERE/'artifacts/model_selection.json').read_text())
    plans={}
    for family in FAMILIES:
        perm=np.load(HERE/'artifacts'/f'{family}_permutation.npy')
        bounds=dict(np.load(HERE/'artifacts'/f'{family}_bounds.npz'))
        rule=selection['pca']['rule'] if family=='pca' else selection['models'][family]['rule']
        plans[family]=dict(layout=family,rule=rule,perm=perm,y=np.ascontiguousarray(x[perm]),bounds=bounds)
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
        prepared,construction=rebuild(x,family,plans[family]['rule'],models.get(family))
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
    record.update(configuration=configuration,family=family,rule=prepared['rule'],
        includes_build=build,includes_training=False,includes_model_mapping=build and family!='pca',
        includes_ode=build and family=='fm',build_peak_allocated_bytes=build_peak)
    record['output_sha256']=retained.output.check_output(output,cell,exact)
    del output
    return record,kernel


def capture_remap(kernel):
    if kernel is None:return None
    stem='remap_'+hashlib.sha256(kernel.asm['cubin']).hexdigest()[:16]
    record={}
    for ext in ['cubin','ptx','llir','ttgir','ttir']:
        data=kernel.asm[ext]
        data=data if isinstance(data,bytes) else data.encode()
        path=HERE/'artifacts'/(stem+'.'+ext)
        expected=hashlib.sha256(data).hexdigest()
        if path.exists():assert sha(path)==expected
        else:path.write_bytes(data)
        record[ext]=expected
    assert kernel.n_spills==0
    return record


def verify_timing_freeze():
    for path,expected in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
        assert sha(path)==expected,path


def mapping_benchmark(q,models):
    rows=[]
    for family,steps in [('direct',0),('fm',8),('fm',16),('fm',32)]:
        model=models[family]
        expected=digest(mapped_coordinates(model,q,steps=steps))
        values=[]
        for _ in range(7):
            torch.cuda.synchronize()
            start=time.perf_counter()
            coords=mapped_coordinates(model,q,steps=steps)
            end=time.perf_counter()
            assert np.isfinite(coords).all() and digest(coords)==expected
            values.append(end-start)
        nfe=2*steps if family=='fm' else 1
        macs=(33 if family=='fm' else 32)*128+128*128+128*32
        rows.append(dict(family=family,device='cuda',solver_steps=steps,nfe=nfe,
            selected=family=='direct' or steps==model.solver_steps,parameters=parameter_count(model),
            host_to_host_seconds=values,median_seconds=float(np.median(values)),
            dense_layer_flops=2*len(q)*macs*nfe,rows=len(q),input_features=32,output_features=32,
            coordinates_sha256=expected,scope='Real full 32D mapping including all FM ODE evaluations, H2D and D2H.'))
    return rows
