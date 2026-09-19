import gc
import torch
from problem import *
from models import configure,load_model,predict,parameter_count
from certificate import minima,certified_reject


def selected_models():
    selection=json.loads((HERE/'artifacts/model_selection.json').read_text())
    models={}
    for family in ['direct','fm']:
        spec=selection['models'][family];path=HERE/'artifacts'/(spec['name']+'.pt')
        assert sha(path)==spec['checkpoint_sha256'] and spec['trained']
        models[family]=load_model(path);models[family].solver_steps=spec['solver_steps']
    return models


def saved_plan(x):
    perm=np.load(HERE/'artifacts/permutation.npy')
    q=np.load(HERE/'artifacts/features_f64.npy')[perm]
    return dict(layout='pca_solution_fm',perm=perm,y=np.ascontiguousarray(x[perm]),
        bounds=dict(np.load(HERE/'artifacts/bounds.npz')),
        features=np.load(HERE/'artifacts/pair_features.npy'),coarse=np.load(HERE/'artifacts/coarse_state.npy'),
        q32=np.ascontiguousarray(q.T,dtype=np.float32),
        prep=json.loads((HERE/'artifacts/preparation.json').read_text()))


def rebuild_plan(x,configuration):
    start=time.perf_counter();_,q,score,scale2,ftime=make_features(x)
    perm=np.argsort(score,kind='stable').astype(np.int64)
    b,btime=projected_bounds(q,perm,scale2)
    feat=coarse=None;stime=0.
    if configuration not in ['pca','all']:feat,coarse,stime=summaries(x,q,perm,b)
    q32=None if configuration=='pca' else np.ascontiguousarray(q[perm].T,dtype=np.float32)
    margin=numerical_margin(q)
    plan=dict(layout='pca_solution_fm',perm=perm,y=np.ascontiguousarray(x[perm]),bounds=b,
        features=feat,coarse=coarse,q32=q32,prep=dict(scale2=scale2,numerical_margin=margin))
    return plan,dict(seconds=time.perf_counter()-start,features=ftime,bound_seconds=btime,summary_seconds=stime,
        includes_oracle=False,includes_training=False,permutation_sha256=digest(perm))


def route(plan,models,cell,configuration,diagnostic=False):
    start=time.perf_counter();b=plan['bounds']
    mask=b['combined']<=cell['T']+SQ_PAD
    assert np.all(mask[b['tr']==b['tc']])
    ids=eligible(b,cell);fraction=BUDGETS[configuration]
    record=dict(configuration=configuration,base_retained_tiles=int(mask.sum()),eligible_tiles=len(ids),budget=fraction,
        prediction_seconds=0.,selection_seconds=0.,certificate_seconds=0.,certificate_tiles=0,
        extra_certified_tiles=0,extra_pruned_pairs=0,certificate_dimensions_sum=0,mean_certificate_dimensions=0.,
        model_parameters=0,nfe=0,solver_steps=0)
    chosen=np.empty(0,np.int64);rejected_ids=chosen;kernel=None
    if configuration!='pca':
        selection_start=time.perf_counter()
        if configuration=='all':chosen=ids
        elif configuration=='random25':
            ci=next(i for i,c in enumerate(CELLS) if c['name']==cell['name'])
            chosen=np.sort(np.random.default_rng(2026090920+ci).permutation(ids)[:int(np.ceil(len(ids)*fraction))])
        elif configuration=='heuristic25':chosen=top_budget(-plan['coarse'][ids],ids,fraction)
        else:
            family='direct' if configuration.startswith('direct') else 'fm';model=models[family]
            inference_start=time.perf_counter()
            feats=plan['features'][ids];initial=initial_state(plan['coarse'][ids],cell['T'])
            scores=predict(model,feats,initial,cell['T'],steps=model.solver_steps)
            assert np.isfinite(scores).all()
            record['prediction_seconds']=time.perf_counter()-inference_start
            record.update(model_parameters=parameter_count(model),solver_steps=model.solver_steps,
                nfe=2*model.solver_steps if family=='fm' else 1)
            chosen=top_budget(scores,ids,fraction)
        record['selection_seconds']=time.perf_counter()-selection_start-record['prediction_seconds']
        certificate_start=time.perf_counter()
        qg=torch.from_numpy(plan['q32']).to('cuda')
        values,dims,kernel=minima(qg,b['tr'][chosen],b['tc'][chosen],cell['T'],
            plan['prep']['numerical_margin']['error'],plan['prep']['scale2'])
        rejected=certified_reject(values,cell['T'],plan['prep']['numerical_margin']['error'],plan['prep']['scale2'])
        rejected_ids=chosen[rejected];mask[rejected_ids]=False
        record.update(certificate_seconds=time.perf_counter()-certificate_start,certificate_tiles=len(chosen),
            extra_certified_tiles=len(rejected_ids),extra_pruned_pairs=int(b['weights'][rejected_ids].sum()),
            certificate_dimensions_sum=int(dims.sum()),mean_certificate_dimensions=float(dims.mean()) if len(dims) else 0.)
        del qg
    assert np.all(mask[b['tr']==b['tc']])
    record['routing_seconds']=time.perf_counter()-start
    # Arrays are returned only for correctness/audit; no oracle is read here.
    return mask,record,kernel,dict(chosen=chosen,rejected=rejected_ids)


def verify_timing_freeze():
    for p,want in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():assert sha(p)==want,p


def mapping_benchmark(plan,models):
    cell=CELLS[3];ids=eligible(plan['bounds'],cell)
    f=plan['features'][ids];z=initial_state(plan['coarse'][ids],cell['T']);rows=[]
    for family,steps in [('direct',0),('fm',2),('fm',4),('fm',8)]:
        model=models[family];expected=digest(predict(model,f,z,cell['T'],steps=steps));samples=[]
        for _ in range(7):
            torch.cuda.synchronize();t=time.perf_counter()
            scores=predict(model,f,z,cell['T'],steps=steps)
            samples.append(time.perf_counter()-t)
            assert np.isfinite(scores).all() and digest(scores)==expected
        nfe=2*steps if family=='fm' else 1
        rows.append(dict(family=family,solver_steps=steps,nfe=nfe,selected=steps==model.solver_steps,
            parameters=parameter_count(model),rows=len(ids),threshold=cell['name'],condition_dimensions=21,
            host_to_host_seconds=samples,median_seconds=float(np.median(samples)),prediction_sha256=expected,
            dense_layer_flops=2*len(ids)*(21*64+nfe*(64*64+64)),
            scope='Prepared host condition features and initial state to host endpoint; includes normalization, H2D, actual ODE, D2H. State/time-independent first affine term cached per chunk.'))
    return rows
