import os
import traceback
from benchmark import *
from models import Transport,heun


@torch.no_grad()
def flow_diagnostics(model,q):
    ids=np.random.default_rng(2026090837).choice(len(q),1024,replace=False)
    start=model.normalize(torch.from_numpy(q[ids].astype(np.float32)).to('cuda'))
    endpoints={steps:heun(model,start,steps) for steps in [8,16,32,64]}
    reference=endpoints[64]
    denom=max(float(reference.square().mean().item()),1e-12)
    initial_denom=max(float(start.square().mean().item()),1e-12)
    rows=[]
    for steps in [8,16,32,64]:
        endpoint=endpoints[steps]
        back=heun(model,endpoint,steps,start=1.,end=0.)
        rows.append(dict(steps=steps,nfe=2*steps,
            endpoint_relative_rms_vs_64=float(torch.sqrt((endpoint-reference).square().mean()/denom).item()),
            reverse_relative_rms=float(torch.sqrt((back-start).square().mean()/initial_denom).item()),
            mean_squared_displacement=float((endpoint-start).square().sum(1).mean().item())))
    endpoint,trajectory=heun(model,start,model.solver_steps,trajectory=True)
    assert torch.equal(endpoint,endpoints[model.solver_steps])
    np.savez_compressed(HERE/'artifacts/flow_trajectory.npz',ids=ids,
        times=np.linspace(0,1,model.solver_steps+1),states=trajectory.cpu().numpy())
    time_effect=float(torch.sqrt((model(start,.25)-model(start,.75)).square().mean()).item())
    assert np.isfinite(time_effect) and float((endpoint-start).square().mean())>0
    return dict(sample_rows=1024,selected_steps=model.solver_steps,solvers=rows,
        time_dependence_rms=time_effect,
        note='64 steps is a numerical comparison, not an exact ODE reference; no model or solver is reselected.')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    configure()
    training=json.loads((HERE/'results/training.json').read_text())
    assert training['pass_']
    for path,expected in json.loads((HERE/'artifacts/training_freeze.json').read_text())['files'].items():
        assert sha(path)==expected,path
    result=dict(started=time.time(),samples=[],diagnostics=[],geometry={},pass_=False)
    x=source()
    q=np.load(HERE/'artifacts/features_f64.npy')
    pca=np.load(HERE/'artifacts/pca_scores_f64.npy')
    scale2=json.loads((HERE/'artifacts/features.json').read_text())['scale2']
    split=dict(np.load(HERE/'artifacts/splits.npz'))
    selection=training['selection']
    models=selected_models()
    coordinates={'pca':q}
    for family in ['direct','fm']:
        coordinates[family]=mapped_coordinates(models[family],q)
        assert np.isfinite(coordinates[family]).all()
    normalizer=json.loads((HERE/'artifacts/normalizer.json').read_text())
    zero=Transport('fm',normalizer['mean'],normalizer['scale'],normalizer['target_std']).to('cuda').eval()
    zero_coords=mapped_coordinates(zero,q[split['validation']],steps=8)
    zero_geometry,_,_=layout_geometry(q[split['validation']],zero_coords,scale2,'first')
    first_baseline=next(c for c in training['baseline_candidates'] if c['rule']=='first')
    assert abs(zero_geometry['mean_pair_pruning']-first_baseline['validation']['mean_pair_pruning'])<1e-4
    result['zero_velocity_control']=zero_geometry
    del zero
    result['flow_diagnostics']=flow_diagnostics(models['fm'],q)
    directions=np.load(HERE/'artifacts/sliced_directions.npy')
    plans={}
    for family in FAMILIES:
        rule=selection['pca']['rule'] if family=='pca' else selection['models'][family]['rule']
        result['geometry'][family]={}
        for name,ids in [('test',split['test']),('full',np.arange(len(x)))]:
            geo,perm,bounds=layout_geometry(q[ids],coordinates[family][ids],scale2,rule,
                pca[ids] if family=='pca' and rule=='first' else None)
            checks={cell['name']:reference_occupancy_check(perm,bounds,cell,None if name=='full' else ids)
                    for cell in CELLS}
            geo['reference_checks']=checks
            qq=q[ids][perm]
            widths=np.asarray([np.ptp(qq[i:i+64],axis=0) for i in range(0,len(qq),64)])
            geo['mean_pc1_block_width']=float(widths[:,0].mean())
            geo['mean_block_bbox_diagonal']=float(np.linalg.norm(widths,axis=1).mean())
            baseline_perm=permutation(q[ids],selection['pca']['rule'],
                pca[ids] if selection['pca']['rule']=='first' else None)
            baseline_rank=np.empty(len(ids),np.int64);baseline_rank[baseline_perm]=np.arange(len(ids))
            current_rank=np.empty(len(ids),np.int64);current_rank[perm]=np.arange(len(ids))
            geo['permutation_equals_pca']=bool(np.array_equal(perm,baseline_perm))
            geo['same_block_assignment_fraction']=float(np.mean(baseline_rank//64==current_rank//64))
            geo['mean_absolute_rank_change']=float(np.abs(baseline_rank-current_rank).mean())
            with torch.no_grad():
                normalized=models['fm'].normalize(torch.from_numpy(q[ids].astype(np.float32)).to('cuda')).cpu().numpy()
            target=np.random.default_rng(2026090840+len(ids)).normal(0,normalizer['target_std'],size=(len(ids),32))
            target_sorted=np.sort(target@directions,axis=0)
            dist_coords=normalized if family=='pca' else coordinates[family][ids]
            geo['distribution']=distribution_diagnostics(dist_coords,normalized,target_sorted,directions,normalizer['target_std'])
            result['geometry'][family][name]=geo
            print(json.dumps(dict(family=family,split=name,geometry=geo)),flush=True)
            if name=='full':
                np.save(HERE/'artifacts'/f'{family}_permutation.npy',perm,allow_pickle=False)
                np.savez_compressed(HERE/'artifacts'/f'{family}_bounds.npz',**bounds)
                np.save(HERE/'artifacts'/f'{family}_coordinates.npy',coordinates[family],allow_pickle=False)
                plans[family]=dict(layout=family,rule=rule,perm=perm,y=np.ascontiguousarray(x[perm]),bounds=bounds)
    result['mapping']=mapping_benchmark(q,models)
    op,radix=retained.output.Matrix(),retained.output.Radix()
    for cell in CELLS:
        for method in METHODS:
            for config in CONFIGS:
                rec,kernel=run(op,radix,x,cell,method,config,plans,models,exact=True)
                capture_remap(kernel)
                result['samples'].append(rec)
                print(json.dumps(rec),flush=True)
            rec,kernel=run(op,radix,x,cell,method,'fm_geo',plans,models,exact=True,diagnostic=True)
            result['diagnostics'].append(rec)
            capture_remap(kernel)
        op.capture()
        gc.collect()
        torch.cuda.empty_cache()
    result['build_admission']=[]
    for family in FAMILIES:
        rec,kernel=run(op,radix,x,CELLS[0],'F8',family+'_geo',plans,models,exact=True,build=True)
        capture_remap(kernel)
        result['build_admission'].append(rec)
    result.update(pass_=True,ended=time.time(),compiled=op.capture(),cub_version=radix.version)
    result['pass']=True
    save_json(HERE/'artifacts/admission.json',result)
    paths=[HERE/'PROTOCOL.md',HERE/'METHODS.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
        *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),
        *sorted((OLD/'src').glob('*.py')),*sorted((retained.SHARED/'src').glob('*.py')),
        retained.SHARED/'artifacts/output.so',retained.SHARED/'artifacts/build.json',
        SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json',
        *sorted((retained.output.OLD/'src').glob('*.py')),*sorted(retained.output.OLD.glob('*manifest.json'))]
    freeze={str(path):sha(path) for path in paths if path.is_file()}
    freeze.update({str(ROOT/path):expected for path,expected in op.identities.items()})
    save_json(HERE/'artifacts/timing_freeze.json',freeze)
    print(json.dumps(dict(admission_complete=True,configurations=len(result['samples']),frozen_files=len(freeze))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results/admission_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
