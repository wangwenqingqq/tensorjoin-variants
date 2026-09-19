import os
import traceback
from engine import *
from preflight import capture
from train import prediction_metrics
from models import heun


@torch.no_grad()
def flow_diagnostics(model,plan):
    cell=CELLS[3];pool=eligible(plan['bounds'],cell)
    ids=np.random.default_rng(2026090928).choice(pool,512,replace=False)
    f=torch.from_numpy(plan['features'][ids]).to('cuda')
    z=torch.from_numpy(initial_state(plan['coarse'][ids],cell['T'])).to('cuda')
    embedded=model.embedding(model.conditions(f,cell['T']));field=lambda zz,t:model.field(embedded,zz,t)
    endpoints={steps:heun(field,z,steps) for steps in [2,4,8,16]}
    ref=endpoints[16];denom=max(float(ref.square().mean().item()),1e-12)
    rows=[]
    for steps,end in endpoints.items():
        back=heun(field,end,steps,start=1.,end=0.)
        rows.append(dict(steps=steps,nfe=2*steps,relative_rms_vs_16=float(torch.sqrt((end-ref).square().mean()/denom).item()),
            reverse_rms=float(torch.sqrt((back-z).square().mean()).item()),mean_squared_displacement=float((end-z).square().mean().item())))
    endpoint,trajectory=heun(field,z,model.solver_steps,trajectory=True)
    effect=float(torch.sqrt((field(z,.25)-field(z,.75)).square().mean()).item())
    assert np.isfinite(effect) and effect>0 and float((endpoint-z).square().mean())>0
    np.savez_compressed(HERE/'artifacts/solution_trajectory.npz',tile_ids=ids,
        times=np.linspace(0,1,model.solver_steps+1),states=trajectory.cpu().numpy())
    return dict(sample_tiles=512,selected_steps=model.solver_steps,time_dependence_rms=effect,solvers=rows,
        note='16-step endpoint is a numerical comparison, not an exact solution. This check never reselects the model.')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4';configure()
    training=json.loads((HERE/'results/training.json').read_text());assert training['pass_']
    for p,want in json.loads((HERE/'artifacts/training_freeze.json').read_text())['files'].items():assert sha(p)==want,p
    x=source();plan=saved_plan(x);models=selected_models();b=plan['bounds']
    result=dict(started=time.time(),pass_=False,raw_prediction_metrics=[],geometry=[],samples=[],diagnostics=[],build_admission=[])
    # First access to this round's held-out block-pair labels happens after selection.
    counts=exact_counts(plan['perm']);np.save(HERE/'artifacts/counts_full.npy',counts,allow_pickle=False)
    splits=dict(np.load(HERE/'artifacts/pair_splits.npz'))
    masks={}
    for name,subset in [('validation',splits['validation']),('test',splits['test']),('full',np.ones(len(b['tr']),bool))]:
        for ci,cell in enumerate(CELLS):
            ids=eligible(b,cell,subset);initial=initial_state(plan['coarse'][ids],cell['T'])
            for family in ['initial','direct','fm']:
                scores=initial if family=='initial' else predict(models[family],plan['features'][ids],initial,cell['T'],steps=models[family].solver_steps)
                row=dict(split=name,family=family,cell=cell['name'],**prediction_metrics(scores,counts[ci,ids]))
                result['raw_prediction_metrics'].append(row)
                if name=='full' and family!='initial':
                    a=np.full(len(b['tr']),np.nan,np.float32);a[ids]=scores
                    np.save(HERE/'artifacts'/f'prediction_{family}_{cell["name"]}.npy',a,allow_pickle=False)
            if name=='full':
                for config in CONFIGS:
                    mask,rec,kernel,audit=route(plan,models,cell,config)
                    assert np.all(counts[ci,~mask]==0),(config,cell['name'])
                    rec.update(cell=cell['name'],retained_tiles=int(mask.sum()),total_pair_pruning=float(b['weights'][~mask].sum()/b['weights'].sum()),
                        ideal_empty_tiles=int(np.sum(counts[ci]==0)),reference_occupancy_pass=True,mask_sha256=digest(mask),
                        certificate_selection_sha256=digest(audit['chosen']))
                    result['geometry'].append(rec);masks[config+'_'+cell['name']]=mask
                    capture(kernel)
    np.savez_compressed(HERE/'artifacts/admitted_masks.npz',**masks)
    result['flow_diagnostics']=flow_diagnostics(models['fm'],plan)
    result['mapping']=mapping_benchmark(plan,models)
    op,radix=retained.output.Matrix(),retained.output.Radix()
    for cell in CELLS:
        for method in METHODS:
            for config in CONFIGS:
                rec,rk,ck,mask=run(op,radix,x,cell,method,config,plan,models,exact=True)
                assert np.array_equal(mask,masks[config+'_'+cell['name']])
                capture_remap(rk);capture(ck);result['samples'].append(rec)
                print(json.dumps(rec),flush=True)
            rec,rk,ck,mask=run(op,radix,x,cell,method,'fm25',plan,models,exact=True,diagnostic=True)
            capture_remap(rk);capture(ck);result['diagnostics'].append(rec)
        op.capture();gc.collect();torch.cuda.empty_cache()
    for config in ['pca','all','heuristic25','direct25','fm25']:
        rec,rk,ck,mask=run(op,radix,x,CELLS[0],'F8',config,plan,models,exact=True,build=True)
        assert np.array_equal(mask,masks[config+'_'+CELLS[0]['name']])
        capture_remap(rk);capture(ck);result['build_admission'].append(rec)
    result.update(pass_=True,ended=time.time(),compiled=op.capture(),cub_version=radix.version)
    result['pass']=True;save_json(HERE/'artifacts/admission.json',result)
    paths=[HERE/'PROTOCOL.md',HERE/'METHODS.md',HERE/'CERTIFICATE_SCOPE.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
        *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),*sorted((HERE/'inherited').glob('*')),
        *sorted((OLD/'src').glob('*.py')),*sorted((retained.SHARED/'src').glob('*.py')),
        retained.SHARED/'artifacts/output.so',retained.SHARED/'artifacts/build.json',
        SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json',
        *sorted((retained.output.OLD/'src').glob('*.py')),*sorted(retained.output.OLD.glob('*manifest.json'))]
    freeze={str(p):sha(p) for p in paths if p.is_file()}
    freeze.update({str(ROOT/p):expected for p,expected in op.identities.items()})
    save_json(HERE/'artifacts/timing_freeze.json',freeze)
    print(json.dumps(dict(admission_complete=True,normal_calls=len(result['samples']),all_admission_calls=113,frozen_files=len(freeze))),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results'/f'admission_failed_{int(time.time())}.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
