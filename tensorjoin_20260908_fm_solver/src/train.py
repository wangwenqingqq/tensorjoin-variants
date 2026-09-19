"""Supervised, data-initialized conditional flow in scalar tile-solution state."""
import gc
import os
import traceback
import torch
from problem import *
from models import Solver,configure,initialize_matched,parameter_count,predict
from certificate import minima,certified_reject
from preflight import capture

SEEDS=[2026090911,2026090912,2026090913]
CHECKPOINTS=[750,1500,3000]
STEPS=[2,4,8]
BATCH=1024


def prediction_metrics(z,counts):
    occupied=counts>0;prediction=z>=0
    missed=occupied & ~prediction
    entries=int(counts.sum())
    return dict(tiles=len(z),occupied_tiles=int(occupied.sum()),empty_tiles=int((~occupied).sum()),
        endpoint_mse=float(np.mean((z-(2*occupied.astype(np.float32)-1))**2)),
        classification_accuracy=float(np.mean(prediction==occupied)),
        false_negative_tiles=int(missed.sum()),false_positive_tiles=int(np.sum(prediction & ~occupied)),
        missed_reference_entries=int(counts[missed].sum()),reference_entries=entries,
        candidate_entry_recall=float(1-counts[missed].sum()/entries) if entries else 1.,
        note='Raw unverified block-candidate predictions; no such prediction can reject a tile in the safe solver.')


def evaluate(model,features,coarse,counts,b,subset,cert_min,prep,steps):
    rows=[];scores=[]
    denominator=float(b['weights'][subset].sum())
    for ci,cell in enumerate(CELLS):
        ids=eligible(b,cell,subset)
        z=predict(model,features[ids],initial_state(coarse[ids],cell['T']),cell['T'],steps=steps)
        assert np.isfinite(z).all()
        chosen=top_budget(z,ids,.25)
        reject=certified_reject(cert_min[chosen],cell['T'],prep['numerical_margin']['error'],prep['scale2'])
        rejected_ids=chosen[reject]
        assert np.all(counts[ci,rejected_ids]==0)
        score=float(b['weights'][rejected_ids].sum()/denominator)
        row=dict(cell=cell['name'],checked_tiles=len(chosen),extra_certified_tiles=len(rejected_ids),
            extra_pair_pruning=score,**prediction_metrics(z,counts[ci,ids]))
        rows.append(row);scores.append(score)
    return dict(mean_extra_pair_pruning=float(np.mean(scores)),mean_endpoint_mse=float(np.mean([r['endpoint_mse'] for r in rows])),cells=rows)


@torch.no_grad()
def fixed_loss(model,features,coarse,counts,b,subset):
    rng=np.random.default_rng(2026090914);values=[]
    for ci,cell in enumerate(CELLS):
        ids=rng.choice(eligible(b,cell,subset),256,replace=False)
        f=torch.from_numpy(features[ids]).to('cuda')
        a=torch.from_numpy(initial_state(coarse[ids],cell['T'])).to('cuda')
        y=torch.from_numpy((counts[ci,ids]>0).astype(np.float32)*2-1).to('cuda')
        c=model.conditions(f,cell['T'])
        if model.family=='fm':
            t=torch.from_numpy(rng.uniform(0,1,len(ids)).astype(np.float32)).to('cuda')
            p=model(c,(1-t)*a+t*y,t);target=y-a
        else:p=model(c,a);target=y
        values.append(float((p-target).square().mean().item()))
    return float(np.mean(values))


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4';configure()
    assert json.loads((HERE/'artifacts/preflight.json').read_text())['pass_']
    paths=[HERE/'PROTOCOL.md',HERE/'METHODS.md',HERE/'CERTIFICATE_SCOPE.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
        *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),*sorted((HERE/'inherited').glob('*')),
        *sorted((OLD/'src').glob('*.py')),SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json']
    save_json(HERE/'artifacts/training_freeze.json',dict(created=time.time(),files={str(p):sha(p) for p in paths if p.is_file()}))
    started=time.time()
    prep=json.loads((HERE/'artifacts/preparation.json').read_text())
    norm=json.loads((HERE/'artifacts/normalizer.json').read_text())
    features=np.load(HERE/'artifacts/pair_features.npy');coarse=np.load(HERE/'artifacts/coarse_state.npy')
    counts=np.load(HERE/'artifacts/counts_train_validation.npy');b=dict(np.load(HERE/'artifacts/bounds.npz'))
    splits=dict(np.load(HERE/'artifacts/pair_splits.npz'))
    validation=splits['validation'];training=splits['train']
    assert not np.any(validation & training) and np.all(counts[:,~(validation|training)]==-1)
    certificate_start=time.perf_counter()
    q=np.load(HERE/'artifacts/features_f64.npy')[np.load(HERE/'artifacts/permutation.npy')]
    qg=torch.from_numpy(np.ascontiguousarray(q.T,dtype=np.float32)).to('cuda')
    ids=np.flatnonzero(validation & (b['tr']!=b['tc']))
    values,_,kernel=minima(qg,b['tr'][ids],b['tc'][ids])
    cert_min=np.full(len(coarse),np.nan,np.float32);cert_min[ids]=values
    capture(kernel)
    for ci,cell in enumerate(CELLS):
        reject=certified_reject(values,cell['T'],prep['numerical_margin']['error'],prep['scale2'])
        assert np.all(counts[ci,ids[reject]]==0)
    np.save(HERE/'artifacts/validation_certificate_minima.npy',cert_min,allow_pickle=False)
    validation_certificate_seconds=time.perf_counter()-certificate_start
    del qg,q
    feature_gpu=torch.from_numpy(features).to('cuda')
    target_gpu=torch.from_numpy((counts>0).astype(np.float32)*2-1).to('cuda')
    states_gpu=torch.from_numpy(np.asarray([initial_state(coarse,c['T']) for c in CELLS])).to('cuda')
    pools=[torch.from_numpy(eligible(b,c,training)).to('cuda') for c in CELLS]
    result=dict(started=started,validation_certificate_seconds=validation_certificate_seconds,
        models=[],checkpoints=[],candidates=[],seeds=[],pass_=False,
        labels_scope='Exact archived occupancy on disjoint training blocks only; held validation blocks select models. Previous reports reused the same dataset.',
        no_gaussian_source=True,solution_state='-1 empty / +1 occupied tile; input-derived coarse starting state')
    for seed in SEEDS:
        seed_start=time.perf_counter();torch.manual_seed(seed)
        models={};initialization={};optimizers={}
        for family in ['direct','fm']:
            t0=time.perf_counter();models[family]=Solver(family,norm).to('cuda')
            optimizers[family]=torch.optim.Adam(models[family].parameters(),lr=.001,foreach=False)
            torch.cuda.synchronize();initialization[family]=time.perf_counter()-t0
        initialize_matched(models['direct'],models['fm'])
        initial={f:{n:p.detach().clone() for n,p in m.named_parameters()} for f,m in models.items()}
        initial_losses={f:fixed_loss(m,features,coarse,counts,b,validation) for f,m in models.items()}
        update_seconds={f:0. for f in models};evaluation_seconds={f:0. for f in models}
        losses={f:[] for f in models};history={f:[] for f in models};batch_seconds=0.
        generator=torch.Generator(device='cuda').manual_seed(seed)
        timegen=torch.Generator(device='cuda').manual_seed(seed+100000)
        for step in range(1,3001):
            torch.cuda.synchronize();batch_start=time.perf_counter()
            ci=(step+seed)%len(CELLS);cell=CELLS[ci]
            pool=pools[ci];chosen=pool[torch.randint(len(pool),(BATCH,),device='cuda',generator=generator)]
            f=feature_gpu[chosen];a=states_gpu[ci,chosen];y=target_gpu[ci,chosen]
            torch.cuda.synchronize();batch_seconds+=time.perf_counter()-batch_start
            for family in (['direct','fm'] if step%2 else ['fm','direct']):
                model=models[family];optimizer=optimizers[family]
                torch.cuda.synchronize();updated=time.perf_counter()
                optimizer.zero_grad(set_to_none=True);c=model.conditions(f,cell['T'])
                if family=='fm':
                    t=torch.rand(len(chosen),device='cuda',generator=timegen)
                    prediction=model(c,(1-t)*a+t*y,t);target=y-a
                else:prediction=model(c,a);target=y
                loss=(prediction-target).square().mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                torch.cuda.synchronize();update_seconds[family]+=time.perf_counter()-updated
                value=float(loss.item());assert np.isfinite(value)
                losses[family].append(value)
                if step%100==0:history[family].append(dict(step=step,loss=float(np.mean(losses[family][-100:]))))
            if step in CHECKPOINTS:
                for family,model in models.items():
                    eval_start=time.perf_counter();model.eval();name=f'{family}_s{seed}_t{step}'
                    path=HERE/'artifacts'/(name+'.pt')
                    torch.save(dict(family=family,seed=seed,step=step,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)
                    changes={n:float(torch.linalg.vector_norm(p-initial[family][n]).item()) for n,p in model.named_parameters()}
                    assert all(v>0 for v in changes.values())
                    checkpoint=dict(name=name,family=family,seed=seed,step=step,parameters=parameter_count(model),trained=True,
                        checkpoint_sha256=sha(path),weight_changes=changes,initial_validation_loss=initial_losses[family],
                        validation_loss=fixed_loss(model,features,coarse,counts,b,validation),
                        update_seconds=update_seconds[family],batch_seconds=batch_seconds,initialization_seconds=initialization[family],
                        training_seconds=update_seconds[family]+batch_seconds)
                    if family=='fm':checkpoint['time_weight_norm']=float(model.time_weight.norm().item())
                    result['checkpoints'].append(checkpoint)
                    for steps in (STEPS if family=='fm' else [0]):
                        evaluation=evaluate(model,features,coarse,counts,b,validation,cert_min,prep,steps)
                        candidate=dict(**{k:checkpoint[k] for k in ['name','family','seed','step','parameters','trained','checkpoint_sha256','training_seconds','initialization_seconds']},
                            solver_steps=steps,nfe=2*steps if family=='fm' else 1,validation=evaluation)
                        result['candidates'].append(candidate)
                    evaluation_seconds[family]+=time.perf_counter()-eval_start;model.train()
                    print(json.dumps(dict(checkpoint=checkpoint,best_extra_pruning=max(c['validation']['mean_extra_pair_pruning'] for c in result['candidates'] if c['name']==name))),flush=True)
        for family in models:
            result['models'].append(dict(family=family,seed=seed,history=history[family],update_seconds=update_seconds[family],
                shared_batch_seconds=batch_seconds,training_seconds=update_seconds[family]+batch_seconds,
                initialization_seconds=initialization[family],evaluation_seconds=evaluation_seconds[family]))
        result['seeds'].append(dict(seed=seed,total_seconds=time.perf_counter()-seed_start,shared_batch_seconds=batch_seconds,
            updates_per_model=3000,batch_size=BATCH,model_updates=update_seconds,model_evaluations=evaluation_seconds))
        del models,model,optimizers,optimizer,initial;gc.collect();torch.cuda.empty_cache()
    selected={}
    for family in ['direct','fm']:
        options=[c for c in result['candidates'] if c['family']==family]
        selected[family]=sorted(options,key=lambda c:(-c['validation']['mean_extra_pair_pruning'],c['solver_steps'],c['step'],c['seed']))[0]
    selection=dict(created=time.time(),models=selected,candidate_counts=dict(direct=9,fm=27),
        rule='Highest validation mean extra certified pair pruning when checking 25% of coarse survivors; then fewer ODE steps, fewer updates, lower seed. No query timings, test labels or full-data certificate outcomes used.')
    save_json(HERE/'artifacts/model_selection.json',selection)
    result.update(selection=selection,ended=time.time(),pass_=True)
    assert len(result['checkpoints'])==18 and len(result['candidates'])==36
    save_json(HERE/'results/training.json',result)
    print(json.dumps(dict(training_complete=True,selection=selection)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results/training_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
