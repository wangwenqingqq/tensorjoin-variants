"""Matched one-pass regression and actual minibatch OT-CFM training."""
import gc
import os
import traceback
import torch
from scipy.optimize import linear_sum_assignment
from grouping import *
from models import Transport,configure,initialize_matched,parameter_count,mapped_coordinates

SEEDS=[2026090831,2026090832,2026090833]
CHECKPOINTS=[1000,2000,4000]
SOLVER_STEPS=[8,16,32]
BATCH=256


@torch.no_grad()
def coupled_batch(source,target_std,generator):
    ids=torch.randint(0,len(source),(BATCH,),device='cuda',generator=generator)
    u=source[ids]
    g=torch.randn((BATCH,32),device='cuda',generator=generator)*target_std
    cost=(u.square().sum(1)[:,None]+g.square().sum(1)[None,:]-2*(u@g.T)).clamp_min(0)
    cpu_cost=cost.cpu().numpy()
    rows,cols=linear_sum_assignment(cpu_cost)
    assert np.array_equal(rows,np.arange(BATCH)) and len(np.unique(cols))==BATCH
    assert float(cpu_cost[rows,cols].sum())<=float(np.trace(cpu_cost))+1e-3
    gg=g[torch.from_numpy(cols).to('cuda')]
    return u,gg,float(cpu_cost[rows,cols].mean()),ids


@torch.no_grad()
def fixed_validation_pairs(source,target_std):
    generator=torch.Generator(device='cuda').manual_seed(2026090834)
    timegen=torch.Generator(device='cuda').manual_seed(2026091834)
    pairs=[coupled_batch(source,target_std,generator) for _ in range(8)]
    return torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs]),\
        torch.rand((8*BATCH,1),device='cuda',generator=timegen)


@torch.no_grad()
def regression_loss(model,pairs):
    u,g,t=pairs
    if model.family=='fm':
        prediction=model((1-t)*u+t*g,t)
        target=g-u
    else:
        prediction=model(u)
        target=g
    return float((prediction-target).square().mean().div(model.target_std.square()).item())


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    configure()
    assert json.loads((HERE/'artifacts/preflight.json').read_text())['pass_']
    paths=[HERE/'PROTOCOL.md',HERE/'METHODS.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
        *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),
        *sorted((HERE/'inherited').glob('*')),*sorted((OLD/'src').glob('*.py')),
        SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json']
    save_json(HERE/'artifacts/training_freeze.json',dict(created=time.time(),
        files={str(p):sha(p) for p in paths if p.is_file()}))
    q=np.load(HERE/'artifacts/features_f64.npy')
    pca=np.load(HERE/'artifacts/pca_scores_f64.npy')
    splits=dict(np.load(HERE/'artifacts/splits.npz'))
    scale2=json.loads((HERE/'artifacts/features.json').read_text())['scale2']
    normalizer=json.loads((HERE/'artifacts/normalizer.json').read_text())
    mean=np.asarray(normalizer['mean'],np.float32)
    scale=np.float32(normalizer['scale'])
    target_std=np.float32(normalizer['target_std'])
    train_rows,valid_rows=splits['train'],splits['validation']
    train_tensor=torch.from_numpy(q[train_rows].astype(np.float32)).to('cuda')
    valid_tensor=torch.from_numpy(q[valid_rows].astype(np.float32)).to('cuda')
    mean_tensor=torch.from_numpy(mean).to('cuda')
    scale_tensor=torch.as_tensor(scale,device='cuda')
    train_source=(train_tensor-mean_tensor)/scale_tensor
    valid_source=(valid_tensor-mean_tensor)/scale_tensor
    validation_pairs=fixed_validation_pairs(valid_source,float(target_std))
    np.savez_compressed(HERE/'artifacts/fixed_validation_pairs.npz',
        source=validation_pairs[0].cpu().numpy(),target=validation_pairs[1].cpu().numpy(),
        time=validation_pairs[2].cpu().numpy())
    directions=np.load(HERE/'artifacts/sliced_directions.npy')
    target=np.load(HERE/'artifacts/gaussian_validation_target.npy')
    target_sorted=np.sort(target@directions,axis=0)
    source_np=valid_source.cpu().numpy().astype(np.float64)
    gaussianization_baseline=distribution_diagnostics(source_np,source_np,target_sorted,directions,float(target_std))
    second_target=np.random.default_rng(2026090839).normal(0,float(target_std),size=(10000,32))
    gaussianization_baseline['gaussian_sampling_floor']=distribution_diagnostics(
        second_target,source_np,target_sorted,directions,float(target_std))['sliced_w2_squared_normalized']
    baselines=[]
    for rule in RULES:
        geo,_,_=layout_geometry(q[valid_rows],q[valid_rows],scale2,rule,pca[valid_rows] if rule=='first' else None)
        baselines.append(dict(family='pca',rule=rule,validation=geo))
    pca_selected=sorted(baselines,key=lambda r:(-r['validation']['mean_pair_pruning'],RULES.index(r['rule'])))[0]
    result=dict(started=time.time(),normalizer=normalizer,baseline_candidates=baselines,
        pca_selection=pca_selected,gaussianization_baseline=gaussianization_baseline,
        candidates=[],checkpoints=[],models=[],seeds=[],
        semantics='Data-to-isotropic-Gaussian minibatch OT-CFM versus matched one-pass endpoint regression.')
    print(json.dumps(dict(baselines=baselines,normalizer=normalizer,
                         gaussianization_baseline=gaussianization_baseline)),flush=True)
    for seed in SEEDS:
        seed_started=time.perf_counter()
        torch.manual_seed(seed)
        models={}
        initialization={}
        for family in ['direct','fm']:
            t0=time.perf_counter()
            models[family]=Transport(family,mean,scale,target_std).to('cuda')
            torch.cuda.synchronize()
            initialization[family]=time.perf_counter()-t0
        initialize_matched(models['direct'],models['fm'])
        assert parameter_count(models['direct'])==24864 and parameter_count(models['fm'])==24992
        initial={family:{name:value.detach().clone() for name,value in model.named_parameters()}
                 for family,model in models.items()}
        optimizers={}
        for family,model in models.items():
            initialized_at=time.perf_counter()
            optimizers[family]=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
            torch.cuda.synchronize()
            initialization[family]+=time.perf_counter()-initialized_at
        generator=torch.Generator(device='cuda').manual_seed(seed)
        timegen=torch.Generator(device='cuda').manual_seed(seed+1000000)
        initial_losses={family:regression_loss(model,validation_pairs) for family,model in models.items()}
        updates={family:0. for family in models}
        losses={family:[] for family in models}
        history={family:[] for family in models}
        coupling_seconds=0.
        evaluation_seconds=0.
        evaluation_by_family={family:0. for family in models}
        assignment_cost_sum=0.
        training_started=time.perf_counter()
        for step in range(1,4001):
            torch.cuda.synchronize()
            coupled_at=time.perf_counter()
            u,g,assignment_cost,_=coupled_batch(train_source,float(target_std),generator)
            torch.cuda.synchronize()
            coupling_seconds+=time.perf_counter()-coupled_at
            assignment_cost_sum+=assignment_cost
            for family in (['direct','fm'] if step%2 else ['fm','direct']):
                model=models[family]
                optimizer=optimizers[family]
                torch.cuda.synchronize()
                updated_at=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if family=='fm':
                    t=torch.rand((BATCH,1),device='cuda',generator=timegen)
                    prediction=model((1-t)*u+t*g,t)
                    target_velocity=g-u
                    loss=(prediction-target_velocity).square().mean()/float(target_std**2)
                else:
                    loss=(model(u)-g).square().mean()/float(target_std**2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
                torch.cuda.synchronize()
                updates[family]+=time.perf_counter()-updated_at
                value=float(loss.item())
                assert np.isfinite(value),(seed,step,family,value)
                losses[family].append(value)
                if step%100==0:
                    history[family].append(dict(step=step,loss=float(np.mean(losses[family][-100:])),
                        update_seconds=updates[family],coupling_seconds=coupling_seconds))
            if step in CHECKPOINTS:
                evaluation_started=time.perf_counter()
                for family,model in models.items():
                    family_evaluation_started=time.perf_counter()
                    model.eval()
                    name=f'{family}_s{seed}_t{step}'
                    path=HERE/'artifacts'/(name+'.pt')
                    torch.save(dict(family=family,seed=seed,step=step,
                        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)
                    weight_changes={name:float(torch.linalg.vector_norm(value-initial[family][name]).item())
                        for name,value in model.named_parameters()}
                    assert all(v>0 for v in weight_changes.values())
                    trained=dict(name=name,family=family,seed=seed,step=step,
                        parameters=parameter_count(model),initialization_seconds=initialization[family],
                        update_seconds=updates[family],coupling_seconds=coupling_seconds,
                        training_seconds=updates[family]+coupling_seconds,
                        checkpoint_sha256=sha(path),initial_validation_loss=initial_losses[family],
                        validation_loss=regression_loss(model,validation_pairs),
                        weight_changes=weight_changes,distribution_by_steps={},trained=True)
                    if family=='fm':
                        trained['time_input_weight_norm']=float(torch.linalg.vector_norm(model.net[0].weight[:,32]).item())
                    for steps in (SOLVER_STEPS if family=='fm' else [0]):
                        coords=mapped_coordinates(model,q[valid_rows],steps=steps)
                        assert np.isfinite(coords).all()
                        distribution=distribution_diagnostics(coords,source_np,target_sorted,directions,float(target_std))
                        trained['distribution_by_steps'][str(steps)]=distribution
                        for rule in RULES:
                            geo,_,_=layout_geometry(q[valid_rows],coords,scale2,rule)
                            candidate=dict(name=name,family=family,seed=seed,step=step,rule=rule,
                                solver_steps=steps,nfe=2*steps if family=='fm' else 1,
                                parameters=parameter_count(model),validation=geo,distribution=distribution,
                                training_seconds=trained['training_seconds'],
                                update_seconds=updates[family],coupling_seconds=coupling_seconds,
                                initialization_seconds=initialization[family],
                                checkpoint_sha256=trained['checkpoint_sha256'],trained=True)
                            result['candidates'].append(candidate)
                    result['checkpoints'].append(trained)
                    print(json.dumps(dict(checkpoint=trained,
                        best_validation_pruning=max(c['validation']['mean_pair_pruning']
                            for c in result['candidates'] if c['name']==name))),flush=True)
                    model.train()
                    evaluation_by_family[family]+=time.perf_counter()-family_evaluation_started
                evaluation_seconds+=time.perf_counter()-evaluation_started
        joint_training_wall=time.perf_counter()-training_started-evaluation_seconds
        for family in ['direct','fm']:
            result['models'].append(dict(family=family,seed=seed,history=history[family],
                initial_validation_loss=initial_losses[family],initialization_seconds=initialization[family],
                evaluation_seconds=evaluation_by_family[family],
                update_seconds=updates[family],coupling_seconds=coupling_seconds,
                training_seconds=updates[family]+coupling_seconds))
        result['seeds'].append(dict(seed=seed,updates_per_model=4000,batch_size=BATCH,
            shared_pairing_seconds=coupling_seconds,model_update_seconds=updates,
            joint_training_wall_seconds=joint_training_wall,evaluation_seconds=evaluation_seconds,
            total_seconds=time.perf_counter()-seed_started,assignment_cost_mean=assignment_cost_sum/4000,
            paired_distance_evaluations=4000*BATCH*BATCH,matched_training_pairs=4000*BATCH))
        del models,optimizers,initial,model,optimizer
        gc.collect()
        torch.cuda.empty_cache()
    selected={}
    for family in ['direct','fm']:
        options=[c for c in result['candidates'] if c['family']==family]
        selected[family]=sorted(options,key=lambda c:(-c['validation']['mean_pair_pruning'],
            c['step'],c['seed'],c['solver_steps'],RULES.index(c['rule'])))[0]
    selection=dict(created=time.time(),pca=pca_selected,models=selected,
        rule='Validation certified pair pruning; trained checkpoints only; then updates, seed, steps, grouping.',
        candidate_counts=dict(pca=2,direct=18,fm=54),test_scope='Reused prior test split; adaptive follow-up.')
    save_json(HERE/'artifacts/model_selection.json',selection)
    result.update(ended=time.time(),selection=selection,pass_=True)
    assert len(result['candidates'])==72 and len(result['checkpoints'])==18
    save_json(HERE/'results/training.json',result)
    print(json.dumps(dict(training_complete=True,selection=selection)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results/training_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
