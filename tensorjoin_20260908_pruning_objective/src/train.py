"""Direct coordinate search using the deployed certified block-pruning reward."""
import gc
import os
import traceback
import torch
from layout_common import *
from models import ScoreMap,parameter_count,configure,mapped_scores,load_model

SEEDS=[2026090821,2026090822,2026090823]
DEVIATIONS=[.03,.01,.003,.001,.0003]


def evaluate(q,scores,scale2):
    geometry,_,bounds=geometry_scores(q,scores,scale2)
    counts=[int(bounds['weights'][bounds['combined']>cell['T']+SQ_PAD].sum()) for cell in CELLS]
    return sum(counts),geometry


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    configure()
    paths=[HERE/'PROTOCOL.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
           *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),
           *sorted((HERE/'inherited').glob('*')),*sorted((OLD/'src').glob('*.py')),
           SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json']
    freeze={str(p):sha(p) for p in paths if p.is_file()}
    save_json(HERE/'artifacts/training_freeze.json',dict(created=time.time(),files=freeze))
    q=np.load(HERE/'artifacts/features_f64.npy')
    pca=np.load(HERE/'artifacts/pca_scores_f64.npy')
    split=dict(np.load(HERE/'artifacts/splits.npz'))
    scale2=json.loads((HERE/'artifacts/features.json').read_text())['scale2']
    train_rows,valid_rows=split['train'],split['validation']
    tq_np=q[train_rows]
    mean=tq_np.mean(axis=0).astype(np.float32)
    std=np.maximum(tq_np.std(axis=0),1e-8).astype(np.float32)
    tq=torch.from_numpy(tq_np.astype(np.float32)).to('cuda')
    baseline_train_reward,baseline_train=evaluate(tq_np,pca[train_rows],scale2)
    baseline,_,_=geometry_scores(q[valid_rows],pca[valid_rows],scale2)
    previous=json.loads((HERE/'inherited/previous_model_selection.json').read_text())['models']['mlp']
    proxy=load_model(HERE/'artifacts/proxy.pt')
    proxy_validation,_,_=geometry_scores(q[valid_rows],mapped_scores(proxy,q[valid_rows]),scale2)
    assert abs(proxy_validation['mean_pair_pruning']-previous['validation']['mean_pair_pruning'])<1e-12
    del proxy
    result=dict(started=time.time(),baseline_training=baseline_train,
        baseline_training_reward=baseline_train_reward,baseline_validation=baseline,
        proxy_validation=proxy_validation,candidates=[],models=[],
        parameter_search='64 output weights; random hidden weights fixed; integer deployed-bound reward',
        total_proposals=0,guide_seconds=0.)
    print(json.dumps(dict(baseline_training=baseline_train,baseline_validation=baseline,
                         proxy_validation=proxy_validation)),flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        initialized=time.perf_counter()
        model=ScoreMap('mlp',mean,std).to('cuda').eval()
        assert parameter_count(model)==6337
        norm=(tq-model.mean)/model.std
        hidden=model.residual[:-1](norm)
        base=norm[:,0]
        activation_std=torch.std(hidden,dim=0,unbiased=False).clamp_min(1e-6).cpu().numpy()
        last=model.residual[-1]
        assert torch.count_nonzero(last.bias)==0 and torch.count_nonzero(last.weight)==0
        torch.cuda.synchronize()
        init_seconds=time.perf_counter()-initialized
        evaluation_seconds=0.
        started=time.perf_counter()
        proposals=[]
        history=[]
        evaluated=0
        accepted=0

        def scores():
            return (base+last(hidden).squeeze(-1)).cpu().numpy().copy()

        current_reward,current_geo=evaluate(tq_np,scores(),scale2)
        assert abs(current_geo['mean_pair_pruning']-baseline_train['mean_pair_pruning'])<1e-4

        def checkpoint(sweep):
            nonlocal evaluation_seconds
            torch.cuda.synchronize()
            training_seconds=time.perf_counter()-started-evaluation_seconds
            eval_start=time.perf_counter()
            assert np.array_equal(scores(),mapped_scores(model,tq_np))
            valid,_,_=geometry_scores(q[valid_rows],mapped_scores(model,q[valid_rows]),scale2)
            name=f'direct_s{seed}_e{evaluated}'
            path=HERE/'artifacts'/(name+'.pt')
            torch.save(dict(family='mlp',seed=seed,step=evaluated,
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)
            rec=dict(name=name,family='direct',seed=seed,step=evaluated,sweep=sweep,
                parameters=parameter_count(model),optimized_parameters=64,
                training_seconds=training_seconds,initialization_seconds=init_seconds,
                training_reward=current_reward,training=current_geo,
                validation=valid,accepted=accepted,
                output_weight_squared_norm=float(last.weight.square().sum().item()),
                checkpoint_sha256=sha(path),zero_initialization=evaluated==0)
            result['candidates'].append(rec)
            history.append(dict(proposals=evaluated,reward=current_reward,
                train_pruning=current_geo['mean_pair_pruning'],
                validation_pruning=valid['mean_pair_pruning'],accepted=accepted))
            print(json.dumps(rec),flush=True)
            evaluation_seconds+=time.perf_counter()-eval_start
            return training_seconds

        checkpoint(0)
        for sweep,deviation in enumerate(DEVIATIONS,1):
            for coordinate in rng.permutation(64):
                coordinate=int(coordinate)
                original=last.weight.detach().clone()
                current_norm=float(original.square().sum().item())
                options=[(current_reward,-current_norm,0,original,current_geo)]
                trial_rows=[]
                delta=float(deviation/activation_std[coordinate])
                for sign in [1,-1]:
                    last.weight.copy_(original)
                    last.weight[0,coordinate]+=sign*delta
                    reward,geo=evaluate(tq_np,scores(),scale2)
                    weight=last.weight.detach().clone()
                    norm2=float(weight.square().sum().item())
                    evaluated+=1
                    options.append((reward,-norm2,-len(options),weight,geo))
                    trial_rows.append(dict(proposal=evaluated,sweep=sweep,
                        target_std=deviation,coordinate=coordinate,coefficient_step=sign*delta,
                        reward=reward,mean_pair_pruning=geo['mean_pair_pruning'],
                        output_weight_squared_norm=norm2))
                best=max(range(3),key=lambda i:options[i][:3])
                chosen=options[best]
                assert chosen[0]>=current_reward
                current_reward,current_geo=chosen[0],chosen[4]
                last.weight.copy_(chosen[3])
                accepted+=int(best!=0)
                for i,row in enumerate(trial_rows,1):
                    row['accepted']=best==i
                    row['incumbent_reward_after']=current_reward
                    proposals.append(row)
            training_seconds=checkpoint(sweep)
        assert evaluated==640 and len(proposals)==640
        result['models'].append(dict(family='direct',seed=seed,history=history,
            proposals=proposals,accepted=accepted,training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,initialization_seconds=init_seconds,
            total_seconds=time.perf_counter()-initialized))
        result['total_proposals']+=evaluated
        del model,hidden,norm,last,base
        gc.collect()
        torch.cuda.empty_cache()
    chosen=sorted(result['candidates'],key=lambda r:(-r['validation']['mean_pair_pruning'],r['step'],r['seed']))[0]
    proxy_spec=dict(previous,name='proxy',family='proxy',historical_training=True,
                    validation=proxy_validation,optimized_parameters=6337)
    selection=dict(created=time.time(),models=dict(proxy=proxy_spec,direct=chosen),
        baseline_validation=baseline,
        rule='Validation certified six-threshold pair pruning; zero initialization eligible; no test/full geometry or query time used.',
        test_scope='Reused prior test rows; adaptive within-dataset follow-up.')
    save_json(HERE/'artifacts/model_selection.json',selection)
    result.update(ended=time.time(),selection=selection,pass_=True)
    assert result['total_proposals']==1920 and len(result['candidates'])==18
    save_json(HERE/'results/training.json',result)
    print(json.dumps(dict(training_complete=True,selection=selection)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results/training_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
