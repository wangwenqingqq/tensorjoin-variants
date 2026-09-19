import gc
import os
import traceback
import torch
from layout_common import *
from models import ScoreMap,parameter_count,configure,mapped_scores

SEEDS=[2026090814,2026090815,2026090816]
CHECKPOINTS=[300,600,1200]

def guide_graph(q):
    torch.cuda.synchronize()
    start=time.perf_counter()
    norms=(q*q).sum(dim=1)
    parts=[]
    for off in range(0,8192,256):
        rows=q[off:off+256]
        dist=(rows*rows).sum(dim=1)[:,None]+norms[None,:]-2*(rows@q.T)
        dist.clamp_min_(0)
        ii=torch.arange(len(rows),device='cuda')
        dist[ii,off+ii]=float('inf')
        parts.append(torch.topk(dist,8,dim=1,largest=False,sorted=False).indices)
    graph=torch.cat(parts)
    torch.cuda.synchronize()
    return graph,time.perf_counter()-start

def sample_pairs(q,graph,generator):
    anchor=torch.randint(0,len(graph),(512,),device='cuda',generator=generator)
    neighbor=torch.randint(0,8,(512,),device='cuda',generator=generator)
    pos=graph[anchor,neighbor]
    r1=torch.randint(0,len(q),(512,),device='cuda',generator=generator)
    r2=torch.randint(0,len(q),(512,),device='cuda',generator=generator)
    return torch.cat([anchor,r1]),torch.cat([pos,r2])

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    configure()
    freeze={str(p):sha(p) for p in [HERE/'PROTOCOL.md',HERE/'targets_r2.json',
        *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*'))] if p.is_file()}
    save_json(HERE/'artifacts/training_freeze.json',dict(created=time.time(),files=freeze))
    q=np.load(HERE/'artifacts/features_f64.npy')
    pca=np.load(HERE/'artifacts/pca_scores_f64.npy')
    split=dict(np.load(HERE/'artifacts/splits.npz'))
    scale2=json.loads((HERE/'artifacts/features.json').read_text())['scale2']
    train_rows,valid_rows=split['train'],split['validation']
    # No test-row geometry is evaluated until the selected checkpoints freeze.
    mean=q[train_rows].mean(axis=0).astype(np.float32)
    std=np.maximum(q[train_rows].std(axis=0),1e-8).astype(np.float32)
    tq=torch.from_numpy(q[train_rows].astype(np.float32)).to('cuda')
    graph,graph_seconds=guide_graph(tq)
    np.save(HERE/'artifacts/guide_neighbors_train_indices.npy',graph.cpu().numpy(),allow_pickle=False)
    baseline,_,_=geometry_scores(q[valid_rows],pca[valid_rows],scale2)
    result=dict(started=time.time(),guide_seconds=graph_seconds,
                guide_distance_pairs=8192*40000,baseline_validation=baseline,candidates=[],models=[])
    print(json.dumps(dict(guide_seconds=graph_seconds,baseline_validation=baseline)),flush=True)
    for family in ['linear','mlp']:
        for seed in SEEDS:
            torch.manual_seed(seed)
            initialized=time.perf_counter()
            model=ScoreMap(family,mean,std).to('cuda')
            assert parameter_count(model)==(33 if family=='linear' else 6337)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
            generator=torch.Generator(device='cuda').manual_seed(seed)
            torch.cuda.synchronize()
            init_seconds=time.perf_counter()-initialized
            started=time.perf_counter()
            evaluation_seconds=0.
            history=[]
            losses=[]
            for step in range(1,1201):
                ii,jj=sample_pairs(tq,graph,generator)
                qi,qj=tq[ii],tq[jj]
                target=torch.linalg.vector_norm(qi-qj,dim=1)/model.std[0]
                inputs=torch.cat([qi,qj])
                outputs=model(inputs)
                si,sj=outputs[:1024],outputs[1024:]
                residual=outputs-(inputs[:,0]-model.mean[0])/model.std[0]
                loss=((torch.abs(si-sj)-target).square()/(target+.1)).mean()+1e-4*residual.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
                losses.append(loss.detach())
                if step%50==0:
                    recent=float(torch.stack(losses[-50:]).mean().item())
                    assert np.isfinite(recent)
                    history.append(dict(step=step,loss=recent))
                if step in CHECKPOINTS:
                    torch.cuda.synchronize()
                    training_seconds=time.perf_counter()-started-evaluation_seconds
                    checkpoint_start=time.perf_counter()
                    model.eval()
                    scores=mapped_scores(model,q[valid_rows])
                    valid,_,_=geometry_scores(q[valid_rows],scores,scale2)
                    name=f'{family}_s{seed}_t{step}'
                    path=HERE/'artifacts'/(name+'.pt')
                    torch.save(dict(family=family,seed=seed,step=step,
                                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)
                    record=dict(name=name,family=family,seed=seed,step=step,
                        parameters=parameter_count(model),training_seconds=training_seconds,
                        initialization_seconds=init_seconds,validation=valid,
                        checkpoint_sha256=sha(path),loss=history[-1]['loss'])
                    result['candidates'].append(record)
                    print(json.dumps(record),flush=True)
                    evaluation_seconds+=time.perf_counter()-checkpoint_start
                    model.train()
            result['models'].append(dict(family=family,seed=seed,history=history,
                training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
                initialization_seconds=init_seconds,total_seconds=time.perf_counter()-initialized))
            del model,optimizer
            gc.collect()
            torch.cuda.empty_cache()
    selected={}
    for family in ['linear','mlp']:
        options=[r for r in result['candidates'] if r['family']==family]
        chosen=sorted(options,key=lambda r:(-r['validation']['mean_pair_pruning'],r['step'],r['seed']))[0]
        selected[family]=chosen
    selection=dict(created=time.time(),models=selected,baseline_validation=baseline,
                   rule='Validation six-cell mean certified pair pruning; test/full rows and query timing unseen.')
    save_json(HERE/'artifacts/model_selection.json',selection)
    result.update(ended=time.time(),selection=selection,pass_=True)
    save_json(HERE/'results/training.json',result)
    print(json.dumps(dict(training_complete=True,selection=selection)),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        save_json(HERE/'results/training_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
