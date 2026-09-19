import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT/'src'))
from engine import *
from train import prediction_metrics
from models import heun
# Wildcard imports carry the parent's HERE; restore the follow-up directory.
FOLLOWUP=Path(__file__).resolve().parent


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4';configure();verify_timing_freeze()
    freeze=json.loads((PARENT/'artifacts/timing_freeze.json').read_text())
    freeze.update({str(p):sha(p) for p in [FOLLOWUP/'PROTOCOL.md',FOLLOWUP/'run.py']})
    save_json(FOLLOWUP/'freeze.json',dict(created=time.time(),files=freeze))
    x=source();plan=saved_plan(x);models=selected_models();model=models['fm']
    split=dict(np.load(PARENT/'artifacts/pair_splits.npz'))['validation']
    cell=CELLS[3];pool=eligible(plan['bounds'],cell,split)
    ids=np.random.default_rng(2026090931).choice(pool,512,replace=False)
    with torch.no_grad():
        f=torch.from_numpy(plan['features'][ids]).to('cuda')
        z=torch.from_numpy(initial_state(plan['coarse'][ids],cell['T'])).to('cuda')
        emb=model.embedding(model.conditions(f,cell['T']));field=lambda zz,t:model.field(emb,zz,t)
        ends={steps:heun(field,z,steps) for steps in [2,4,8,16,32,64,128,256,512]}
        reference=ends[512];denom=max(float(reference.square().mean().item()),1e-12)
        diagnostics=[]
        for steps,endpoint in ends.items():
            diagnostics.append(dict(steps=steps,nfe=2*steps,
                relative_rms_vs_512=float(torch.sqrt((endpoint-reference).square().mean()/denom).item()),
                sign_disagreement=float(((endpoint>=0)!=(reference>=0)).float().mean().item())))
        choices=[r['steps'] for r in diagnostics if r['steps'] in [16,32,64,128,256] and r['relative_rms_vs_512']<=.01 and r['sign_disagreement']<=.01]
        chosen=choices[0] if choices else 512
        np.savez_compressed(FOLLOWUP/'validation_endpoints.npz',tile_ids=ids,steps=np.asarray(list(ends)),endpoints=np.asarray([v.cpu().numpy() for v in ends.values()]),initial=z.cpu().numpy())
    result=dict(started=time.time(),diagnostics=diagnostics,selected_steps=chosen,convergence_gate_passed=bool(choices),
        selection_scope='512 validation tiles only, fixed parent model weights, adaptive supplementary resolution check.',
        raw_prediction_metrics=[],samples=[],pass_=False)
    save_json(FOLLOWUP/'resolution_selection.json',dict(diagnostics=diagnostics,selected_steps=chosen,convergence_gate_passed=bool(choices),created=time.time()))
    print(json.dumps(dict(resolution_selection=result['selected_steps'],diagnostics=diagnostics)),flush=True)
    model.solver_steps=chosen
    counts=np.load(PARENT/'artifacts/counts_full.npy');b=plan['bounds']
    for ci,cell in enumerate(CELLS):
        ids=eligible(b,cell);initial=initial_state(plan['coarse'][ids],cell['T'])
        pred=predict(model,plan['features'][ids],initial,cell['T'],steps=chosen)
        result['raw_prediction_metrics'].append(dict(cell=cell['name'],solver_steps=chosen,**prediction_metrics(pred,counts[ci,ids])))
        np.save(FOLLOWUP/f'prediction_fm_{cell["name"]}.npy',pred,allow_pickle=False)
    op,radix=retained.output.Matrix(),retained.output.Radix()
    configs=['pca','heuristic25','direct25','fm25'];expected={}
    for ci,cell in enumerate(CELLS):
        for method in METHODS:
            for config in configs:
                rec,_,_,mask=run(op,radix,x,cell,method,config,plan,models,exact=True)
                assert np.all(counts[ci,~mask]==0)
                expected[config+'_'+cell['name']]=digest(mask)
                rec.update(phase='warmup',repeat=-1,resolution_followup=True)
                result['samples'].append(rec)
        for repeat in range(3):
            offset=(ci+repeat)%4;order=configs[offset:]+configs[:offset]
            if repeat%2:order=order[::-1]
            for method in METHODS[::(-1 if repeat%2 else 1)]:
                for config in order:
                    rec,_,_,mask=run(op,radix,x,cell,method,config,plan,models)
                    assert digest(mask)==expected[config+'_'+cell['name']]
                    rec.update(phase='retained',repeat=repeat,resolution_followup=True)
                    result['samples'].append(rec);print(json.dumps(rec),flush=True)
        op.capture();gc.collect();torch.cuda.empty_cache()
    verify_timing_freeze()
    result.update(pass_=True,ended=time.time(),compiled=op.capture(),admitted_mask_hashes=expected)
    result['pass']=True;save_json(FOLLOWUP/'results.json',result)
    print(json.dumps(dict(resolution_complete=True,selected_steps=chosen,complete_calls=len(result['samples']))),flush=True)

if __name__=='__main__':main()
