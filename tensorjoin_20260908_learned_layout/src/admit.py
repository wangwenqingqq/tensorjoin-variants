import os
import traceback
from benchmark import *
from models import ScoreMap

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    configure()
    assert json.loads((HERE/'results/training.json').read_text())['pass_']
    # Verify every resource frozen before learning; later evaluation code is
    # separate and cannot change already-selected weights or training inputs.
    for path,h in json.loads((HERE/'artifacts/training_freeze.json').read_text())['files'].items():
        assert sha(path)==h,path
    result=dict(started=time.time(),samples=[],diagnostics=[],geometry={},pass_=False)
    x=source()
    q=np.load(HERE/'artifacts/features_f64.npy')
    pca=np.load(HERE/'artifacts/pca_scores_f64.npy')
    scale2=json.loads((HERE/'artifacts/features.json').read_text())['scale2']
    split=dict(np.load(HERE/'artifacts/splits.npz'))
    models=selected_models()
    scores={'pca':pca}
    for family in ['linear','mlp']:
        scores[family]=mapped_scores(models[family],q)
        assert np.isfinite(scores[family]).all()
    # A zero-residual neural map should preserve the PCA ordering/geometry
    # up to tiny FP32 score ties. This checks the evaluator independently of
    # whether the trained objective succeeds.
    zero=ScoreMap('mlp',models['mlp'].mean.cpu(),models['mlp'].std.cpu()).to('cuda').eval()
    zero_score=mapped_scores(zero,q[split['validation']])
    zero_geom,_,_=geometry_scores(q[split['validation']],zero_score,scale2)
    baseline=json.loads((HERE/'artifacts/model_selection.json').read_text())['baseline_validation']
    assert abs(zero_geom['mean_pair_pruning']-baseline['mean_pair_pruning'])<1e-4
    result['zero_residual_control']=zero_geom
    del zero
    plans={}
    for family in FAMILIES:
        result['geometry'][family]={}
        for name,ids in [('test',split['test']),('full',np.arange(len(x)))]:
            geo,perm,b=geometry_scores(q[ids],scores[family][ids],scale2)
            checks={c['name']:reference_occupancy_check(perm,b,c,None if name=='full' else ids)
                    for c in CELLS}
            geo['reference_checks']=checks
            # Width is diagnostic only, never used for model selection.
            qq=q[ids][perm]
            width=np.asarray([np.ptp(qq[i:i+64],axis=0) for i in range(0,len(qq),64)])
            geo['mean_pc1_block_width']=float(width[:,0].mean())
            geo['mean_block_bbox_diagonal']=float(np.linalg.norm(width,axis=1).mean())
            result['geometry'][family][name]=geo
            print(json.dumps(dict(family=family,split=name,geometry=geo)),flush=True)
            if name=='full':
                np.save(HERE/'artifacts'/f'{family}_permutation.npy',perm,allow_pickle=False)
                np.savez_compressed(HERE/'artifacts'/f'{family}_bounds.npz',**b)
                np.save(HERE/'artifacts'/f'{family}_scores.npy',scores[family],allow_pickle=False)
                plans[family]=dict(layout=family,perm=perm,y=np.ascontiguousarray(x[perm]),bounds=b)
    result['mapping']=mapping_benchmark(q,models)
    print(json.dumps(dict(mapping=result['mapping'])),flush=True)
    op,radix=retained.output.Matrix(),retained.output.Radix()
    for cell in CELLS:
        for method in METHODS:
            for config in CONFIGS:
                rec,kernel=run(op,radix,x,cell,method,config,plans,models,exact=True)
                capture_remap(kernel)
                result['samples'].append(rec)
                print(json.dumps(rec),flush=True)
            rec,kernel=run(op,radix,x,cell,method,'mlp_geo',plans,models,exact=True,diagnostic=True)
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
    paths=[HERE/'PROTOCOL.md',HERE/'targets_r2.json',HERE/'run_campaign.py',
           *sorted((HERE/'src').glob('*.py')),*sorted((HERE/'artifacts').glob('*')),
           *sorted((OLD/'src').glob('*.py')),*sorted((retained.SHARED/'src').glob('*.py')),
           retained.SHARED/'artifacts/output.so',retained.SHARED/'artifacts/build.json',
           SWEEP/'artifacts/thresholds.json',SWEEP/'artifacts/reference_manifest.json',
           *sorted((retained.output.OLD/'src').glob('*.py')),
           *sorted(retained.output.OLD.glob('*manifest.json'))]
    freeze={str(p):sha(p) for p in paths if p.is_file()}
    freeze.update({str(ROOT/p):h for p,h in op.identities.items()})
    save_json(HERE/'artifacts/timing_freeze.json',freeze)
    print(json.dumps(dict(admission_complete=True,configurations=len(result['samples']),
                         frozen_files=len(freeze))),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results/admission_failed_a0.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
