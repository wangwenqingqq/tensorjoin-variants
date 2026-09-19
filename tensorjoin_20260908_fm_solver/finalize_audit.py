"""CPU-only independent output, selection, dependency and archive checks."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
UUID='GPU-863c06a5-9f33-0265-b098-013fa840d5db'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads((HERE/p).read_text())
def save(p,r):
    with (HERE/p).open('x') as f:json.dump(r,f,indent=2)
def command(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip()


def main():
    training_freeze=read('artifacts/training_freeze.json')['files']
    timing_freeze=read('artifacts/timing_freeze.json')
    resolution_freeze=read('resolution_check/freeze.json')['files']
    for freeze in [training_freeze,timing_freeze,resolution_freeze]:
        for p,want in freeze.items():assert sha(p)==want,p
    training=read('results/training.json');selection=read('artifacts/model_selection.json')
    assert training['pass_'] and selection==training['selection']
    assert len(training['checkpoints'])==18 and len(training['candidates'])==36
    assert selection['candidate_counts']==dict(direct=9,fm=27)
    for ck in training['checkpoints']:
        assert ck['trained'] and all(v>0 for v in ck['weight_changes'].values())
        assert ck['parameters']==(5761 if ck['family']=='fm' else 5697)
        assert sha(HERE/'artifacts'/(ck['name']+'.pt'))==ck['checkpoint_sha256']
        if ck['family']=='fm':assert ck['time_weight_norm']>0
    for family,count in [('direct',9),('fm',27)]:
        rows=[c for c in training['candidates'] if c['family']==family];assert len(rows)==count
        best=sorted(rows,key=lambda c:(-c['validation']['mean_extra_pair_pruning'],c['solver_steps'],c['step'],c['seed']))[0]
        assert best==selection['models'][family]
    preflight=read('artifacts/preflight.json');assert preflight['pass_']
    assert read('analysis/import_paths.json')['pass_']
    blocks=dict(np.load(HERE/'artifacts/block_splits.npz'))
    assert [len(blocks[k]) for k in ['train','validation','test']]==[625,156,157]
    assert np.array_equal(np.sort(np.concatenate(list(blocks.values()))),np.arange(938))
    split=dict(np.load(HERE/'artifacts/pair_splits.npz'))
    assert not np.any(split['train']&split['validation']) and not np.any(split['test']&(split['train']|split['validation']))
    labels=np.load(HERE/'artifacts/counts_train_validation.npy');counts=np.load(HERE/'artifacts/counts_full.npy')
    seen=split['train']|split['validation'];assert np.all(labels[:,~seen]==-1) and np.array_equal(labels[:,seen],counts[:,seen])
    refs=read('inherited/reference_manifest.json')['cells'];cells=['k4','k16','k64','original','k256','k1024']
    for i,cell in enumerate(cells):assert counts[i].sum()==refs[cell]['count']
    admission=read('artifacts/admission.json');assert admission['pass']
    masks=dict(np.load(HERE/'artifacts/admitted_masks.npz'));assert len(masks)==48
    geometry={(r['configuration'],r['cell']):r for r in admission['geometry']};assert len(geometry)==48
    for (config,cell),row in geometry.items():
        mask=masks[config+'_'+cell]
        assert digest(mask)==row['mask_sha256'] and row['reference_occupancy_pass']
        assert np.all(counts[cells.index(cell),~mask]==0)
    groups={'admission_'+k:admission[k] for k in ['samples','diagnostics','build_admission']}
    assert [len(admission[k]) for k in ['samples','diagnostics','build_admission']]==[96,12,5]
    for i in range(3):
        r=read(f'results/confirm_{i:02d}.json');assert r['pass']
        assert len(r['samples'])==480 and len(r['single_queries'])==30
        assert sum(s['phase']=='retained' for s in r['samples'])==384
        assert len(r['mapping'])==4 and all(len(m['host_to_host_seconds'])==7 for m in r['mapping'])
        for k in ['samples','single_queries']:groups[f'confirm_{i:02d}_{k}']=r[k]
    for records in groups.values():
        for s in records:
            ref=refs[s['cell']];assert s['output_count']==ref['count'] and s['output_sha256']==ref['sha256']
            g=geometry[(s['configuration'],s['cell'])]
            assert s['tile_mask_sha256']==g['mask_sha256']
            assert s['certificate_selection_sha256']==g['certificate_selection_sha256']
            if s['configuration'].startswith('fm'):
                assert s['includes_ode'] and s['routing']['solver_steps']==selection['models']['fm']['solver_steps']
    assert sum(map(len,groups.values()))==1643
    resolution=read('resolution_check/results.json');assert resolution['pass']
    assert len(resolution['samples'])==192 and sum(s['phase']=='retained' for s in resolution['samples'])==144
    options=[r['steps'] for r in resolution['diagnostics'] if r['steps'] in [16,32,64,128,256]
        and r['relative_rms_vs_512']<=.01 and r['sign_disagreement']<=.01]
    assert resolution['selected_steps']==(options[0] if options else 512)
    # Independently reconstruct supplementary masks from saved predictions and
    # the admitted all-certificate mask, using full stable sorting for selection.
    # This is an after-the-fact audit; timed routing never reads these masks.
    b=dict(np.load(HERE/'artifacts/bounds.npz'))
    for ci,cell in enumerate(cells):
        base=masks['pca_'+cell];ids=np.flatnonzero(base & (b['tr']!=b['tc']))
        scores=np.load(HERE/'resolution_check'/f'prediction_fm_{cell}.npy')
        assert len(scores)==len(ids) and np.isfinite(scores).all()
        selected=ids[np.argsort(scores,kind='stable')[:int(np.ceil(.25*len(ids)))]]
        mask=base.copy();mask[selected[~masks['all_'+cell][selected]]]=False
        assert digest(mask)==resolution['admitted_mask_hashes']['fm25_'+cell]
        assert np.all(counts[ci,~mask]==0)
        row=next(r for r in resolution['raw_prediction_metrics'] if r['cell']==cell)
        assert row['missed_reference_entries']==int(counts[ci,ids[scores<0]].sum())
        for family in ['direct','fm']:
            pred=np.load(HERE/'artifacts'/f'prediction_{family}_{cell}.npy')[ids]
            assert np.isfinite(pred).all()
            row=next(r for r in admission['raw_prediction_metrics'] if r['split']=='full' and r['family']==family and r['cell']==cell)
            assert row['missed_reference_entries']==int(counts[ci,ids[pred<0]].sum())
    for s in resolution['samples']:
        ref=refs[s['cell']];assert s['output_count']==ref['count'] and s['output_sha256']==ref['sha256']
        assert s['tile_mask_sha256']==resolution['admitted_mask_hashes'][s['configuration']+'_'+s['cell']]
        if s['configuration']=='fm25':assert s['routing']['solver_steps']==resolution['selected_steps']
    groups['resolution']=resolution['samples']
    success=['preflight_a2','training_a0','admission_a0','confirm_00_a0','confirm_01_a0','confirm_02_a0','resolution_a0']
    for name in success:
        g=read(f'results/{name}_guard.json');assert g['pass'] and g['exit_code']==0 and g['gpu_uuid']==UUID
        assert g['max_device_used_mib']<4096
    for name in ['preflight_a0','preflight_a1']:
        g=read(f'results/{name}_guard.json');assert not g['pass'] and g['exit_code']==1
        version=HERE/'development'/(name+'_src')
        for file,want in g['source_sha256'].items():assert sha(version/file)==want,(name,file)
    prep=read('artifacts/preparation.json')
    data=Path('@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/data/g2b_cifar60000/vectors_f32.npy')
    assert sha(data)==prep['input_sha256']
    total=sum(map(len,groups.values()));assert total==1835
    save('analysis/output_audit.json',dict(pass_=True,full_output_records=total,
        groups={k:len(v) for k,v in groups.items()},full_arrays_compared=161,main_masks_checked=48,supplementary_fm_masks_reconstructed=6,
        raw_full_prediction_missed_entries_recomputed=True,trained_checkpoint_hashes_verified=18,candidate_evaluations_verified=36,
        validation_selection_recomputed=True,resolution_selection_recomputed=True,whole_block_splits_disjoint=True,
        training_frozen_files=len(training_freeze),timing_frozen_files=len(timing_freeze),resolution_frozen_files=len(resolution_freeze),
        successful_guards=len(success),preserved_development_failures=2,
        note='Saved-record audit; original workers performed each full-array or count/hash check. Supplementary masks independently reconstructed without rerunning GPU work.'))
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(HERE/'src'))
    import torch
    from models import configure,load_model,parameter_count
    configure();assert not torch.cuda.is_initialized()
    for family in ['direct','fm']:
        spec=selection['models'][family];m=load_model(HERE/'artifacts'/(spec['name']+'.pt'),'cpu')
        assert parameter_count(m)==spec['parameters'] and all(torch.isfinite(p).all() for p in m.parameters())
    assert not torch.cuda.is_initialized()
    versions={}
    for name in ['torch','numpy','scipy','triton','threadpoolctl']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    save('analysis/final_environment.json',dict(captured_at=time.time(),hostname=platform.node(),platform=platform.platform(),python=sys.version,
        executable=sys.executable,packages=versions,torch_cuda_build=torch.version.cuda,cpu=command(['lscpu','--json']),
        cpu_affinity_count=len(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        audit_created_cuda_context=False))
    gpu=command(['nvidia-smi','-i','4','--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
    processes=command(['nvidia-smi','-i','4','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    assert UUID in gpu
    save('analysis/final_gpu_state.json',dict(captured_at=time.time(),gpu=gpu,compute_processes=processes,gpu4_has_no_compute_processes=not bool(processes)))
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=HERE/'archive_manifest.json'}
    save('archive_manifest.json',dict(created_at=time.time(),files=files,exclusions=['any __pycache__ directory','root archive_manifest.json']))
    print(json.dumps(dict(pass_=True,full_output_records=total,archived_files=len(files),successful_guards=len(success),gpu4_has_no_compute_processes=not bool(processes))))

if __name__=='__main__':main()
