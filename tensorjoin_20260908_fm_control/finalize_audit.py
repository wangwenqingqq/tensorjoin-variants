"""Post-run CPU audit and immutable archive, run on the experiment host."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads((HERE / path).read_text())


def save(path, value):
    with (HERE / path).open('x') as f:
        json.dump(value, f, indent=2)


def command(args):
    p = subprocess.run(args, text=True, capture_output=True, check=True)
    return p.stdout.strip()


def main():
    training_freeze = read('artifacts/training_freeze.json')['files']
    for path, expected in training_freeze.items():
        assert sha(path) == expected, path
    freeze = read('artifacts/timing_freeze.json')
    for path, expected in freeze.items():
        assert sha(path) == expected, path
    training = read('results/training.json')
    selected = read('artifacts/model_selection.json')
    assert training['pass_'] and training['selection'] == selected
    assert read('artifacts/preflight.json')['pass_']
    assert len(training['checkpoints']) == 18 and len(training['candidates']) == 72
    assert selected['candidate_counts'] == dict(pca=2, direct=18, fm=54)
    for checkpoint in training['checkpoints']:
        assert checkpoint['trained'] and checkpoint['step'] in [1000, 2000, 4000]
        assert all(v > 0 for v in checkpoint['weight_changes'].values())
        assert checkpoint['parameters'] == (24992 if checkpoint['family'] == 'fm' else 24864)
        assert sha(HERE/'artifacts'/(checkpoint['name']+'.pt')) == checkpoint['checkpoint_sha256']
        if checkpoint['family'] == 'fm':
            assert checkpoint['time_input_weight_norm'] > 0
    for family, count in [('direct', 18), ('fm', 54)]:
        options = [c for c in training['candidates'] if c['family'] == family]
        assert len(options) == count
        best = sorted(options, key=lambda c: (-c['validation']['mean_pair_pruning'],
            c['step'], c['seed'], c['solver_steps'], ['first', 'balanced'].index(c['rule'])))[0]
        assert best == selected['models'][family]
    assert selected['pca'] == sorted(training['baseline_candidates'], key=lambda c:
        (-c['validation']['mean_pair_pruning'], ['first', 'balanced'].index(c['rule'])))[0]
    refs = read('inherited/reference_manifest.json')['cells']
    admission = read('artifacts/admission.json')
    assert admission['pass'] and admission['flow_diagnostics']['time_dependence_rms'] > 0
    assert [r['nfe'] for r in admission['flow_diagnostics']['solvers']] == [16,32,64,128]
    groups = {f'admission_{key}': admission[key]
              for key in ['samples', 'diagnostics', 'build_admission']}
    for i in range(3):
        run = read(f'results/confirm_{i:02d}.json')
        assert run['pass']
        assert len(run['samples']) == 300 and len(run['single_queries']) == 18
        assert sum(s['phase'] == 'retained' for s in run['samples']) == 240
        assert len(run['mapping']) == 4
        for mapping in run['mapping']:
            assert len(mapping['host_to_host_seconds']) == 7
            assert mapping['nfe'] == (2*mapping['solver_steps'] if mapping['family'] == 'fm' else 1)
        for record in run['single_queries']:
            assert record['includes_build'] and not record['includes_training']
            assert record['includes_ode'] == (record['family'] == 'fm')
        for key in ['samples', 'single_queries']:
            groups[f'confirm_{i:02d}_{key}'] = run[key]
    for records in groups.values():
        for record in records:
            ref = refs[record['cell']]
            assert record['output_count'] == ref['count']
            assert record['output_sha256'] == ref['sha256']
    count = sum(map(len, groups.values()))
    assert count == 1029
    guard_names = ['training_a0', 'admission_a0'] + [f'confirm_{i:02d}_a0' for i in range(3)]
    for label in guard_names:
        guard = read(f'results/{label}_guard.json')
        assert guard['pass'] and guard['exit_code'] == 0
        assert guard['gpu_uuid'] == 'GPU-863c06a5-9f33-0265-b098-013fa840d5db'
        assert guard['max_device_used_mib'] < 4096
    occupancy = [check for family in admission['geometry'].values()
                 for split in family.values() for check in split['reference_checks'].values()]
    assert len(occupancy) == 36 and all(c['pass_'] for c in occupancy)
    save('analysis/output_audit.json', dict(
        pass_=True, full_output_records=count,
        groups={name: len(rows) for name, rows in groups.items()},
        full_arrays_compared_at_admission=75,
        recorded_count_and_sha256_match_reference=True,
        occupancy_checks=len(occupancy), guard_passes=len(guard_names),
        frozen_files_verified=len(freeze), training_frozen_files_verified=len(training_freeze),
        trained_checkpoint_hashes_verified=18, candidate_evaluations_verified=72,
        selection_recomputed_on_validation=True, real_time_dependent_velocity_and_ode=True,
        reference_manifest_sha256=sha(HERE/'inherited/reference_manifest.json'),
        note='Post-run audit checks saved records; full output checks were executed within each original call.'))

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    sys.path.insert(0, str(HERE/'src'))
    from models import configure
    configure()
    assert not torch.cuda.is_initialized()
    from models import load_model, parameter_count
    for family in ['direct', 'fm']:
        spec = selected['models'][family]
        model = load_model(HERE/'artifacts'/(spec['name']+'.pt'), 'cpu')
        assert parameter_count(model) == spec['parameters']
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.cuda.is_initialized()
    packages = {}
    for name in ['torch', 'numpy', 'scipy', 'triton', 'threadpoolctl']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    save('analysis/final_environment.json', dict(
        captured_at=time.time(), hostname=platform.node(), platform=platform.platform(),
        python=sys.version, executable=sys.executable, packages=packages,
        torch_cuda_build=torch.version.cuda,
        cpu=command(['lscpu', '--json']),
        process_cpu_affinity_count=len(os.sched_getaffinity(0)),
        recreated_frozen_model_configuration=dict(
            torch_threads=torch.get_num_threads(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        audit_created_cuda_context=False))
    gpu_query = command(['nvidia-smi', '-i', '4',
        '--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'])
    processes = command(['nvidia-smi', '-i', '4',
        '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
    assert 'GPU-863c06a5-9f33-0265-b098-013fa840d5db' in gpu_query
    save('analysis/final_gpu_state.json', dict(captured_at=time.time(), gpu=gpu_query,
        compute_processes=processes, gpu4_has_no_compute_processes=not bool(processes)))
    files = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p != HERE/'archive_manifest.json'}
    save('archive_manifest.json', dict(created_at=time.time(), files=files,
        exclusions=['any __pycache__ directory', 'root archive_manifest.json']))
    print(json.dumps(dict(pass_=True, output_records=count, frozen_files=len(freeze),
        archived_files=len(files), gpu4_has_no_compute_processes=not bool(processes))))


if __name__ == '__main__':
    main()
