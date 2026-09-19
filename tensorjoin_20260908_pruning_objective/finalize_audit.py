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
    freeze = read('artifacts/timing_freeze.json')
    for path, expected in freeze.items():
        assert sha(path) == expected, path
    training = read('results/training.json')
    proposal_count = 0
    for model in training['models']:
        held = model['history'][0]['reward']
        for j in range(0, 640, 2):
            pair = model['proposals'][j:j+2]
            assert [r['proposal'] for r in pair] == [j+1, j+2]
            assert pair[0]['coordinate'] == pair[1]['coordinate']
            assert pair[0]['coefficient_step'] == -pair[1]['coefficient_step']
            best = max(held, *[r['reward'] for r in pair])
            assert all(r['incumbent_reward_after'] == best for r in pair)
            held = best
            proposal_count += 2
        assert held == model['history'][-1]['reward']
    assert proposal_count == 1920 and len(training['candidates']) == 18
    for checkpoint in training['candidates']:
        assert sha(HERE/'artifacts'/(checkpoint['name']+'.pt')) == checkpoint['checkpoint_sha256']
    refs = read('inherited/reference_manifest.json')['cells']
    admission = read('artifacts/admission.json')
    groups = {f'admission_{key}': admission[key]
              for key in ['samples', 'diagnostics', 'build_admission']}
    for i in range(3):
        run = read(f'results/confirm_{i:02d}.json')
        assert run['pass']
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
    occupancy = [check for family in admission['geometry'].values()
                 for split in family.values() for check in split['reference_checks'].values()]
    assert len(occupancy) == 36 and all(c['pass_'] for c in occupancy)
    save('analysis/output_audit.json', dict(
        pass_=True, full_output_records=count,
        groups={name: len(rows) for name, rows in groups.items()},
        full_arrays_compared_at_admission=75,
        recorded_count_and_sha256_match_reference=True,
        occupancy_checks=len(occupancy), guard_passes=len(guard_names),
        proposal_decisions_verified=proposal_count, checkpoint_hashes_verified=18,
        frozen_files_verified=len(freeze),
        reference_manifest_sha256=sha(HERE/'inherited/reference_manifest.json'),
        note='Post-run audit checks saved records; full output checks were executed within each original call.'))

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    sys.path.insert(0, str(HERE/'src'))
    from models import configure
    configure()
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
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'archive_manifest.json'}
    save('archive_manifest.json', dict(created_at=time.time(), files=files,
        exclusions=['__pycache__', 'archive_manifest.json']))
    print(json.dumps(dict(pass_=True, output_records=count, frozen_files=len(freeze),
        archived_files=len(files), gpu4_has_no_compute_processes=not bool(processes))))


if __name__ == '__main__':
    main()
