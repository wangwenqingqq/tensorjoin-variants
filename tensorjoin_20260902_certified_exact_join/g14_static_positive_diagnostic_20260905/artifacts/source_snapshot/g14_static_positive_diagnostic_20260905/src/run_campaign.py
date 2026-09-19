"""Run the bounded G14 diagnostic, preserving every slot without replacement."""

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'src'))
from g2b_public_common import atomic_json, sha256_file

PYTHON = '@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
GUARD = PROJECT / 'src/run_g5_guarded_process.py'
SCHEDULE = [
    ('default', ('g2b', 'mistic', 'g5')),
    ('node0', ('g2b', 'mistic', 'g5')),
    ('node0', ('g5', 'mistic', 'g2b')),
    ('default', ('g5', 'mistic', 'g2b')),
]


def text_command(command):
    r = subprocess.run(command, text=True, capture_output=True, check=False)
    return {'command': command, 'returncode': r.returncode,
            'stdout': r.stdout, 'stderr': r.stderr}


def verify_sources():
    frozen = json.loads((HERE / 'artifacts/frozen_hashes.json').read_text())
    for relative, digest in frozen.items():
        if sha256_file(PROJECT / relative) != digest:
            raise RuntimeError(f'Frozen source/evidence mismatch: {relative}')
    return frozen


def code_identity(result):
    return {k: {suffix: v[suffix + '_sha256'] for suffix in ('cubin', 'ptx')}
            for k, v in result['selected_cache_artifacts_after_timer'].items()}


def main():
    if platform.node() != 'gpu-host-8':
        raise RuntimeError('Wrong host')
    manifest = HERE / 'results/campaign.json'
    if manifest.exists() or (HERE / 'results/start.json').exists():
        raise FileExistsError('This campaign is append-only and not restartable')
    frozen = verify_sources()
    expected_code = code_identity(json.loads((PROJECT / 'results/g5_screen_tensorjoin_r0_p2_a0.json').read_text()))
    expected_binary = json.loads((PROJECT / 'results/g5_screen_mistic_r0_p1_a0.json').read_text())['binary_sha256']
    binary = PROJECT / 'adapters/mistic_g2b_public/build/main_d512'
    if sha256_file(binary) != expected_binary:
        raise RuntimeError('MiSTIC binary mismatch')
    lock_path = '/tmp/tensorjoin_gpu2_campaign.lock'
    with open(lock_path, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = {name: text_command(command) for name, command in {
            'gpu': ['nvidia-smi'], 'topology': ['nvidia-smi', 'topo', '-m'],
            'numa': ['numactl', '-H'], 'cpu': ['lscpu'],
            'users': ['who'], 'processes': ['ps', '-eo', 'pid,ppid,user,psr,pcpu,rss,args', '--sort=-pcpu'],
            'numa_permission': ['numactl', '--cpunodebind=0', '--membind=0', '/usr/bin/true'],
        }.items()}
        if preflight['numa_permission']['returncode'] != 0:
            raise RuntimeError('Strict NUMA binding unavailable; do not change control silently')
        record = {'diagnostic_only': True, 'schedule': SCHEDULE, 'host': platform.node(),
                  'pid': os.getpid(), 'started_unix': time.time(), 'lock': lock_path,
                  'frozen_hashes': frozen, 'expected_g5_code': expected_code,
                  'mistic_binary_sha256': expected_binary, 'preflight': preflight}
        atomic_json(HERE / 'results/start.json', record)
        records = []
        status = 'complete_diagnostic_no_promotion'
        for block, (placement, order) in enumerate(SCHEDULE):
            for position, method in enumerate(order):
                verify_sources()
                rid = f'g14_b{block}_p{position}_{placement}_a0'
                label = f'{rid}_{method}'
                if method == 'mistic':
                    runner = PROJECT / 'src/run_g5_mistic_public.py'
                    result_path = PROJECT / f'results/g5_screen_mistic_{rid}.json'
                else:
                    runner = HERE / f'src/run_{method}_diagnostic.py'
                    result_path = HERE / f'results/{method}_{rid}.json'
                child = [PYTHON, str(runner), '--record-id', rid, '--phase', 'screen']
                if placement == 'node0':
                    child = ['numactl', '--cpunodebind=0', '--membind=0'] + child
                command = [PYTHON, str(GUARD), '--label', label, '--expected-result',
                           str(result_path.relative_to(PROJECT)), '--physical-gpu', '2', '--'] + child
                env = os.environ.copy()
                env['PYTHONPATH'] = str(PROJECT / 'src') + ':' + str(HERE / 'src')
                env['G14_PLACEMENT'] = placement
                slot = {'block': block, 'placement': placement, 'position': position,
                        'method': method, 'record_id': rid, 'command': command,
                        'started_unix': time.time(), 'loadavg_before': list(os.getloadavg())}
                print('G14_SLOT_START ' + json.dumps(slot), flush=True)
                # The guard owns and terminates only its own process group on contamination.
                completed = subprocess.run(command, env=env, check=False)
                slot['returncode'] = completed.returncode
                slot['loadavg_after'] = list(os.getloadavg())
                guard_path = PROJECT / f'results/g5_guard_{label}.json'
                if guard_path.is_file():
                    guard = json.loads(guard_path.read_text())
                    slot.update(guard_admitted=guard.get('admitted'),
                                guard_path=str(guard_path.relative_to(PROJECT)),
                                guard_sha256=sha256_file(guard_path))
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    slot.update(result_path=str(result_path.relative_to(PROJECT)),
                                result_sha256=sha256_file(result_path),
                                public_seconds=result['public_seconds'],
                                exact_contract_pass=result['correctness']['exact_contract_pass'])
                    if method != 'mistic':
                        diagnostic = result['g14_diagnostic']
                        discrepancy = abs(diagnostic['phase_wall_sum_seconds'] - result['public_seconds'])
                        slot['phase_closure_abs_seconds'] = discrepancy
                        slot['phase_closure_pass'] = discrepancy <= .001
                    if method == 'g5':
                        slot['binary_identity_pass'] = code_identity(result) == expected_code
                    elif method == 'mistic':
                        slot['binary_identity_pass'] = result['binary_sha256'] == expected_binary
                admitted = (completed.returncode == 0 and slot.get('guard_admitted')
                            and slot.get('exact_contract_pass')
                            and slot.get('phase_closure_pass', method == 'mistic')
                            and slot.get('binary_identity_pass', method == 'g2b'))
                slot['admitted'] = bool(admitted)
                records.append(slot)
                atomic_json(HERE / f'results/slot_b{block}_p{position}.json', slot)
                print('G14_SLOT_COMPLETE ' + json.dumps(slot), flush=True)
                if not admitted:
                    status = 'stopped_on_failed_slot_no_replacement'
                    break
            if status != 'complete_diagnostic_no_promotion':
                break
        atomic_json(manifest, {**record, 'status': status, 'records': records,
                               'finished_unix': time.time()})
        print('G14_COMPLETE ' + str(manifest), flush=True)
        return 0 if status == 'complete_diagnostic_no_promotion' else 2


if __name__ == '__main__':
    raise SystemExit(main())
