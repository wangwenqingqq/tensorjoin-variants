"""Capture the actual full-operator dispatch and all three pedantic GEMM shapes."""

import fcntl
import json
import os
import subprocess
from pathlib import Path

from g2b_public_common import atomic_json, sha256_file

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
PYTHON = '@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
RUNNER = HERE/'src/run_fp32_first.py'
GUARD = PROJECT/'src/run_g5_guarded_process.py'


def main():
    assert json.loads((HERE/'results/correctness_gates.json').read_text())['complete']
    frozen = json.loads((HERE/'artifacts/frozen_b_sources.json').read_text())
    artifacts = HERE/'artifacts'
    jobs = [('nsys_public_a0', 'fp32_compatibility_nsys_a0.json',
             ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
              '--force-overwrite=false', '-o', str(artifacts/'nsys_public_a0'),
              PYTHON, str(RUNNER), '--phase', 'compatibility', '--record-id', 'nsys_a0'])]
    for i, (rows, columns) in enumerate(((4096, 4096), (4096, 2656), (2656, 2656))):
        label = f'ncu_shape{i}_a0'
        jobs.append((label, f'fp32_trace_{label}.json',
                     ['ncu', '--nvtx', '--nvtx-include', f'G15_PEDANTIC_{rows}_{columns}_512/',
                      '--section', 'LaunchStats', '--section', 'SourceCounters',
                      '--export', str(artifacts/label), '--page', 'raw', '--csv',
                      PYTHON, str(RUNNER), '--phase', 'trace', '--record-id', label,
                      '--shape-index', str(i)]))
    records = []
    with open('/tmp/tensorjoin_gpu2_campaign.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(HERE/'results/precision_start.json', {'jobs': jobs, 'frozen': frozen, 'pid': os.getpid()})
        for label, result_name, child in jobs:
            for relative, digest in frozen.items():
                assert sha256_file(PROJECT/relative) == digest, relative
            full_label = 'g15b_' + label
            path = HERE/'results'/result_name
            command = [PYTHON, str(GUARD), '--label', full_label, '--expected-result',
                       str(path.relative_to(PROJECT)), '--physical-gpu', '2', '--'] + child
            print('PRECISION_START ' + json.dumps(command), flush=True)
            r = subprocess.run(command, check=False)
            guard_path = PROJECT/f'results/g5_guard_{full_label}.json'
            g = json.loads(guard_path.read_text()) if guard_path.exists() else {}
            record = {'label': label, 'command': command, 'returncode': r.returncode,
                      'guard_path': str(guard_path.relative_to(PROJECT)),
                      'admitted': bool(g.get('admitted')),
                      'result_path': str(path.relative_to(PROJECT))}
            records.append(record)
            if r.returncode or not record['admitted']:
                break
            if label.startswith('nsys'):
                export = ['nsys', 'export', '--type=sqlite', '--force-overwrite=false',
                          '--output', str(artifacts/(label+'.sqlite')),
                          str(artifacts/(label+'.nsys-rep'))]
                subprocess.run(export, check=True)
            else:
                for page, extra in [('source', ['--print-source', 'sass']), ('raw', [])]:
                    output = HERE/f'raw/{label}_{page}.csv'
                    export = ['ncu', '--import', str(artifacts/(label+'.ncu-rep')),
                              '--page', page, '--csv'] + extra
                    with output.open('x') as handle:
                        subprocess.run(export, check=True, stdout=handle)
    complete = len(records) == len(jobs) and all(r['admitted'] for r in records)
    atomic_json(HERE/'results/precision_collection.json', {'complete': complete, 'records': records,
                                                         'diagnostic_only': True,
                                                         'semantic_audit_still_required': True})
    return 0 if complete else 2


if __name__ == '__main__':
    main()
