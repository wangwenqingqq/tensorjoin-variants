"""Serialize G15 full compatibility, stress and bounded sanitizer gates."""

import fcntl
import json
import os
import subprocess
from pathlib import Path

from g2b_public_common import atomic_json, sha256_file

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
PYTHON = '@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
FP32 = HERE/'src/run_fp32_first.py'
G5 = HERE/'src/run_chunked_g5.py'
GUARD = PROJECT/'src/run_g5_guarded_process.py'


def main():
    if (HERE/'results/correctness_gates.json').exists():
        raise FileExistsError('Gate campaign already exists')
    initial = json.loads((HERE/'results/fp32_validation_a1.json').read_text())
    assert initial['correctness']['exact_contract_pass']
    frozen = json.loads((HERE/'artifacts/frozen_b_sources.json').read_text())
    slots = [
        ('fp32_full_a0', 'fp32_compatibility_full_a0.json',
         [PYTHON, str(FP32), '--phase', 'compatibility', '--record-id', 'full_a0']),
        ('chunked_full_a0', 'chunked_compatibility_full_a0.json',
         [PYTHON, str(G5), '--phase', 'compatibility', '--record-id', 'full_a0']),
        ('fp32_stress_a0', 'fp32_stress_a0.json',
         [PYTHON, str(FP32), '--phase', 'stress', '--record-id', 'a0']),
        ('fp32_memcheck_a0', 'fp32_compatibility_memcheck_a0.json',
         ['compute-sanitizer', '--tool', 'memcheck', '--leak-check', 'no', '--error-exitcode', '99',
          PYTHON, str(FP32), '--phase', 'compatibility', '--record-id', 'memcheck_a0']),
        ('fp32_synccheck_a0', 'fp32_validation_synccheck_a0.json',
         ['compute-sanitizer', '--tool', 'synccheck', '--error-exitcode', '99',
          PYTHON, str(FP32), '--phase', 'validation', '--record-id', 'synccheck_a0']),
    ]
    records = []
    with open('/tmp/tensorjoin_gpu2_campaign.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(HERE/'results/correctness_start.json', {'slots': slots, 'frozen': frozen, 'pid': os.getpid()})
        for label, name, child in slots:
            for relative, digest in frozen.items():
                assert sha256_file(PROJECT/relative) == digest, relative
            result_path = HERE/'results'/name
            full_label = 'g15b_' + label
            command = [PYTHON, str(GUARD), '--label', full_label, '--expected-result',
                       str(result_path.relative_to(PROJECT)), '--physical-gpu', '2', '--'] + child
            print('GATE_START ' + json.dumps(command), flush=True)
            completed = subprocess.run(command, check=False)
            guard_path = PROJECT/f'results/g5_guard_{full_label}.json'
            record = {'label': full_label, 'command': command, 'returncode': completed.returncode,
                      'guard_path': str(guard_path.relative_to(PROJECT)),
                      'result_path': str(result_path.relative_to(PROJECT))}
            if guard_path.exists():
                g = json.loads(guard_path.read_text())
                record.update(admitted=g['admitted'], guard_sha256=sha256_file(guard_path))
            if result_path.exists():
                r = json.loads(result_path.read_text())
                record.update(result_sha256=sha256_file(result_path), correctness=r['correctness'],
                              public_seconds_diagnostic=r.get('public_seconds'))
            records.append(record)
            print('GATE_COMPLETE ' + json.dumps(record), flush=True)
            if completed.returncode or not record.get('admitted'):
                break
    complete = len(records) == len(slots) and all(r.get('admitted') for r in records)
    atomic_json(HERE/'results/correctness_gates.json', {'records': records, 'complete': complete,
                                                      'precision_audit_still_required': True})
    return 0 if complete else 2


if __name__ == '__main__':
    main()
