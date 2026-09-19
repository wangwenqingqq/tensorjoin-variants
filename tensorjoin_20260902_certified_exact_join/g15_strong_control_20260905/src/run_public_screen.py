"""Run the two predeclared three-method G15 blocks without replacements."""

import fcntl
import json
import os
import subprocess
from pathlib import Path

from g2b_public_common import atomic_json, sha256_file

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
PYTHON = '@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
ORDERS = (('original', 'fp32', 'chunked'), ('chunked', 'fp32', 'original'))


def main():
    assert json.loads((HERE/'results/correctness_gates.json').read_text())['complete']
    assert json.loads((HERE/'results/precision_audit.json').read_text())['precision_gate_pass']
    assert json.loads((HERE/'results/cpu_campaign.json').read_text())['integration_admitted']
    frozen = json.loads((HERE/'artifacts/frozen_b_sources.json').read_text())
    records = []
    with open('/tmp/tensorjoin_gpu2_campaign.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(HERE/'results/public_start.json', {'orders': ORDERS, 'pid': os.getpid(), 'frozen': frozen,
                                                      'orchestrator_sha256': sha256_file(Path(__file__))})
        for block, order in enumerate(ORDERS):
            for position, method in enumerate(order):
                for relative, digest in frozen.items():
                    assert sha256_file(PROJECT/relative) == digest, relative
                rid = f'g15_b{block}_p{position}_{method}_a0'
                label = rid
                if method == 'original':
                    runner = PROJECT/'src/run_g5_tensorjoin_public.py'
                    result_path = PROJECT/f'results/g5_screen_tensorjoin_{rid}.json'
                elif method == 'chunked':
                    runner = HERE/'src/run_chunked_g5.py'
                    result_path = HERE/f'results/chunked_screen_{rid}.json'
                else:
                    runner = HERE/'src/run_fp32_first.py'
                    result_path = HERE/f'results/fp32_screen_{rid}.json'
                command = [PYTHON, str(PROJECT/'src/run_g5_guarded_process.py'), '--label', label,
                           '--expected-result', str(result_path.relative_to(PROJECT)), '--physical-gpu', '2', '--',
                           PYTHON, str(runner), '--phase', 'screen', '--record-id', rid]
                print('PUBLIC_START '+json.dumps(command), flush=True)
                r = subprocess.run(command, check=False)
                gpath = PROJECT/f'results/g5_guard_{label}.json'
                guard = json.loads(gpath.read_text()) if gpath.exists() else {}
                row = {'block': block, 'position': position, 'method': method, 'command': command,
                       'returncode': r.returncode, 'admitted': bool(guard.get('admitted')),
                       'guard_path': str(gpath.relative_to(PROJECT)), 'result_path': str(result_path.relative_to(PROJECT))}
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    row.update(public_seconds=result['public_seconds'], result_sha256=sha256_file(result_path),
                               exact_contract_pass=result['correctness']['exact_contract_pass'])
                records.append(row)
                atomic_json(HERE/f'results/public_b{block}_p{position}.json', row)
                print('PUBLIC_COMPLETE '+json.dumps(row), flush=True)
                if r.returncode or not row['admitted']:
                    break
            if r.returncode or not records[-1]['admitted']:
                break
    complete = len(records) == 6 and all(r['admitted'] for r in records)
    pairs = []
    if complete:
        for block in range(2):
            times = {r['method']: r['public_seconds'] for r in records if r['block'] == block}
            pairs.append({'block': block, 'times': times, 'original_over_chunked': times['original']/times['chunked'],
                          'fp32_over_chunked': times['fp32']/times['chunked'],
                          'fp32_over_original': times['fp32']/times['original']})
    result = {'complete': complete, 'records': records, 'pairs': pairs,
              'chunked_integration_gate_pass': complete and min(p['original_over_chunked'] for p in pairs) >= 1.15,
              'tc_specific_screen_pass': complete and min(p['fp32_over_chunked'] for p in pairs) >= 1.25,
              'formal_or_sustained_promotion': False}
    atomic_json(HERE/'results/public_screen.json', result)
    print('PUBLIC_SUMMARY '+json.dumps(result), flush=True)
    return 0 if complete else 2


if __name__ == '__main__':
    main()
