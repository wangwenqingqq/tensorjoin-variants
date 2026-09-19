"""Execute exactly the four predeclared CPU preparation processes."""

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from g2b_public_common import atomic_json, sha256_file

HERE = Path(__file__).resolve().parents[1]
PYTHON = '@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
ORDER = ('original', 'chunked', 'chunked', 'original')


def main():
    if (HERE / 'results/cpu_campaign.json').exists():
        raise FileExistsError('Campaign already completed')
    records = []
    frozen = {str(p.relative_to(HERE)): sha256_file(p)
              for p in [HERE/'PLAN_AND_GATE0.md', HERE/'src/preparation.py',
                        HERE/'src/run_preparation_cpu.py', Path(__file__)]}
    atomic_json(HERE/'results/cpu_start.json', {'order': ORDER, 'frozen': frozen,
                                              'started_unix': time.time(), 'pid': os.getpid()})
    with open('/tmp/tensorjoin_gpu2_campaign.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for position, variant in enumerate(ORDER):
            rid = f'p{position}_{variant}_a0'
            command = ['numactl', '--cpunodebind=0', '--membind=0', PYTHON,
                       str(HERE/'src/run_preparation_cpu.py'), '--record-id', rid,
                       '--variant', variant]
            raw = HERE / f'raw/preparation_{rid}.log'
            raw.parent.mkdir(parents=True, exist_ok=True)
            with raw.open('x') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            record = {'position': position, 'variant': variant, 'command': command,
                      'returncode': process.returncode, 'raw': str(raw.relative_to(HERE)),
                      'raw_sha256': sha256_file(raw)}
            result_path = HERE/f'results/preparation_{rid}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                record.update(preparation_seconds=result['preparation_seconds'],
                              bitwise_equal=result['all_bitwise_equal'],
                              result=str(result_path.relative_to(HERE)), result_sha256=sha256_file(result_path))
            records.append(record)
            print('CPU_SLOT ' + json.dumps(record), flush=True)
            if process.returncode or not record.get('bitwise_equal'):
                break
    complete = len(records) == 4 and all(r.get('bitwise_equal') for r in records)
    ratios = ([records[0]['preparation_seconds']/records[1]['preparation_seconds'],
               records[3]['preparation_seconds']/records[2]['preparation_seconds']] if complete else [])
    atomic_json(HERE/'results/cpu_campaign.json', {'records': records, 'complete': complete,
                                                 'paired_original_over_chunked': ratios,
                                                 'integration_admitted': complete and min(ratios) >= 1.20,
                                                 'frozen': frozen})
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
