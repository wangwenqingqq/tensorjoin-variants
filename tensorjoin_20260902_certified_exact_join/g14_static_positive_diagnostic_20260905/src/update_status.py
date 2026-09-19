"""Add the completed G14 route without overwriting divergent prior histories."""

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
ROUTE = '''## Current continuation decision (G14: return to static positives)

Read `g14_static_positive_diagnostic_20260905/DECISION.md` and its
`CLAIM_EVIDENCE.md`. The bounded static-join diagnostic is complete on
gpu-host-8 GPU 2: all 12 independent processes pass the frozen exact-output
and occupancy gates. All four G5 selected cubin/PTX sets match old P4.

G5 measures 0.911–0.951 s; freshly measured MiSTIC is 4.742–5.482 s, with
same-block diagnostic time ratios of 4.99–5.88. Preparation, combined
layout/H2D/allocation, and canonicalization account for 94.12–94.54% of G5
wall time. The original 5.285 s slow mode did not recur and its cause remains
unresolved. No old result or failed gate has been reclassified.

Continue from these positive assets: first split the measured preparation
costs and evaluate a separately frozen data-path improvement; add a complete
adaptive-FP32-first comparator before attributing a TC-specific advantage.
No such new variant was run. Do not expand datasets, replace the story, or
promote instrumented diagnostic ratios into formal/sustained paper claims.
Generic certification novelty remains unproven; data-path engineering alone
does not resolve the existing prior-art overlap.

Core sources are unchanged. No manuscript/Overleaf edit. GPU 1's foreign
process was untouched and GPU 2 is idle at postflight. G14 source snapshots,
logs, phase records and caches are retained on 8p and copied to the local
project; large MiSTIC pair files and executable remain on 8p with verified
hash receipts. G13 remains LOCAL ONLY; remote and local pre-G14 status
histories are preserved separately rather than overwritten by synchronization.

Evidence: `g14_static_positive_diagnostic_20260905/results/campaign.json`,
`results/summary.json` inside that directory, and
`artifacts/raw_evidence_manifest.json` inside that directory.
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', choices=('local', 'remote'), required=True)
    args = parser.parse_args()
    receipt = {}
    for name in ('CURRENT_STATUS.md', 'EXPERIMENTS.md', 'CLAIMS.md'):
        path = PROJECT / name
        before = path.read_bytes()
        old = before.decode()
        if 'G14' in old:
            raise RuntimeError(f'G14 already present: {name}')
        backup = HERE / 'artifacts' / f'{name}.{args.site}_before_G14'
        with backup.open('xb') as handle:
            handle.write(before)
        if name == 'CURRENT_STATUS.md':
            boundary = old.index('## Current continuation decision')
            history = old[boundary:].replace('## Current continuation decision',
                                             '## Previous continuation decision', 1)
            after = ('# Current TensorJoin Continuation Route\n\n'
                     'Last verified: 2026-09-05 (G14, 12-process static-join diagnostic).\n\n'
                     + ROUTE + '\n---\n\n' + history)
        elif name == 'EXPERIMENTS.md':
            after = old + '''

## G14 static-positive diagnostic (2026-09-05)

| Experiment | Hypothesis | Contract | Keeper/candidate | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---|---|---|---|
| `tensorjoin_20260905_static_positive_g14_cifar60k` | Phase timing and fixed placement can distinguish a persistent audited-runner cost from an unlocalized historical slowdown. | `g14_static_positive_diagnostic_20260905/PROTOCOL.md`; 12 independent processes, default/node-0 ABBA placement, original 60K host-to-canonical-host output. | Fresh MiSTIC and additive G2B/G5 diagnostic adapters; unchanged kernels. | Zero change in routing work versus each original. | Exact 3,926,078-ID hash, GPU isolation, G5 P4 binary identity, phase closure. | G14 `results/campaign.json`, `results/summary.json`, `raw/campaign.log`, `artifacts/raw_evidence_manifest.json`. | Diagnostic complete: all 12 admitted; G5 0.911–0.951 s and 4.99–5.88 same-block MiSTIC/G5 ratios; preparation/movement/sort 94.12–94.54%. Historical slow-mode cause inconclusive; no novelty/formal promotion. |
'''
        else:
            after = old + '''

## G14 diagnostic supplement (2026-09-05; C35 unchanged)

See `g14_static_positive_diagnostic_20260905/CLAIM_EVIDENCE.md` for G14-C1
through G14-C5. The completed same-output phase diagnostic supplies new positive
instrumented observations, not a rewrite of C35's original failed gate. It
supports continued static-join data-path attribution, not a novelty pass,
production speedup, NUMA root-cause claim or automatic formal campaign.
'''
        if path.read_bytes() != before:
            raise RuntimeError(f'Concurrent edit detected: {name}')
        temporary = path.with_name(f'.{name}.g14.{os.getpid()}.tmp')
        with temporary.open('x') as handle:
            handle.write(after)
        os.replace(temporary, path)
        receipt[name] = {'before_sha256': digest(before), 'after_sha256': digest(after.encode()),
                         'backup': str(backup.relative_to(PROJECT))}
    target = HERE / 'artifacts' / f'status_update_{args.site}.json'
    with target.open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
