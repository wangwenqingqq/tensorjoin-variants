"""Prepend G18 routing, preserving each host's divergent historical records."""
import hashlib
import json
import socket
from pathlib import Path

H = Path(__file__).resolve().parents[1]
P = H.parent
block = '''## Latest evidence (G18: bounded two-sided RT-HiSS repair)

Read `g18_rthiss_conservative_repair_20260905/DECISION.md`, its
`CLAIM_EVIDENCE.md`, and `results/closure_checks.json`. The separate conservative
FP32-prefix / FP64-terminal repair now reproduces all frozen original IDs on
nine D512 inputs. CIFAR4096 returns all 262,144 directed IDs, zero missing/extra,
with only 184 of 16,777,216 candidates reaching FP64 (0.0010967255%). Both native
G17 error directions are corrected. The real-data <1% terminal-work screen passes;
boundary_zero32 needs 62/1024 terminals (6.0546875%), so selectivity is not universal.

Twenty main GPU3 slots pass: eighteen operator observations and two predicate
observations. Four bounded sanitizer runs, reversed fresh-process operator repeats
and 1000 two-buffer/stride predicate invocations pass. Predicate stress is not a
sustained full join. Runtime-selected refinement has 40 registers / 992 static
instructions; compression's 408-instruction stream matches G17. No speed follows.
All pairs remain RT candidates on this finite matrix; no general traversal
certificate, exact-real, Graph, leak-free lifecycle or 60K admission is established.

Wrong-D18 build, two occupancy blocks, the failed cross-run map-byte proxy and
one unadmitted native diagnostic guard remain retained. A3 explicitly replaces
the invalid snapshot proxy with checked actual input/maps and complete original
IDs; old failures are not rewritten. All 420 G17 evidence files and 42 old core /
manuscript files were hash-verified unchanged at closure. No foreign job was touched.

Next: `NEXT_COST_CONTRACT.md` in G18. Build a separately frozen diagnostic-free
same-output comparator, resolve reusable-engine versus fresh-invocation fairness,
then remeasure repaired RT-HiSS, TensorJoin and the strong FP32-first control on
the SAME shape and full host-input-to-canonical-host-output denominator. Do not
compare instrumented G18 timings or 4096 results to G16's old 60K screen.

G16's 1.499724/1.267190 early-screen ratios and prior negatives remain unchanged.
G18 admits a bounded repaired reference comparator, not external superiority,
novelty or paper readiness. A differentiated non-incremental thesis still needs
Gate 0 before drafting or expensive breadth. No manuscript edit/compile/visual
QA/Overleaf sync occurred. G13 remains LOCAL ONLY; host histories are independently
backed up and updated, never wholesale synchronized.

---

'''
exp = '''\n\n## G18 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G18 RT-HiSS conservative two-sided repair | A bounded FP32 prefix/full-sum predicate plus selective original-order FP64 terminal can restore the frozen reference IDs without full-FP64 candidate refinement. | G18 PROTOCOL / NUMERICAL_DESIGN / explicit A1-A3 addenda; nine D512 inputs; 20 GPU3 main slots; exact hashes, four sanitizer runs, predicate stress, selected-code and closure receipts. | Bounded same-output admission: zero missing/extra IDs; real4096 uses 184/16,777,216 FP64 terminals (0.0010967255%), boundary_zero32 6.0546875%. No complete-cost timing or novelty admission. All failures and old evidence retained; next diagnostic-free artifact plus explicit engine-lifecycle fairness. |
'''
claim = '''\n\n## G18 evidence supplement (2026-09-05; historical claims unchanged)

Route to `g18_rthiss_conservative_repair_20260905/CLAIM_EVIDENCE.md` (G18-C1--C9).
Nine-input reference-output equality, selective real-data terminal work and
bounded safety/code gates are measured. The arithmetic envelope is theoretical
under its explicit assumptions. Cross-run point-map byte invariance is rejected
as a correctness proxy, without relaxing input mapping or complete-ID equality.
General candidate certification, modern equal-output speed, reusable lifecycle
and novelty remain unestablished. G16's positive narrow screen and all historical
negative evidence remain intact; comparator repair is not TensorJoin novelty.
'''
label = 'remote' if str(P).startswith('/home/') else 'local'
backup = H / 'artifacts' / ('root_routes_before_g18_' + label)
targets = {name: (P / name).read_text() for name in
           ['CURRENT_STATUS.md', 'EXPERIMENTS.md', 'CLAIMS.md']}
for text in targets.values():
    assert '## G18 evidence supplement' not in text
    assert '## Latest evidence (G18:' not in text
marker = '## Latest evidence (G17: modern baseline pair-contract audit)'
assert marker in targets['CURRENT_STATUS.md']
assert 'Last verified: 2026-09-05 (G17 modern-baseline pair-contract closure).' in targets['CURRENT_STATUS.md']
backup.mkdir(exist_ok=False)
receipt = {}
for name, old in targets.items():
    (backup / name).write_text(old)
    if name == 'CURRENT_STATUS.md':
        new = old.replace('Last verified: 2026-09-05 (G17 modern-baseline pair-contract closure).',
                          'Last verified: 2026-09-05 (G18 bounded repaired-reference closure).', 1)
        new = new.replace(marker, block + '## Previous evidence (G17: modern baseline pair-contract audit)', 1)
    else:
        new = old + (exp if name == 'EXPERIMENTS.md' else claim)
    (P / name).write_text(new)
    receipt[name] = dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                         after_sha256=hashlib.sha256(new.encode()).hexdigest())
(H / 'artifacts' / ('root_route_update_' + label + '.json')).write_text(
    json.dumps(dict(host=socket.gethostname(), scope='Host-local additive route update; old history retained',
                    files=receipt), indent=2) + '\n')
print(json.dumps(receipt))
