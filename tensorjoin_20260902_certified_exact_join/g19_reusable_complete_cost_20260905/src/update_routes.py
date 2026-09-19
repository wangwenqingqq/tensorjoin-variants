"""Add G19 closure without mixing divergent local/remote historical routes."""
import hashlib
import json
import socket
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent
label='remote' if str(P).startswith('/home/') else 'local'
block='''## Latest evidence (G19: reusable comparison admitted; timing screen failed)

Read `g19_reusable_complete_cost_20260905/DECISION.md`, its `CLAIM_EVIDENCE.md`,
`results/closure_checks.json` and `results/screen.json`. The reusable,
diagnostic-free RT-HiSS comparator, TC path and owned-handle pedantic FP32
control are admitted on the nine D512 fixtures. Fourteen composite gates pass:
full exact-ID matrices, three clean full shutdown memchecks plus RT synccheck,
164 full alternating-input stress calls and three runtime profiles. R1--R3
repair resource lifetime without modifying shared OWL/PyTorch or old artifacts.
All original failed admissions remain failed and archived.

The complete 4096 host-input-to-canonical-host-ID screen does NOT pass its
predeclared every-block 1.10x strongest-control gate. FP32/TC ratios are
1.287224 / 1.059163; RT/TC ratios 11.217547 / 10.243591. Lower TC medians in both
blocks do not waive the second block's small margin. All 30 timed observations,
12 warm calls, source/cache hashes and six clean GPU guards remain. No favorable
rerun, CI, formal/sustained promotion or novelty pass. Do not advertise only the
large RT ratio while omitting the stronger FP32 control.

Actual cuBLAS selected-function SASS matches G16 after adopting explicit handle
ownership and the 8,519,680-byte chosen workspace. All 420 G17 files, 593 G18
files and 42 prior core/manuscript files were hash-verified unchanged. GPU3 was
idle at closure; foreign processes and clocks/power were untouched. This is
bounded reference equality, not a general exact-real or RT traversal guarantee.
The fixed-panel FP32 control also computes both within-panel triangles, so it
is not asserted to be the fastest possible FP32 blocking/SYRK implementation.

Next: `NEXT_RESEARCH_GATE.md` in G19. Retain the admitted comparator and stop
performance promotion/broad paper expansion. Identify a concrete non-incremental
work-elimination mechanism, check the closest current primary prior art, then
freeze its cheap falsification test. Existing certificates, ordinary tree/shape
routing and comparator cleanup do not by themselves supply a new contribution.
Do not blame CPU/NUMA/clock noise without causal evidence or pool G16's separate
60K positive screen with G19. No manuscript edit/compile/visual QA/Overleaf sync.
G13 remains LOCAL ONLY; route histories are backed up per host, not synchronized.

---

'''
exp='''\n\n## G19 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen evidence | Decision |
|---|---|---|---|
| G19 reusable complete-cost comparison | Data-independent engine reuse and explicit cleanup permit same-output full host-to-host comparison against repaired RT-HiSS and the faster pedantic FP32-first control. | PROTOCOL plus R1--R3 addenda; nine D512 fixtures; 14 composite admission slots; clean full shutdown checks; 164 full stress calls; actual-code binding; six timing processes / 30 observations. | Comparator implementation admitted. Every-block >=1.10x strongest-control screen rejected: FP32/TC 1.287224 / 1.059163, despite RT/TC 11.217547 / 10.243591. No favorable rerun or novelty/formal/sustained promotion. Preserve all failed ownership attempts and G16/G17/G18 evidence. |
'''
claim='''\n\n## G19 evidence supplement (2026-09-05; historical claims unchanged)

Route to `g19_reusable_complete_cost_20260905/CLAIM_EVIDENCE.md` (G19-C1--C9).
Reusable complete-ID equality, bounded full shutdown safety/stability and actual
numerical-code identity are measured. The every-block >=1.10x complete-cost
claim is rejected: the strongest-control ratios are 1.287224 / 1.059163. The
large RT-only ratios are partial screen observations, not evidence of general
external superiority or a reason to omit FP32. Host/NUMA/clock causality and
non-incremental novelty remain unknown. No paper promotion or manuscript edit.
'''
targets={n:(P/n).read_text() for n in ['CURRENT_STATUS.md','EXPERIMENTS.md','CLAIMS.md']}
for s in targets.values():assert '## G19 evidence supplement' not in s and '## Latest evidence (G19:' not in s
marker='## Latest evidence (G18: bounded two-sided RT-HiSS repair)'
old_date='Last verified: 2026-09-05 (G18 bounded repaired-reference closure).'
assert marker in targets['CURRENT_STATUS.md'] and old_date in targets['CURRENT_STATUS.md']
backup=H/'artifacts'/f'root_routes_before_g19_{label}';backup.mkdir(exist_ok=False)
receipt={}
for name,old in targets.items():
 (backup/name).write_text(old)
 if name=='CURRENT_STATUS.md':
  new=old.replace(old_date,'Last verified: 2026-09-05 (G19 complete-cost comparator closure; screen failed).',1)
  new=new.replace(marker,block+'## Previous evidence (G18: bounded two-sided RT-HiSS repair)',1)
 else:new=old+(exp if name=='EXPERIMENTS.md' else claim)
 (P/name).write_text(new)
 receipt[name]=dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=hashlib.sha256(new.encode()).hexdigest())
(H/'artifacts'/f'root_route_update_{label}.json').write_text(json.dumps(dict(host=socket.gethostname(),scope='Host-local additive update; historical text retained',files=receipt),indent=2)+'\n')
print(json.dumps(receipt))
