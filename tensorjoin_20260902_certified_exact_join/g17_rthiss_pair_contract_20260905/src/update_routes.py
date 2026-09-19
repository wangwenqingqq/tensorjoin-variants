"""Add a G17 route without synchronizing divergent local/remote history."""
import hashlib,json,socket
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent
block='''## Latest evidence (G17: modern baseline pair-contract audit)

Read `g17_rthiss_pair_contract_20260905/DECISION.md`, `CLAIM_EVIDENCE.md`, and
`results/closure_checks.json`. The separate RT-HiSS adapter now returns checked
original-row IDs on nine frozen D512 inputs. All sixteen adapter runs pass
independent decoder/permutation/capacity checks. On CIFAR4096 the native and
FP64-reference counts are both 262,144, but there are 2 missing and 2 extra
**directed IDs**. All disputed pairs are candidates; scalar fmaf and same-launch
CUDA FP32/FP64 replay reproduce the difference. Six access/synchronization
runs are clean, without turning numerical failures into exact-output passes.
Runtime-observed adapter refinement/compression SASS matches the native-source
control build. OptiX driver-JIT identity and public latency remain unestablished.

G16's positive 1.499724/1.267190 full host-to-host early-screen ratios below are
preserved. G17 does not supply a new speedup, novelty approval or paper revision.
Keep RT-HiSS as a modern comparator: next independently design a conservative
**two-sided** predicate repair, retain the efficient native path, and charge all
repair/output costs. Do not only rerank positives, loosen the reference or drop
the baseline. `NEXT_BASELINE_REPAIR.md` freezes that next admission boundary;
it is not an implemented repair. A differentiated non-incremental thesis still
needs its own Gate 0 before drafting or expensive broad experiments.

Original sources and historical failures remain unchanged. No Overleaf edit,
compile, visual QA or synchronization occurred. G13 remains LOCAL ONLY. Root
status histories are backed up and updated independently on each host, not
wholesale synchronized. G7's old pending-export route below is superseded only
by G17's bounded adapter evidence; equal-output/performance admission is pending.

---

'''
exp='''\n\n## G17 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G17 RT-HiSS pair-contract admission | Preserve the native algorithm while exporting and independently auditing complete original IDs. | `g17_rthiss_pair_contract_20260905/PROTOCOL.md`, additive zero32 protocol, frozen hashes, 16 adapter runs, 1 same-launch diagnostic, code/safety/closure receipts. | Bounded decoder/structure/safety/code checks pass. Native CIFAR4096 has 2 FN + 2 FP despite equal counts; zero32 has 2 FP. Numeric failures retained. No exact-reference, performance or novelty promotion; separately designed two-sided repair next. |
'''
claim='''\n\n## G17 evidence supplement (2026-09-05; historical claims unchanged)

Route to `g17_rthiss_pair_contract_20260905/CLAIM_EVIDENCE.md`. Modern RT-HiSS
original-ID export is structurally validated; native FP32 output fails frozen
FP64 equality on one real and one additive boundary input. Candidate/decoder
checks, same-launch numerical replay and selected CUDA-code identity localize
the discrepancy without disqualifying the system. No comparative-performance
or novelty claim is admitted. Preserve G16's narrow positive and all prior
negative evidence; two-sided baseline repair remains designed, not measured.
'''
label='remote' if str(P).startswith('/home/') else 'local'
backup=H/'artifacts'/('root_routes_before_g17_'+label);backup.mkdir(exist_ok=False)
receipt={}
for name in ['CURRENT_STATUS.md','EXPERIMENTS.md','CLAIMS.md']:
 p=P/name;text=p.read_text();assert '## G17 evidence supplement' not in text and '## Latest evidence (G17:' not in text
 (backup/name).write_text(text);receipt[name]={'before_sha256':hashlib.sha256(text.encode()).hexdigest()}
 if name=='CURRENT_STATUS.md':
  text=text.replace('Last verified: 2026-09-05 (G15/G16 strong-control and GPU-preparation screens).','Last verified: 2026-09-05 (G17 modern-baseline pair-contract closure).',1)
  marker='## Current continuation decision (G15/G16: strong control and GPU preparation)'
  assert marker in text
  text=text.replace(marker,block+'## Previous continuation decision (G15/G16: strong control and GPU preparation)',1)
 elif name=='EXPERIMENTS.md':text+=exp
 else:text+=claim
 p.write_text(text);receipt[name]['after_sha256']=hashlib.sha256(text.encode()).hexdigest()
(H/'artifacts'/('root_route_update_'+label+'.json')).write_text(json.dumps(dict(host=socket.gethostname(),scope='Host-local route update; divergent old history retained',files=receipt),indent=2)+'\n')
print(json.dumps(receipt))
