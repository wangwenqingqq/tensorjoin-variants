"""Append completed G15/G16 evidence without merging divergent old histories."""
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent

def sha(x):return hashlib.sha256(x).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--site',choices=('local','remote'),required=True);args=parser.parse_args()
    screen=json.loads((HERE/'results/public_screen.json').read_text());assert screen['complete']
    assert json.loads((HERE/'results/closure_checks.json').read_text())['all_public_identity_gates_pass']
    if not screen['narrow_engineering_screen_pass']:raise RuntimeError('Write a failed-screen route instead; do not reuse this positive route')
    p=screen['pairs'];ratios=[r['fastest_control_over_tc'] for r in p]
    tc=[r['times']['gpu_g5'] for r in p]; fp=[r['times']['fp32_gpu'] for r in p]
    route=f'''## Current continuation decision (G15/G16: strong control and GPU preparation)

Read `g16_gpu_preparation_20260905/DECISION.md`, `CLAIM_EVIDENCE.md`,
`PAPER_READINESS.md`, and G15's corresponding decision first. G15 completed a
proper host-to-host pedantic FP32-first control and retained its adverse result:
CPU-prepared TC paths lose to that control. CPU blocking helps G5 by only
1.23–1.27x end-to-end; this does not establish TC necessity.

G16 then moved the measured preparation work to the GPU, without changing the
three G5 stages, and gave the FP32 control the same GPU-norm opportunity.
Both predeclared reversed-order blocks pass the narrow >=1.25x screen against
the faster admitted control. GPU-G5 times are {tc[0]:.9f}/{tc[1]:.9f} s;
GPU-prepared FP32 is {fp[0]:.9f}/{fp[1]:.9f} s; same-block ratios are
{ratios[0]:.6f}/{ratios[1]:.6f}. All six timed outputs match the same
3,926,078-ID hash, selected code is stable, and source/work/capacity/occupancy
gates pass. These are complete pageable-host-to-canonical-host **early-screen**
observations, not formal confidence, tails, sustained60K or multi-dataset claims.

Both metadata constructors pass all60K plus six adversarial fixtures; real
quantized metadata is byte-identical to the original CPU construction.
Subnormal quantized code/scale bytes differ in the recorded tiny fixture, with
no residual enclosure violation: do not claim universal byte identity. Full
access memchecks, bounded syncchecks and separate1000-call pointer-churn tests
pass. G16 FP32's actual runtime-selected cuBLAS function matches G15's no-MMA
SIMT function for all three shapes; original G5 stage cubins/work also match.
No exact-real, leak-free shutdown, Graph or arbitrary-stream claim is admitted.

**Research boundary remains:** GPU preprocessing is an attributed engineering
improvement, not a novelty pass. Generic certification/precision escalation has
prior art. Next resolve a differentiated non-incremental mechanism and admit
modern same-output RT-HiSS/repaired-FaSTED/COSS comparisons with their exact
artifact boundaries. Do not add a conventional tree, broaden data or rewrite
the story merely to disguise a subsumed mechanism. Missing evidence is unknown,
not automatic rejection of all TC join work.

All original core sources, old failed gates and historical manuscript remain
unchanged. No Overleaf edit/compile/sync occurred. G13 stays LOCAL ONLY; local
and remote old status histories are backed up and preserved separately. See
`artifacts/raw_evidence_manifest.json` for G15/G16 raw logs, receipts, frozen
sources, caches and remote-only profiler hashes; deployment-only failed attempts
and the earlier failed SASS normalizer are preserved, not performance reruns.
'''
    receipts={}
    for name in ('CURRENT_STATUS.md','EXPERIMENTS.md','CLAIMS.md'):
        path=PROJECT/name;before=path.read_bytes();old=before.decode()
        if 'G15/G16: strong control' in old or 'G15/G16 evidence supplement' in old:raise RuntimeError('Already updated: '+name)
        backup=HERE/'artifacts'/f'{name}.{args.site}_before_G15_G16'
        with backup.open('xb') as f:f.write(before)
        if name=='CURRENT_STATUS.md':
            boundary=old.index('## Current continuation decision')
            history=old[boundary:].replace('## Current continuation decision','## Previous continuation decision',1)
            after='# Current TensorJoin Continuation Route\n\nLast verified: 2026-09-05 (G15/G16 strong-control and GPU-preparation screens).\n\n'+route+'\n---\n\n'+history
        elif name=='EXPERIMENTS.md':
            after=old+f'''\n\n## G15/G16 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G15 strong control | CPU blocking removes preparation overhead; compare TC fairly with a complete FP32-first operator. | `g15_strong_control_20260905/PLAN_AND_GATE0.md`, `BASELINE_DESIGN.md`, results cpu/correctness/precision/public_screen, DECISION/CLAIM_EVIDENCE. | CPU integration local win; complete FP32-first decisively beats CPU-prepared TC. Preserve the failed TC-specific gate. |
| G16 GPU preparation | Build the unchanged G5 metadata on GPU, and grant the FP32 control equal GPU preparation. | `g16_gpu_preparation_20260905/PROTOCOL.md`, `ADDENDUM_FAIR_GPU_NORMS.md`, results metadata/safety/precision/public_screen/closure_checks, raw_evidence_manifest. | Narrow same-contract screen passes {ratios[0]:.6f}/{ratios[1]:.6f} versus faster control. No novelty/formal/sustained/multidataset promotion. |
'''
        else:
            after=old+'''\n\n## G15/G16 evidence supplement (2026-09-05; historical claims unchanged)

Route to `g15_strong_control_20260905/CLAIM_EVIDENCE.md` and
`g16_gpu_preparation_20260905/CLAIM_EVIDENCE.md`. G15's strong-control loss is
retained. G16 adds a bounded positive same-contract screen after equally
optimizing both preparation paths and completing new code/safety gates. Neither
rewrites C35's old failure or establishes generic novelty/exact-real semantics,
modern external dominance, formal confidence, sustained60K or generality.
'''
        assert path.read_bytes()==before,'Concurrent edit: '+name
        tmp=path.with_name('.'+name+f'.g16.{os.getpid()}.tmp')
        with tmp.open('x') as f:f.write(after)
        os.replace(tmp,path)
        receipts[name]=dict(before_sha256=sha(before),after_sha256=sha(after.encode()),backup=str(backup.relative_to(PROJECT)))
    with (HERE/'artifacts'/f'status_update_{args.site}.json').open('x') as f:json.dump(receipts,f,indent=2);f.write('\n')
    print(json.dumps(receipts))
if __name__=='__main__':main()
