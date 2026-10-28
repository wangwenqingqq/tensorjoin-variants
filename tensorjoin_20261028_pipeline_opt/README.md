# TensorJoin 60K host/pipeline optimization

This additive experiment preserves the archived baseline and all input data.
It is an engineering optimization, not a new research novelty or exact-real claim.
New task-owned sources, the exact task-supplied baseline runner, and curated numeric evidence are added here; this
change does not redistribute the archival collection or change any visibility
or license. Existing baseline rights and notices remain unchanged.

## Result and important caveat

The selected candidate combines bounded (1024-row) NumPy preprocessing with
same-stream device-count submission. Both shared host sorts remain unchanged.
All measured outputs contain 3,926,078 directed IDs and SHA-256
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.

| Frozen population | A median E2E | B median E2E | A/B | Wins |
| --- | ---: | ---: | ---: | ---: |
| One process, five repetitions each, AB/BA alternating | 0.793191 s | 0.406352 s | 1.951979x | 5/5 |
| Eight fresh paired processes, alternating AB/BA | 1.173279 s | 0.412713 s | 2.842845x | 8/8 |

The formal ratio-of-medians paired-bootstrap 95% CI is [1.795947, 3.407502]
(seed 20261028, 10000 process-pair resamples). The formal paired-geometric ratio
is separately 2.480123x. Join excluding shared canonicalization improves from
1.072662 to 0.310089 s (3.459206x) in those eight paired processes.

**Order/host effects are material:** formal baseline-first ratio is 3.321790x,
candidate-first is 1.782210x. These are not interchangeable with the warmed
single-process screen or the historical 0.783704 s observation. CPU allocation,
cache state and shared-host contention are plausible explanations, not isolated
causes. Both directions favor B, but this is not a 2.84x INT8-kernel speedup.
The bottleneck remains CPU preprocessing; sorting's relative share increases.

## Hardware and software

Measured GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability
12.0; generated PTX target sm_120a; driver 590.48.01, nvcc 13.1.115, PyTorch
2.11.0+cu130, Triton 3.6.0, Python 3.12.3 and NumPy 2.3.5. Shared host: two Intel
Xeon Platinum 8575C sockets, 48 cores/socket, 2 threads/core (192 logical CPUs).
Clocks/power were not changed, no CPU affinity or host-exclusive allocation was
claimed. This host context matters because the principal change is CPU-side.

## Attribution and implementation

Stage0 instrumented n=3 median was 1.290030 s: host preprocessing 901.761 ms,
transpose 199.633 ms (nested within H2D/workspace), host canonicalization
118.653 ms. CUDA-event ranges are stream envelopes, not pure device-active time.
The original uninstrumented runner was separately checked at 1.291170 s (n=1).
Stage1 n=3 ablations and all raw observations are retained under `results/`
and `raw/samples.ndjson`.

- The original stage1 already fused certificate and queue compaction; no new
  compaction-fusion improvement is claimed.
- Fixed grids (16384/1024 programs) use device-count grid-stride loops, not
  empirical candidate limits. Same-stream kernel boundaries publish queues.
- Graph capture, scalar updates and setup remain timed. Graph is implemented
  and tested but not selected: no stable incremental benefit over no-bubble.
- GPU sorting was not triggered: Stage0 did not establish sorting as dominant.
- Full-input codes/scales/errors/norms/transpose are byte-identical to the
  unchanged NumPy formulas. Threshold, guard and FP64 pair reduction are unchanged.

The task-supplied baseline runner is included for exact source provenance. Its
only difference from the archived runner is accepting one numeric visible GPU
instead of requiring physical GPU0; the timed arithmetic is unchanged. The four
other imported baseline helpers match the archive byte-for-byte.

## Reproduce in a fresh workspace

Do not run against or overwrite the historical experiment. Assemble a separate
workspace from the existing corresponding baseline sources and these additions:

```sh
# Run from the repository root; WORK must be a new absolute directory.
WORK=/configured/new-workspace
mkdir "$WORK" && mkdir "$WORK/src" "$WORK/results" "$WORK/raw"
BASE=tensorjoin_20260902_certified_exact_join
for f in run_g2b_tensorjoin_public.py g2b_public_common.py run_d1_triton.py \
         run_g2a_tensorjoin.py run_g2a2_tensorjoin_triangular.py; do
    cp "$BASE/src/$f" "$WORK/src/$f"
done
cp tensorjoin_20261028_pipeline_opt/src/*.py "$WORK/src/"
cp "$BASE/results/g2b_smoke_summary.json" "$WORK/results/"
ln -s /configured/read-only-data "$WORK/data"
cd "$WORK"
# Inspect users/processes first; this frozen runner intentionally targets GPU4.
# PY is the configured torch 2.11.0+cu130 / Triton 3.6.0 Python environment.
flock -n .campaign.lock env CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src \
  "$PY" src/test_optimization.py
flock -n .campaign.lock env CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src \
  "$PY" src/bench.py --record-id new_ab --variants baseline,chunked_nobubble \
  --repeats 5 --ab
"$PY" src/summarize_campaign.py --output results/new_ab_summary.json results/new_ab.json
```

File IO, context initialization and tiny-fixture JIT are excluded; all real
input preprocessing, H2D, allocations, graph setup if selected, execution, D2H
and canonical output are included. Each timed call recomputes method state.
No output cache survives. Count/hash and upper/overflow checks run after timing.
Foreign GPU4 compute PIDs fail admission; before/after snapshots cannot exclude
brief intervening contention or shared CPU interference. Never kill other users.

Formal protocol: eight sequential fresh Python processes, each one AB or BA
pair; alternate direction. Summarize their eight JSON files together. Keep both
public and join-excluding-shared-canonicalization denominators. Do not combine
the screen and formal observations.

## Validation and evidence scope

Full 60K candidate memcheck: zero errors. High-entropy/empty/ragged/capacity
fixtures: memcheck/synccheck/initcheck zero errors; racecheck zero errors and
warnings; three pointer-churn fixtures and 90 Graph replays pass. Static math
ASTs preserve the original per-pair formulas. Stage1 resources remain 140
registers/16384 B shared; stage2/3 dynamic loops use 38 registers and zero spills.
Generated-code hashes/resources are retained, not vendor/library mechanism claims.
An independent source/raw-log/record audit found CLEAN within its stated scope;
no complete immutable conversation export was available for that audit.

Publication strips hostnames, device UUIDs, executable paths, PIDs and unrelated
process inventories. Numeric observations and correctness checks are unchanged;
`PROVENANCE.json` distinguishes original and curated evidence. Bulk data,
checkpoints, raw conversations, opaque profiling binaries and device credentials
are not included. No claim is made beyond the frozen full 60K workload.
