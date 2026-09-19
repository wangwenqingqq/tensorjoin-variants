# G4A-R1 dynamic-count router attribution decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_g4a_r1_dynamic_count_router_gpu1`

## Decision

**PASS for shape-local, host-dispatched dynamic-router attribution.  On the
frozen `4096x512` G2A contract, the exact G3C-B-R1 three-stage pipeline is
2.195x faster than the unchanged exact G3B-R1 two-stage pipeline after charging
actual device-counter reads, their synchronizations, Python/Triton launches,
and dynamic stage extents.  The 95% process-bootstrap interval is
[2.159x, 2.213x], with 8/8 process-median wins.**

The separate 1,000-invocation host-dispatched sequences favor the candidate in
8/8 processes, with a 2.227x geometric-mean speedup.  Every admitted process
preserves the exact output and expected dynamic counts, and every fresh cache
contains the accepted G3B, G3C, and FP64 cubins.

This result closes G3D's fixed-count shortcut.  It remains a resident-input and
resident-output operator result; it is not the public end-to-end denominator.

## Frozen comparison and dynamic path

Both variants use the same frozen CIFAR-10-GIST `4096x512` float32 input,
radius, upper-triangle schedule, G3B-R1 certificate kernel, FP64 refinement
kernel, output capacity, and exact 133,120-pair upper-triangle output.

- Keeper: execute G3B-R1, read its actual ambiguity counter to the host, launch
  FP64 with that observed extent, and read the final count.
- Candidate: execute the identical G3B-R1 stage, read its actual ambiguity
  count, launch certified G3C-B-R1 FP32 with that extent, read its actual
  residual count, launch FP64 with that extent, and read the final count.

Expected counts are assertions rather than supplied launch inputs:

| Variant | G3B ambiguity | FP64 refinement | Final upper output |
|---|---:|---:|---:|
| G3B-R1 keeper | 102,079 | 102,079 | 133,120 |
| G3C-B-R1 candidate | 102,079 | 97 | 133,120 |

All retained observations and every sustained sequence reproduce these tuples.

## Timing result

Eight fresh processes use block orders `KC, CK, KC, CK, CK, KC, CK, KC`, 20
warmups, and 200 retained host-wall observations per variant.  `perf_counter_ns`
starts before the device counter reset and stops after the final count is
available on the host.

| Metric | G3B-R1 keeper | G3C-B-R1 candidate |
|---|---:|---:|
| Marginal p10 | 784.817 us | 352.629 us |
| Marginal median | 787.333 us | 356.213 us |
| Marginal p90 | 792.484 us | 371.294 us |

The decision statistic is the geometric mean of the eight paired process
median ratios, not the marginal ratio.  Individual ratios range from 2.074x to
2.216x.  Their geometric mean is 2.194639x; a deterministic 20,000-sample
process bootstrap with seed `20260903` gives [2.159275x, 2.213376x].  All eight
processes favor the candidate and the lower bound clears the frozen 1.25x gate.

The balanced order diagnostic does not reverse the result.  First-position and
second-position medians remain close for both variants, apart from one slower
candidate process already retained by the paired estimator.

## Sustained result

Every formal process separately runs 1,000 sequential, dynamically dispatched
complete pipelines per variant.  Per-process speedups range from 2.108x to
2.322x; the geometric mean is 2.226875x and the candidate wins 8/8.  This
passes the frozen 1.20x sustained gate.

This sequence includes the host/device scalar-count round trips on every
invocation.  It is not an asynchronous device-resident scheduler or a
multi-client service test.

## Correctness, binary identity, and isolation

Immediately before and after timing, every formal process reproduces:

- 133,120 sorted upper-triangle pair IDs;
- output SHA-256
  `036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959`;
- keeper dynamic counts `102079 -> 102079 -> 133120`; and
- candidate dynamic counts `102079 -> 97 -> 133120`.

The runtime audit finds exactly one accepted copy of each cubin in every fresh
process cache:

- G3B-R1 certificate `dd32e979...114ac`;
- G3C-B-R1 FP32 filter `db750295...20670`; and
- FP64 refinement `ff0572a7...477c50`.

All ten admitted G4A-R1 screen/formal processes ran on physical GPU1, UUID
`GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3`, under its own lock after 30
continuous empty seconds.  The 0.25-second monitor saw no foreign GPU1 process,
and every admitted process left an empty postflight compute state.  GPU clocks
and power policy were not changed.

Compilation identity, correctness, inherited memcheck/stress evidence,
per-invocation timing, and sustained timing remain separate gates.

## Preserved GPU0 interruptions and device revision

The original GPU0 G4A cheap screen passed with 2.254x paired and 2.280x
sustained geometric-mean speedups.  Three later formal campaign attempts were
excluded before formal summarization:

1. four clean slots followed by three prelaunch-blocked attempts when foreign
   `dense_n8` / `sparse_n8` workloads acquired GPU0;
2. one clean slot followed by three prelaunch-blocked attempts during a second
   external dense/sparse campaign; and
3. two clean slots plus one prelaunch-blocked attempt, followed by a run-time
   invalidation when a foreign SGLang scheduler acquired GPU0.

The isolation monitor terminated only the TensorJoin process group.  No
foreign process was killed or modified.  Each partial campaign, including its
clean-but-unused slots, blockage records, occupancy logs, and hashes, is
preserved under `rejected/` and excluded from the estimator.

`PROTOCOL_G4A_R1.md` was frozen before any GPU1 execution and changes only the
same-model physical device, UUID, lock, and evidence prefix.  It reruns both the
screen and the formal campaign from scratch; no GPU0 timing is mixed into the
GPU1 statistic.

## Claim boundary and next gate

Allowed wording:

**On the frozen `4096x512` G2A resident-input/output contract, the exact
host-dispatched precision router is 2.195x faster than refining every
first-stage ambiguity in FP64 after charging actual dynamic counter reads and
launch extents (95% process-bootstrap interval [2.159x, 2.213x], 8/8 wins).**

Not allowed:

- ingest-inclusive or host-output-materializing end-to-end wording;
- full CIFAR60K, cross-dimension, cross-density, or cross-dataset
  generalization;
- an asynchronous or device-resident scheduler claim;
- claiming that G4A-R1 proves the separately timed G2B implementation; or
- submission readiness.

The next paper-critical gate is the compact public breadth matrix.  It must
adapt the analytically certified dynamic router to multiple dimensions,
output-density regimes, and scales under the same complete-output denominator,
and must expose when routing overhead or residual work removes the benefit.

## Evidence

- Formal summary: `results/g4a_r1_formal_summary.json`, SHA-256
  `1a18a5ed84d21f116a28c7ed1400feef00cd6957f5a9603ba90a20e057ffd209`.
- Formal runtime audit: `results/g4a_r1_formal_runtime_audit.json`, SHA-256
  `437a87c385b4ba004da7d861064422b70da1a64e0cfbe8a3c3109c8c9b389799`.
- G4A-R1 screen summary: `results/g4a_r1_screen_summary.json`, SHA-256
  `3ab358cfa751de2ad98d34e2c9ce50e3f600524d5806af4a33feabafb2f319fc`.
- G4A-R1 protocol: `PROTOCOL_G4A_R1.md`, SHA-256
  `87e25bbadbb79c804013431af9fd1f30242bc192e4b5d4df93643aca01782fac`.
- Measurement runner: `src/run_g4a_r1_dynamic_router_timing_gpu1.py`, SHA-256
  `f06064fb891d310e4119aa125fa808319cf50159c6364cbc6d4fe51013b26db6`.
- Formal driver log: `raw/g4a_r1_formal_driver.log`, SHA-256
  `153c5f5ddc27321be7c1288800debbe0ddf0e7247df5b6a8453e4453aa800ba2`.
- Complete 708-file G4A/G4A-R1 evidence ledger:
  `receipts/g4a_r1_final_evidence_sha256.txt`, SHA-256
  `b42c41a24a3afe989c61c4ccef544881c7b522471dd5897ef7a2df597963f088`.
- Preserved interruption ledgers: `b65871b2...d6ed3`,
  `a761e612...0bfd6`, and `1f20c568...0de3`.
