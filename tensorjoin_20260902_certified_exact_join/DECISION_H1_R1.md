# Decision: H1-R1 Exact GPU Multi-Vector Join Screen

Date: 2026-09-03

## Decision

**PASS the frozen H1 screen and admit H2.**  On the two frozen learned
audio/video multi-vector workloads, the exact object-first INT8 certificate
operator reproduces the independent CPU and exhaustive GPU keeper output,
passes bounded Compute Sanitizer coverage, and exceeds the 1.25x median screen
gate in every threshold cell.

This is strong prototype evidence for the upper candidate, not yet a database
or systems paper result.  The comparator is exhaustive direct FP64, the object
matrix is only 128x512, and the run is resident-GPU and single-process.

## Frozen screen results

The public denominator is outer operator wall time, including the dynamic
ambiguity-count synchronization and final canonical-ID transfer/sort.
Each cell uses 10 warmups and 50 retained alternating KC/CK observations.

| Dataset | Target/query | Ambiguous objects | Valid FP64 token fraction | Keeper median | Candidate median | Speedup | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESC-50/PANNs D2048 | 1 | 70 / 65,536 (0.1068%) | 0.1068% | 35.693 ms | 0.549 ms | 65.063x | 50/50 |
| ESC-50/PANNs D2048 | 8 | 358 / 65,536 (0.5463%) | 0.5463% | 35.703 ms | 0.953 ms | 37.477x | 50/50 |
| UCF101/R3D-18 D512 | 1 | 94 / 65,536 (0.1434%) | 0.1567% | 9.860 ms | 0.457 ms | 21.570x | 50/50 |
| UCF101/R3D-18 D512 | 8 | 822 / 65,536 (1.2543%) | 1.2936% | 9.864 ms | 0.720 ms | 13.707x | 50/50 |

The candidate p10/median/p90 wall-time ranges are respectively
0.530/0.549/0.591 ms, 0.935/0.953/0.998 ms,
0.448/0.457/0.484 ms, and 0.714/0.720/0.738 ms.  Every retained output hash is
invariant and equal to the frozen oracle hash.

## Exactness and safety

Across all four cells:

- zero token or object interval-containment violations;
- zero unsafe direct accepts or rejects;
- zero candidate, keeper, or candidate/keeper output mismatches;
- zero duplicate IDs, invalid scores, or capacity overflows;
- keeper-to-CPU maximum object-score error at most
  `1.1102230246251565e-15`;
- four bounded actual-data Compute Sanitizer runs (keeper/candidate x
  D2048-fixed/D512-ragged) report `ERROR SUMMARY: 0 errors`.

The generated-code audit finds 16 static `IMMA.16832.S8.S8.SAT` instructions
in every captured token-certificate cubin.  `cuobjdump` reports 95--96
registers, zero stack, and zero local bytes for these instances.  This proves
that the low-precision scan lowered to signed INT8 Tensor Core MMA in the
observed SM120 binaries; it is mechanism evidence, not a latency claim by
itself.

## Material caveats

1. Exhaustive direct FP64 is intentionally the H0/H1 keeper but is not the
   strongest practical exact comparator.  A certified FP32/TF32 scan with
   FP64 boundary repair could erase much of the reported advantage.
2. The object matrix is small and resident.  No ingest, preprocessing,
   out-of-core scheduling, tree traversal, multi-GPU, or concurrent workload is
   included.
3. Only two pretrained representations, 4--7 tokens/object, two output
   densities, one GPU, one process, and one toolchain are measured.
4. The 50-observation screen has no process-level confidence interval or
   sustained-loop gate.  It is an admission screen, not formal timing.
5. The nearest-prior-art search remains conditional.  H1 does not justify
   “first exact multi-vector join” wording.

## H2 kill test

Before tree integration or paper drafting, freeze and measure on the same
contract:

1. a direct-FP32-difference interval scan with certified FP64 repair;
2. a GEMM/TF32 or cuBLAS-based exact filter/refine baseline where a sound error
   interval can be established;
3. the current INT8 object-first candidate;
4. larger, still disjoint object matrices from the same public caches, charging
   streaming/materialization and final IDs.

H2 passes only if the candidate remains exact and decisively faster than the
fastest exact baseline.  If it passes, use the tree-index background only as an
object-level scheduling layer that reduces token panels and interval
materialization at larger scale.  If it loses to certified FP32/TF32, reject
the upper thesis rather than rescuing it with packaging.

## Evidence

- Correctness: `results/h1_r1_multivector_correctness.json`, SHA-256
  `06461064d1b237aaa29f4face0eb3c223f5487abf478ddcf5fccb353ed28dca4`.
- Memcheck summary: `results/h1_r1_multivector_memcheck_summary.json`, SHA-256
  `f022cc90db19b295dd481a5b7946f3690b0cd935ba398befb31dab8218305d09`.
- Timing screen: `results/h1_r1_multivector_timing_screen.json`, SHA-256
  `c98d0e33c95ff1707caee5a1b37cd345fbbb1dabe239644a1f82582b8c1f3173`.
- Generated-code audit: `raw/h1_r1_generated_code_audit.txt`, SHA-256
  `da4c39f5981583f3072350cdb3d301906563df1a6c37d5c00c7fb0723ddcea1b`.
- Runner SHA-256 `11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e`.
- Kernel source SHA-256
  `c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db`.
- Target: `gpu-host-8`, physical GPU3, RTX PRO 6000 Blackwell Server Edition,
  compute capability 12.0.

