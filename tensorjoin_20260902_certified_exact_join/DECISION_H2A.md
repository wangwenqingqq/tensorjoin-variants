# Decision: H2A Certified FP32 Strong-Baseline Screen

Date: 2026-09-03

## Decision

**PASS H2A and admit H2B.**  The accepted H1-R1 object-first INT8 Tensor Core
candidate remains exact and is 2.758x--10.455x faster than a same-contract
certified direct-FP32-difference baseline across all four frozen learned
audio/video cells.  Every cell passes the predeclared 1.25x outer-wall gate
and has 50/50 paired candidate wins.

This closes the specific concern that exhaustive FP64 alone created H1's
advantage.  It does not yet establish superiority to a pedantic SGEMM/cuBLAS
exact filter/refine implementation, larger streamed joins, an external
multi-vector system, or a tree/indexed execution.

## Frozen screen results

The public screen denominator is outer operator wall time.  It includes the
full token-interval scan, all 65,536 object reductions, compaction, the
host-visible dynamic ambiguity count, direct-FP64 repair, final ID transfer,
and canonical host sort.  Each variant has 10 warmups and 50 retained
alternating baseline/candidate observations per cell.

| Dataset | Target/query | FP32 ambiguous objects | Baseline wall p10/median/p90 | Candidate wall p10/median/p90 | Median speedup | Wins |
|---|---:|---:|---:|---:|---:|---:|
| ESC-50/PANNs D2048 | 1 | 0 / 65,536 | 6.130/6.141/6.162 ms | 0.570/0.587/0.598 ms | 10.455x | 50/50 |
| ESC-50/PANNs D2048 | 8 | 0 / 65,536 | 6.102/6.109/6.141 ms | 0.935/0.940/0.948 ms | 6.499x | 50/50 |
| UCF101/R3D-18 D512 | 1 | 0 / 65,536 | 1.927/1.931/1.947 ms | 0.454/0.460/0.465 ms | 4.202x | 50/50 |
| UCF101/R3D-18 D512 | 8 | 2 / 65,536 | 1.988/1.994/2.014 ms | 0.717/0.723/0.730 ms | 2.758x | 50/50 |

The final video target-8 cell refines 44 valid token cells, or
`2.4763508493883413e-05` of the global padded token-cell domain.  The other
three FP32 baseline cells require no repair.  This means the measured baseline
is almost entirely an exhaustive certified FP32 scan rather than a slow FP64
fallback path.

CUDA-event timings are retained as secondary diagnostics only.  They are not
used to compute the table's speedups because the frozen denominator is outer
wall time.

## Exactness and safety

Across all four full cells:

- zero token-interval or object-interval containment violations;
- zero unsafe direct accepts or rejects;
- zero baseline/oracle, candidate/oracle, or candidate/baseline output
  mismatches;
- zero duplicate IDs, nonfinite bounds, or capacity overflows;
- invariant baseline, candidate, and oracle canonical-ID hashes.

The two bounded actual-data Compute Sanitizer runs cover the new baseline on
the fixed-cardinality D2048 and ragged D512 paths.  Both report
`ERROR SUMMARY: 0 errors`.  H1 candidate safety is an immutable dependency,
not silently reused as new H2A sanitizer evidence.

## Generated-code mechanism evidence

The runtime-generated direct-FP32 token kernels for both full workloads have
the same static resource profile: 40 registers, 1,024 bytes shared memory,
zero stack, and zero local bytes.  Each normalized instruction stream contains
167 `FADD`, 34 `FFMA`, 33 `FMUL`, and 160 `SHFL.BFLY` instructions, with zero
`IMMA`, `HMMA`, `MMA`, `QGMMA`, or `QMMA`.  This confirms that the measured
strong baseline is a direct FP32 reduction rather than an accidental Tensor
Core path.  Static code is mechanism evidence, not the timing denominator.

## Material caveats

1. The strong baseline is a custom direct-difference Triton implementation,
   not a runtime-resolved pedantic SGEMM/cuBLAS engine.  H2B must freeze its
   math mode, exact error certificate, selected kernel, and full binary/SASS
   provenance before comparison.
2. The 128x512 object matrices are resident and single-process.  Allocation,
   ingest, normalization, quantization, streaming, concurrency, and indexing
   are outside the denominator.
3. H2A is a 50-observation admission screen without fresh-process confidence
   intervals or a sustained workload gate.
4. Only two representations, 4--7 tokens/object, two output densities, one
   GPU model, and one compiler stack are covered.
5. The novelty search remains conditional.  H2A supports a mechanism result;
   it does not justify a broad "first exact multi-vector join" claim.

## H2B gate

Before tree integration or paper drafting:

1. implement and certify a pedantic SGEMM/cuBLAS FP32 filter/refine keeper,
   resolving the actual runtime kernel and math mode;
2. rerun exactness, generated-code, sanitizer, direction-balanced fresh-process
   timing, and sustained gates against the unchanged H1 candidate;
3. scale to larger disjoint object matrices while charging token-panel
   materialization/streaming and final IDs;
4. only after those gates pass, test an object-level scheduling/index layer
   that reduces token-panel work.

If the candidate loses to H2B's fastest certified exact keeper, close the upper
performance thesis instead of rescuing it with broader packaging.

## Evidence

- Correctness: `results/h2a_fp32_strong_correctness.json`, SHA-256
  `d773c3cf08578af1b4845d5dc3bdddced4138f167659e24c7bf25eb3a3b5c9be`.
- Memcheck summary: `results/h2a_fp32_strong_memcheck_summary.json`, SHA-256
  `d10f7dab881175cb742325104d1f93479a0c76aacf5f107a5768505b42e39da4`.
- Timing screen: `results/h2a_fp32_strong_timing_screen.json`, SHA-256
  `a97efa258375b3bc87949fc6a0efaf15fb863958dc8d099aa6f1344952cfdcb4`.
- Generated-code audit: `raw/h2a_generated_code_audit.txt`, SHA-256
  `c00576dc9c2ad47b51c3ad24517382a1509d706bcbc1f752aa8c3bc09e8cecfa`.
- GPU3 postflight: `raw/h2a_postflight_gpu3.txt`, SHA-256
  `3b7643ec3cc4895f46fe5f425af6874a3b004a0adab569c1f761cbc9691a06c1`.
- H2A runner/kernel SHA-256:
  `1013c1a783ffc3610d7e5af80176fa8468302030ba308a6e84e9ae3503301fd4` /
  `b65092657a54785386abfd56db83e32061016d01927d8fc4629342baed89d17b`.
- Target: `gpu-host-8`, physical GPU3, RTX PRO 6000 Blackwell Server Edition,
  compute capability 12.0.  GPU3 had no foreign compute process before or
  after the campaign.
