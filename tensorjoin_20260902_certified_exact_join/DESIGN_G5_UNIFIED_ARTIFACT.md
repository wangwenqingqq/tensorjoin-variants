# G5 Design: Unified Audited Dynamic Router on the Full Public Join

Date: 2026-09-03

## Decision target

G5 closes the paper's largest evidence gap: the generated-code-audited G4C
three-stage router and the G2B full public external comparison currently use
different TensorJoin implementations.  G5 composes the already accepted G4C
stage implementations into a scalable batched 60,000-vector runner without
changing their arithmetic mechanisms.

The G5 thesis impact is accepted only if the same TensorJoin source and the
same runtime-selected kernel specializations are used for full-output
correctness, generated-code audit, sanitizer/stress, and the public timing
campaign.

## Frozen system contract

| Field | Value |
|---|---|
| Host | `gpu-host-8`, live hostname `gpu-host-8` |
| Physical GPU | GPU2, UUID `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245` (R1 active contract) |
| Do not touch | GPU0, GPU1, GPU7, and every process not descended from the G5 runner |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120 |
| Driver/toolkit | driver 590.48.01; CUDA toolkit 13.1; PyTorch CUDA 13.0 |
| Python stack | Python 3.12.3, NumPy 2.3.1, PyTorch 2.11.0+cu130, Triton 3.6.0 |
| Dataset | CIFAR-10-GIST, 60,000 x 512 contiguous float32 |
| Input hashes | NPY `95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c`; raw `f7b51da49d6555fff8e094584b27c8c5c4efbfdcde1db2b7f86e58d9142af326` |
| Radius | epsilon 0.62890625; squared threshold 0.3955230712890625 |
| Exact output | 3,926,078 sorted directed `uint64` IDs, including 60,000 self pairs |
| Exact output hash | `13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495` |
| Public denominator | pageable host float32 source to sorted canonical host `uint64` IDs |
| Before timer | file I/O, process startup, CUDA context creation, Triton compilation |
| Inside timer | CPU quantization/analytic residuals, triangular schedule, allocations, H2D, all dynamic GPU stages and counter reads, D2H accepted IDs, symmetry expansion, host sort |
| After timer | hashes, exactness checks, JSON, optional pair-file serialization |

The frozen external keepers are the existing exact FP64 GDS-Join library
(`e0d31f51a8a2ed844b988b2c2b1c08ed9757a55dfceda338c787d999b34b54bb`)
and MiSTIC binary
(`20739abefe2fc21cfb93ec70b0c9cddbde20b76dc896243c72336981c402326a`).
Both are remeasured on physical GPU2; no G2B timing is reused.

The initial GPU1 preflight was invalidated before candidate startup when a
foreign 552 MiB process appeared.  The append-only R1 change and evidence are
recorded in `PROTOCOL_G5_P1_R1_GPU2.md`; no GPU1 measurement is part of G5.

## Candidate mechanism and immutable upstream sources

The candidate is one output-sensitive three-stage route:

1. `analytic_certificate_ragged_safe_i64`: per-vector INT8 quantization,
   INT32 Tensor-Core dot products, and analytic accept/reject intervals;
2. `certified_fp32_filter_i64`: FP32 squared-distance interval over actual
   stage-1 ambiguity;
3. `refine_ambiguous_fp64_i64`: exact FP64 refinement only for residual
   uncertainty.

Frozen upstream source hashes before G5 implementation:

| Source | SHA-256 |
|---|---|
| `src/g4b_r1_ragged_safe_kernel.py` | `439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6` |
| `src/run_g3b_r1_gpu_analytic_certificate.py` | `057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec` |
| `src/run_g3c_b_r1_gpu_cascade.py` | `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d` |
| `src/run_g2a_tensorjoin.py` | `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8` |
| `src/run_g4b_r1_public_opportunity.py` | `c1d539458233fb3268149e74847d8e6a5c8f4e3a3400954e7a5ee44a74f36234` |

G5 may add orchestration, batching, audit, and evidence code.  It may not edit
these five arithmetic/quantization sources during the frozen campaign.

## Dispatch and resource design card

- Full triangular tile schedule: 64 x 64 output tiles, K step 64, upper
  triangle including self.
- Batching: 4,096 tiles per batch; each stage buffer has 16,777,216 `int64`
  slots.  The final short batch retains the same compile-time capacity.
- Stage-1 CTA: four warps, three Triton stages.  Every CTA owns one independent
  tile and its INT32 accumulator; no inter-CTA barrier or shared ownership.
- Stage-2/3 work ownership: one Triton program per compacted pair.  Counter
  reads are the ready edges that determine the next launch extent.
- Ready graph: stage-1 writes accepted/ambiguous IDs -> host reads ambiguity
  count -> stage-2 consumes ambiguous IDs and writes residual IDs -> host reads
  residual count -> stage-3 consumes residual IDs -> host reads final count ->
  D2H accepted IDs.  A buffer is not overwritten before the corresponding
  count read and accepted-ID copy complete.
- Large live objects: full float32 vectors, INT8 codes and transpose, scales,
  reconstructed norms, residual bounds, triangular tile schedule, and three
  fixed-capacity ID buffers.  Per-batch counters die after the accepted-ID
  copy; accepted host chunks remain live until final concatenation and sort.
- Compile identity: `N_=60000`, `K=512`, `CAPACITY=16777216`, block sizes and
  warp/stage settings are identical in compatibility, safety/stress, and
  timing.  Safety may launch fewer selected tiles but must retain these exact
  specialization constants.

Expected useful-work delta versus fixed-FP64 external keepers: all upper pairs
receive INT8 matrix-unit work; only measured ambiguity receives scalar FP32;
only residual uncertainty receives FP64.  Added work is quantization,
certificate arithmetic, compaction atomics, counter synchronization, and
output reconstruction.

## Verification ladder and stop rules

1. **Compatibility:** one full candidate run must reproduce the exact count and
   hash with zero duplicates, invalid pairs, lower-triangle pairs, or overflow.
2. **Generated code:** the three selected cubins must be byte-stable across two
   fresh Triton caches.  Stage 1 must contain low-precision MMA, stage 2 must
   contain neither low-precision MMA nor FP64, and stage 3 must contain FP64;
   all must have zero stack/local and no LDL/STL.
3. **Safety/stress:** the same specializations must pass memcheck and a 1,000
   iteration two-buffer stress over selected full-N tiles that exercise all
   three precision stages, with an independent CPU FP64 oracle.
4. **Cheap timing kill:** two direction-balanced rounds over TensorJoin,
   MiSTIC, and GDS-Join.  Every exact method must reproduce the frozen output.
   G5 advances only if TensorJoin beats the faster exact keeper in both rounds
   by at least 1.50x and the paired geometric mean is at least 1.60x.
5. **Formal timing:** only after steps 1--4, run eight balanced three-method
   rounds.  Acceptance requires exact output in all admitted runs, at least
   seven of eight wins over both keepers, and a process-bootstrap 95% lower
   bound of at least 1.50x over the faster exact keeper.

Failure of exactness, runtime binary identity, safety, or the 1.50x cheap-kill
stops G5.  No SIFT/Fashion full-scale expansion is allowed before CIFAR passes.
