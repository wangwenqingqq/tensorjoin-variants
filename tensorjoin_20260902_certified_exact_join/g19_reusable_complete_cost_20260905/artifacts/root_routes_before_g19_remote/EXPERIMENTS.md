# Experiment Ledger

Append-only from the first measurement.

| Experiment | Hypothesis | Contract | Keeper/candidate | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---|---|---|---|
| `tensorjoin_20260903_multivector_certificate_h0` | Pairwise INT8 residual intervals remain selective after exact propagation through variable-cardinality symmetric Chamfer scoring on public-derived audio and video objects. | CPU-only; ESC-50/PANNs clips (5 tokens/object) and UCF101/R3D-18 groups (4--7 tokens/object); 128 disjoint query objects x 512 base objects; target 1/8 strict-mid-gap thresholds; direct-FP64-difference oracle. | Keeper: exhaustive FP64 token distances and exact object aggregation. Candidate: INT8 pair interval -> object interval -> competitive-cell FP64 refinement. | At most 5% ambiguous object pairs and at most 2.5% of all token cells refined in every cell. | Zero containment/decision/final mismatches; strict object disjointness; INT32 accumulator safe. | Pending; `PROTOCOL_H0_MULTIVECTOR_CERTIFICATE.md`. | `designed`; Gate 0 is conditional and GPU work is not admitted before this kill test passes. |
| `tensorjoin_20260902_int8_certificate_esc50_a0` | Per-vector INT8 intervals leave <=5% of pairs ambiguous on real audio windows | `PROTOCOL_A0.md` | FP64 all-pairs oracle / INT8 three-way certificate | INT8 evaluates every pair; FP64 refinement should evaluate <=5% | Exact output, fixed data/seed/features/shapes/radii | `raw/a0.log`, `results/a0_summary.json`, `results/a0_thresholds.csv` | Accepted for A0: exact output; 0.0765%, 0.3481%, 1.2486% ambiguity; median 0.3481% |
| `tensorjoin_20260902_pytorch_gpu_exact_join_b0` | INT8 dense scoring plus certified refinement beats exhaustive FP64 exact join end to end | `PROTOCOL_B0.md` | PyTorch FP64 exhaustive join / PyTorch INT8 certificate prototype | Same all-pairs scan; refine only A0-sized ambiguity set | Same source vectors and exact pair IDs; resident-input scope | `raw/b0_process_*.log`, `results/b0_process_*.json`, `results/b0_summary.json` | Accepted under B0: exact in 8/8; dense 2.138x; e2e 6.920x/5.065x/1.897x |
| `tensorjoin_20260902_fp32_strong_baseline_b0x` | The eager exact candidate also beats the strongest straightforward same-output FP32 exhaustive join | `PROTOCOL_B0X.md` | PyTorch FP32 exhaustive join / unchanged PyTorch candidate | Same as B0 | Both outputs exactly equal FP64 oracle | Planned: `raw/b0x_process_*.log`, `results/b0x_process_*.json`, `results/b0x_summary.json` | Designed; not measured |
| `tensorjoin_20260902_fp32_guarded_exact_b0y` | The eager INT8 candidate beats an FP32 scan with conservative FP64 boundary refinement | `PROTOCOL_B0Y.md` | Guarded FP32 exact-output join / unchanged INT8 candidate | Both scan all pairs; different ambiguity bands | Exact FP64-oracle pair IDs | `raw/b0y_process_*.log`, `results/b0y_process_*.json`, `results/b0y_summary.json` | Candidate rejected: exact, but 0.771x/0.559x/0.207x; 0/8 wins |
| `tensorjoin_20260902_triton_fused_status_c0a` | Fusing INT8 MMA and certificate classification removes enough intermediate traffic to leave an end-to-end win feasible | `PROTOCOL_C0A.md`, `DESIGN_C0A.md` | PyTorch materialized scan / Triton fused status scan | Same INT8 dot products and pair classification; candidate stores one byte/pair instead of INT32 plus FP64 intermediates | No false direct decisions; exact refined output | `raw/c0a_formal.log`, `results/c0a_formal.json`, `artifacts/c0a.sass`, `raw/c0a_memcheck.log` | Accepted: exact; IMMA; memcheck clean; 39.616 us median, 40.768 us p90 |
| `tensorjoin_20260902_triton_compact_refine_c0b` | In-kernel sparse compaction plus FP64 ambiguous refinement turns the fused scan into a >1.5x end-to-end win | `PROTOCOL_C0B.md`, `DESIGN_C0B.md` | Guarded FP32 exact-output B0Y keeper / Triton compact-and-refine candidate | Candidate stores only compact pair IDs and counters; all exact pairs emitted | Exact pair IDs; overflow-free; sanitizer; paired timing | `raw/c0b_*.log`, `results/c0b_*.json`, `artifacts/c0b.sass`, `receipts/c0b_*` | Accepted in frozen scope: exact and stable; 3.476x, 95% interval [3.402x, 3.542x], 8/8 wins |
| `tensorjoin_20260902_panns_certificate_d0` | A0's selective INT8 certificate survives a real 2048-D pretrained audio representation | `PROTOCOL_D0.md` | Direct-difference FP64 oracle / per-vector INT8 certificate | Refine only ambiguous learned-audio pairs | Clip-disjoint split; exact final output; fixed ambiguity gates | `raw/d0.log`, `results/d0_summary.json`, `results/d0_thresholds.csv`, `receipts/d0_panns_embedding.json` | Accepted for geometry with tie caveat: exact; 0.331%-0.535% ambiguity; nominal 1/8 radii yield 7.03/12.64 results/query because of exact embedding ties |
| `tensorjoin_20260902_panns_triton_d1` | The fused end-to-end win survives 2048-D learned audio, exact ties, and a larger output set | `PROTOCOL_D1.md`, `DESIGN_D1.md` | Guarded FP32 exact keeper / generalized Triton compact-and-refine candidate | Same exact pair IDs; candidate emits only compact IDs and refines ambiguity | Direct FP64 oracle; overflow-free; sanitizer; stress; paired timing at two radii | `raw/d1_*.log`, `results/d1_*.json`, `artifacts/d1_formal.sass`, `receipts/d1_*` | Accepted in frozen scope at both radii: exact and stable; 4.044x and 1.548x, lower bounds 4.035x and 1.536x, 8/8 wins each |
| `tensorjoin_20260903_ucf101_r3d18_embedding_d2` | A fixed open video corpus can supply a second learned-modality breadth workload | `PROTOCOL_D2.md` | Official torchvision R3D-18 Kinetics weights / deterministic UCF101 center clips | One 512-D normalized vector per video | Fixed archive/checkpoint/script hashes; all 13,320 files decode; finite nonzero output | `raw/d2_r3d18_extract.log`, `receipts/d2_r3d18_embedding.json`, `receipts/ucf101_*` | Accepted as data artifact: 13,320/13,320 decoded, no failures, 512-D finite normalized output |
| `tensorjoin_20260903_ucf101_r3d18_geometry_d2a` | The exact INT8 certificate remains selective on learned video embeddings | `PROTOCOL_D2.md` | Direct FP64 oracle / per-vector INT8 certificate | Candidate refines only ambiguous video pairs | Source-group-disjoint split; exact decisions; <=10% ambiguity | `raw/d2a_geometry.log`, `results/d2a_summary.json`, `results/d2a_thresholds.csv` | Accepted: exact; 0.0156% and 1.3010% ambiguity at exactly 1 and 64 results/query |
| `tensorjoin_20260903_ucf101_r3d18_triton_d2b` | The fused exact candidate keeps a >=1.5x advantage at both video radii | `PROTOCOL_D2.md` | Guarded FP32 exact keeper / D=512 fused compact-and-refine candidate | Same exact pair IDs | Cheap smoke before sanitizer/stress/formal campaign | `raw/d2b_smoke.log`, `results/d2b_t{1,64}_smoke.json` | Stopped at cheap performance kill: exact, but nominal-64 smoke is only 1.184x; no formal video performance claim |
| `tensorjoin_20260903_pivot_tree_ceiling_e0` | A fixed GTS-style pivot tree exposes enough rectangular pruning to rescue the high-output path | `GATE0_INDEX.md`, `PROTOCOL_E0.md` | Exhaustive scan / optimistic exact-FP64 pivot-tree ceiling | Tree prunes or bulk-accepts leaves before per-vector Tensor-Core panels | Exact output/bounds; padding charged; all four audio/video radius cells pass | `raw/e0_tree_ceiling.log`, `results/e0_summary.json` | Rejected: video prunes 0% at both radii; audio nominal-64 retains 97.22% of pairs; optimistic ceiling fails before CUDA |
| `tensorjoin_20260903_indian_pines_embedding_d3` | A fixed open scientific tensor corpus can supply a third representation workload | `PROTOCOL_D3.md` | Verified Indian Pines cube / deterministic standardized 3x3 spectral patches | One normalized 1,984-D vector per valid patch | Fixed source/script/cache hashes; finite nonzero output | `raw/d3_hsi_extract.log`, `receipts/d3_hsi_embedding.json` | Accepted as data artifact: 145x145x220 cube; 20,449 finite normalized 1,984-D patches |
| `tensorjoin_20260903_indian_pines_geometry_d3a` | The exact INT8 certificate remains selective on scientific tensor patches | `PROTOCOL_D3.md` | Direct FP64 oracle / per-vector INT8 certificate | Candidate refines only ambiguous HSI pairs | Source-pixel-disjoint split; exact decisions; <=10% ambiguity | `raw/d3a_geometry.log`, `results/d3a_summary.json`, `results/d3a_thresholds.csv` | Accepted: exact; 0.0222% and 0.5515% ambiguity at exactly 1 and 64 results/query |
| `tensorjoin_20260903_indian_pines_triton_d3b` | The fused exact candidate keeps a >=1.35x smoke advantage at both HSI radii | `PROTOCOL_D3.md` | Guarded FP32 exact keeper / D=1,984 fused compact-and-refine candidate | Same exact pair IDs | Cheap smoke before sanitizer/stress/formal campaign | `raw/d3b_smoke.log`, `results/d3b_t{1,64}_smoke.json` | Stopped at cheap performance kill: exact, but target-64 smoke is only 1.191x; no formal HSI performance claim |
| `tensorjoin_20260903_precision_cascade_f0` | A fixed INT8-to-FP32-to-FP64 cascade reduces FP64 refinement to <=25% of INT8-ambiguous pairs on all modality/radius cells | `GATE0_CASCADE.md`, `PROTOCOL_F0.md` | Existing two-stage certificate / simulated fixed three-precision cascade | FP32 handles most INT8 ambiguity; only its fixed guard band reaches FP64 | All six cells exact; universal <=25% FP64 fraction | `raw/f0_cascade.log`, `results/f0_summary.json` | Rejected as a universal mechanism: exact in all six cells, but tie-expanded low-radius audio sends 67.96% of ambiguity to FP64; five other cells pass |
| `tensorjoin_20260903_gpu_cascade_f1` | The F0 target-64 opportunity improves the current two-stage GPU path by >=1.15x and restores each modality's frozen overall speed gate | `DESIGN_F1.md`, `PROTOCOL_F1.md`, `RUN_CARD_F1.md`, `DECISION_F1.md` | Guarded FP32 / unchanged two-stage INT8+FP64 / INT8+FP32+FP64 cascade | FP32 stage replaces 91.98%-96.11% of target-64 FP64 distance work | Exact pair IDs and F0 counts; smoke, memcheck, stress, SASS, then eight-process timing | `raw/f1_*.log`, `results/f1_*.json`, `artifacts/f1_formal.sass`, `receipts/f1_*` | Accepted in frozen scope: exact and stable; cascade beats two-stage by 1.588x/1.690x/1.909x and guarded FP32 by 2.439x/1.959x/2.210x for audio/video/HSI; every lower confidence bound passes |
| `tensorjoin_20260903_external_exact_selfjoin_g2` | A scalable certified precision cascade can beat exact external GPU self-join systems while materializing the same complete Cifar60K radius-join output | `GATE0_G2.md`, `PROTOCOL_G2.md`, `RUN_CARD_G2.md`, `DESIGN_G2A.md`, `DECISION_G2A.md` | GDS-Join FP64 and MiSTIC FP64 independently / scalable TensorJoin cascade; FaSTED is lower-quality context only | Same canonical directed IDs including self; candidate avoids a full status matrix and fixed 65K output capacity | G2A 4,096-point canonical-ID admission before G2B full 60,000-point exact/safety/eight-round timing | `raw/g2a_*.log`, `results/g2a_*.json`, `receipts/g2a_*`; G2B planned | G2A accepted: all three methods reproduce the 262,144-ID oracle hash in 2/2 isolated runs; TensorJoin geometric capacity grows 32K->262K and is exact. G2B performance remains unmeasured |
| `tensorjoin_20260903_g3c_b_r1_gpu_cascade` | A GPU-computable outward FP32 interval can restore the exact precision router while sending <=0.1% of G3B-R1 ambiguity to FP64 and retaining selectivity under power-of-two scaling | `PROTOCOL_G3C_A.md`, `PROTOCOL_G3C_B.md`, `PROTOCOL_G3C_B_R1.md`, `DECISION_G3C.md` | G3B-R1 INT8+direct FP64 / unchanged G3B-R1 plus certified FP32 plus FP64 | FP32 should partition 102,079 first-stage ambiguous pairs and leave <=10,207 for FP64; R1 downscale case <=10,326 | Exact final IDs; zero unsafe decisions; stable runtime cubins; seven transformations; full memcheck; 1,000 complete three-stage iterations | `results/g3c_b_r1_final_summary.json`, `results/g3c_b_r1_*.json`, `raw/g3c_b_r1_*`, `artifacts/g3c_b_r1_*` | Accepted for correctness/generated-code/safety, not timing: G2A leaves 97/102,079 (0.0950%) for FP64; scale `2^-8` leaves 97 versus original G3C-B's 49,452; seven cases exact; memcheck clean; 1,000 complete cascades stable |

## Run history

- 2026-09-03 G5 P1 attempt 0: invalid before candidate startup. During the
  required GPU1 quiescence interval, a foreign process appeared with 552 MiB
  device memory. The guard launched no candidate and signalled no foreign
  process. Preflight/occupancy SHA-256 values are `c536ada5...a7c0` and
  `255b7f50...7970`. R1 moves the still-unmeasured campaign to identical
  physical GPU2; see `PROTOCOL_G5_P1_R1_GPU2.md`.

- 2026-09-02 A0 attempt 0: invalid before measurement. Feature construction
  encountered an all-zero centered audio window, so no split, threshold, or
  ambiguity metric was produced. Evidence:
  `raw/a0_invalid_zero_norm_20260902.log`. Repair: exclude exact zero-norm
  windows deterministically before the frozen seeded split and record them.
  Decision thresholds are unchanged.

- 2026-09-02 A0 attempt 1: accepted. From 18,000 possible windows, 1,356
  exact-zero centered windows were deterministically excluded, leaving 16,644.
  Query/base clips are disjoint. Ambiguity is 0.0765%, 0.3481%, and 1.2486%
  at 1, 8, and 64 results/query. All containment, false-accept, false-reject,
  and final-classification mismatch counts are zero. The hot representation is
  25.293% of FP32 bytes; retaining original vectors makes total storage
  125.293%. Evidence: `results/a0_summary.json`, SHA-256
  `9a62ca01c5b6afd7ef010d84c02dbab512ea57cb5564d9c134bdbc2f654178d1`.

- 2026-09-02 B0: accepted under its frozen FP64 comparator contract. All eight
  processes passed exact pair-ID equality. Dense INT8 versus non-TF32 FP32 is
  2.138x (95% bootstrap interval 2.133x-2.144x). End-to-end speedups versus
  exhaustive FP64 are 6.920x, 5.065x, and 1.897x at 1, 8, and 64
  results/query, with lower bounds 6.783x, 5.015x, and 1.875x. This evidence is
  insufficient for a system claim until B0X tests the same-output FP32 keeper.
  Evidence: `results/b0_summary.json`, SHA-256
  `d6836242e653c29806c4c5059cd8d649d790a6b4180dd5e204bcabd14fbd8ad6`.

- 2026-09-02 B0X smoke: rejected at correctness before formal sampling. Raw
  FP32 matched the FP64 oracle at 1 result/query but differed by one pair at
  both 8 and 64 results/query. The diagnostic two-sample latency is not a valid
  campaign result. Evidence: `raw/b0x_smoke.log`, `results/b0x_smoke.json`.

- 2026-09-02 B0Y: the guarded FP32 keeper and INT8 candidate both match the
  FP64 oracle in all eight processes. The eager candidate is nevertheless
  rejected: keeper/candidate ratios are 0.771x, 0.559x, and 0.207x, with 0/8
  wins at every radius. Candidate scan-plus-bounds is about 219 microseconds
  while raw INT8 GEMM is about 43.5 microseconds, admitting the one
  predeclared fused low-selectivity prototype. Evidence:
  `results/b0y_summary.json`, SHA-256
  `e7d9e0b759f211492ff346cc0a9a3709cc452cac23a55d1c33702d4b8518ef1c`.

- 2026-09-02 C0A memcheck attempt 0: instrumentation reported zero memory
  errors, but the target returned nonzero because the script incorrectly
  applied the normal latency gate to sanitizer-instrumented timing. This run is
  invalid as a sanitizer pass. Evidence:
  `raw/c0a_memcheck_instrumented_timing_invalid.log`. Repair: add a
  validation-only mode that retains correctness while omitting the timing gate;
  the kernel mechanism and C0A decision thresholds are unchanged.

- 2026-09-02 C0A: accepted. The fused status kernel has zero unsafe direct
  decisions, 1,608 ambiguous pairs (0.0767%), and exact refined output. SASS
  contains 16 `IMMA.16832.S8.S8.SAT` instructions; memcheck reports zero
  errors. Formal 20/200 latency is 39.616 microseconds median and 40.768
  microseconds p90, passing the 100/110 microsecond gate. Evidence:
  `results/c0a_formal.json`, SHA-256
  `24a1f539acf1cf8c47d7aace8eada475ae78eab2c1577fba4997e89af4320443`.

- 2026-09-02 C0B compile attempt 0: invalid before kernel launch. Triton
  rejected a scalar counter pointer combined with a block mask for atomic slot
  allocation. Repair: explicitly broadcast each counter address to a block
  pointer. No experiment metric was produced and the contract is unchanged.
  Evidence: `raw/c0b_compile_invalid_scalar_atomic.log`.

- 2026-09-02 C0B smoke/safety: accepted for formal measurement. The repaired
  fused scan produced 81 direct accepts and 1,608 ambiguous pairs, then emitted
  all 512 oracle pairs with zero direct false accepts, mismatches, duplicates,
  or overflows. `compute-sanitizer --tool memcheck` exited normally with
  `ERROR SUMMARY: 0 errors`. A 1,000-launch stress test preserved the same
  final-output SHA-256 on every launch. Extracted SASS contains 16 native
  `IMMA.16832.S8.S8.SAT` instructions in the scan and FP64 arithmetic in the
  refinement kernel. Evidence: `results/c0b_memcheck.json`,
  `raw/c0b_memcheck.log`, `results/c0b_stress1000.json`,
  `raw/c0b_stress1000.log`, `artifacts/c0b.sass`, and `receipts/c0b_*`.

- 2026-09-02 C0B formal: accepted under its frozen resident-input,
  one-result/query contract. All eight fresh processes exactly matched the
  FP64 oracle and the candidate won in 8/8. Guarded FP32 / fused candidate
  geometric-mean speedup is 3.476x with a deterministic 20,000-sample
  process-bootstrap 95% interval [3.402x, 3.542x]. Marginal medians are
  338.688 microseconds for guarded FP32 and 98.576 microseconds for the fused
  candidate; AB and BA split geomeans are 3.488x and 3.464x. This admits the
  broader-dataset/shape/external-baseline stage but is not a paper-level claim
  by itself. Evidence: `results/c0b_summary.json`, SHA-256
  `6b62c6885d13997a3594183832b31e4ed47f051b812024086afb0af982d2d267`.

- 2026-09-02 D0 environment attempt 0: invalid before model construction.
  Remote PyPI retrieval stalled while fetching `torchlibrosa`; it was
  interrupted without modifying the shared Isaac environment. Repair: fetch
  pinned Linux wheels locally and use an isolated project overlay.

- 2026-09-02 D0 smoke attempts 0-1: invalid before model forward. Attempt 0
  found an incompatibility between Numba 0.67.0 and the shared Coverage 7.4.4;
  project-local Coverage 7.10.7 repairs the missing `coverage.types.Tracer`
  interface. Attempt 1 then found TorchAudio 2.11 requires the absent
  `torchcodec` package for `torchaudio.load`. Repair: read the fixed WAV inputs
  with project-local SoundFile 0.14.0 and retain TorchAudio's frozen resampler.
  No embeddings or D0 certificate metrics were produced. Evidence:
  `raw/d0_panns_smoke.log` and `raw/d0_panns_smoke_attempt1.log`.

- 2026-09-02 D0 certificate attempt 0: invalid after measurement. The
  norm-expansion FP64 oracle reported 1,392 zero-distance cross pairs, while a
  direct-difference diagnostic found 3,597 truly identical pairs among 9,976
  near-zero candidates. Catastrophic cancellation therefore violated the
  intended exhaustive FP64 squared-distance oracle at the two smallest
  thresholds. Repair: compute `sum((q-x)^2)` directly in bounded query blocks;
  data, quantization, target ranks, and pass thresholds remain unchanged.
  Evidence: `raw/d0_nearzero_diagnostic.log`,
  `raw/d0_invalid_cancellation.log`, and
  `results/d0_summary_invalid_cancellation.json`.

- 2026-09-02 D0 repaired certificate: accepted for certificate geometry, with
  a material tie caveat. Blocked direct FP64 differences confirm exact output
  and zero containment/direct-decision errors at all radii. Ambiguity is
  0.535%, 0.535%, and 0.331%. PANNs emits bitwise-identical vectors for many
  silent/near-silent segments, so the nominal 1 and 8 order statistics produce
  7.025 and 12.639 exact results/query; the nominal 64 point produces 64.020.
  The low-rank points cannot be described as 1/8-result workloads. Evidence:
  `results/d0_summary.json`, SHA-256
  `a870e5c12b4cae2b3fdcc1d78dfe7e04c7eef406dbe14c222bc4902f85a42db4`.

- 2026-09-02 D1 compile attempt 0: invalid before refinement launch. Triton
  rejected a one-lane mask paired with an unbroadcast scalar result-counter
  pointer in the new blocked FP64 refinement kernel. Repair: explicitly
  broadcast counter addresses and stored pair IDs to one lane, matching the
  already validated C0B atomic pattern. No correctness or timing result was
  produced. Evidence: `raw/d1_t1_smoke_compile_invalid.log`.

- 2026-09-02 D1 order-statistic smoke: rejected for exact-output use at the
  nominal-64 radius. The nominal-1 run matched its 3,597-pair oracle and gave
  diagnostic latencies only. At nominal 64, both the candidate and guarded
  FP32 keeper missed the same 12 of 32,778 CPU-oracle pairs, localizing the
  failure to a threshold placed exactly on an observed FP64 distance rather
  than to the INT8 certificate. The adjacent-distance gaps are
  `2.116e-13` at nominal 1 and `1.901e-7` at nominal 64. Repair before formal
  measurement: use the midpoint between the ranked distance and the next
  strictly greater distance, preserving all ties and creating a cross-device
  decision margin. Evidence: `raw/d1_threshold_gap_diagnostic.log`,
  `results/d1_t1_smoke_orderstat.json`, and
  `results/d1_t64_smoke_orderstat.json`.

- 2026-09-02 D1 repaired smoke/safety: accepted for formal measurement. The
  tie-aware mid-gap radii yield 3,597 and 32,778 exact pairs (7.025 and 64.020
  results/query). Candidate and keeper both exactly match the direct-difference
  FP64 oracle, with zero unsafe direct accepts, duplicates, or overflows.
  Memcheck reports `ERROR SUMMARY: 0 errors` at both radii. Each 1,000-launch
  stress test retains one invariant output hash. The isolated formal-cache SASS
  contains 16 INT8 IMMA instructions plus FP64 refinement arithmetic. Evidence:
  `results/d1_t{1,64}_memcheck.json`, `results/d1_t{1,64}_stress1000.json`,
  `raw/d1_t{1,64}_memcheck.log`, `raw/d1_t{1,64}_stress1000.log`,
  `artifacts/d1_formal.sass`, and `receipts/d1_formal_*`.

- 2026-09-02 D1 formal: accepted under the frozen resident-input learned-audio
  contract at both radii. At nominal target 1 (actual 7.025 results/query), the
  guarded-FP32/candidate geometric-mean speedup is 4.044x with deterministic
  process-bootstrap 95% interval [4.035x, 4.051x]; marginal medians are
  1,430.416 and 353.824 microseconds. At nominal target 64 (actual 64.020), the
  speedup is 1.548x with interval [1.536x, 1.561x]; marginal medians are
  403.264 and 260.352 microseconds. Both radii pass correctness in 8/8 fresh
  processes and the candidate wins 8/8; AB/BA split geomeans are respectively
  4.049x/4.038x and 1.542x/1.553x. This passes the predeclared breadth gate but
  remains one audio model, dataset, shape, GPU, and exhaustive-scan contract.
  Evidence: `results/d1_summary.json`, SHA-256
  `a6f5e1c2976db510282abdbd591ba3adbd5fdc06c543036591ec6400cd7f6606`.

- 2026-09-02 D2 acquisition attempt 0: stopped as an operationally invalid
  transport path, before any feature measurement. The canonical UCF archive
  endpoint required a TLS-chain exception and sustained only about 70 KB/s
  from the experiment host, projecting roughly 27 hours. The owned download
  process was stopped without touching GPU work; its 9.5 MB partial archive is
  retained as `raw/UCF101.rar.partial_official_slow`. Before D2 measurement,
  the transport contract was changed to a pinned community ZIP whose stated
  role is a byte-preserving archive conversion. The new route exposes a linked
  SHA-256 and measured about 14.6 MB/s over a 64 MB range. Dataset identity is
  additionally gated by exact size, archive SHA-256, 13,320-file count, and a
  sorted-path manifest. Evidence: `raw/d2_ucf101_acquire_attempt0_slow.log` and
  `PROTOCOL_D2.md`.

- 2026-09-02 D2 acquisition/extraction: accepted. The pinned 6,957,373,664-byte
  ZIP matches linked SHA-256 `eb77e54d...fc4756e`, contains exactly 13,320 AVI
  files, and has sorted-path manifest SHA-256 `aa5734f8...e78d7`. The official
  torchvision R3D-18 checkpoint is 133,546,016 bytes with SHA-256
  `b3b3357e...f2e9d`. Full deterministic extraction decoded 13,320/13,320
  videos without failure in 161.8 seconds and produced a finite normalized
  `13320x512` cache, SHA-256 `132eb8b8...c80ae`. An earlier 32-video smoke
  attempt reached valid inference but failed only because the script applied
  the formal >=4,608-vector check to smoke output; the repaired smoke and full
  extraction are retained separately. Evidence: `raw/d2_r3d18_*.log`,
  `receipts/d2_r3d18_embedding.json`, and `receipts/ucf101_*`.

- 2026-09-02 D2A: accepted. On the source-group-disjoint 512x4096 video split,
  exact mid-gap radii yield exactly 1 and 64 results/query. The per-vector INT8
  certificate has zero containment violations, unsafe direct decisions, or
  final mismatches. Ambiguity is 328 pairs (0.0156%) and 27,283 pairs (1.3010%).
  Evidence: `results/d2a_summary.json`, script SHA-256
  `3aa92160...ce40da`, feature SHA-256 `132eb8b8...c80ae`.

- 2026-09-02 D2B smoke: stopped before safety/formal promotion by the frozen
  cheap performance kill. Both candidate and keeper exactly match the FP64
  oracle with zero unsafe decisions, duplicates, or overflows. At 1
  result/query, diagnostic medians are 303.392/87.536 microseconds (3.466x).
  At 64 results/query, they are 306.816/259.120 microseconds (1.184x), far
  below the 1.5x gate. No sanitizer, stress, or eight-process formal timing was
  run, and no video performance claim is allowed. Evidence:
  `results/d2b_t{1,64}_smoke.json`, `raw/d2b_smoke.log`.

- 2026-09-02 E0 optimistic tree ceiling: rejected. The fixed 8-way, 64-vector
  leaf pivot tree returned exact output with zero bound violations, but it did
  not create useful broad pruning. Video retained every pair at both radii and
  would execute 101.56% of full-scan padded cells after the pivot panel. Audio
  nominal 64 retained 2,038,784/2,097,152 pairs and charged 99.56% of a scan.
  Audio nominal 1 pruned strongly but leaf-panel utilization was only 56.74%,
  below 70%. Because E0 grants exact FP64 pivot distances and excludes packing
  and launch costs, this is an optimistic rejection; quantized GPU traversal
  cannot rescue the fixed design. Evidence: `results/e0_summary.json`, script
  SHA-256 `eb3fe854...a25151`.

- 2026-09-03 D3 extraction: accepted as a deterministic scientific-tensor
  artifact. The verified Indian Pines source has MD5
  `cea396f8a7bdf947f26a7a36c7b7c81a` and shape `145x145x220`. Global
  per-band standardization followed by valid `3x3x220` patch extraction,
  flattening, normalization, and four-coordinate zero padding produced 20,449
  finite `1,984`-D vectors in 2.08 seconds. Cache SHA-256 is
  `b95402ce...adb2`; extractor SHA-256 is `5ac045de...`. Evidence:
  `raw/d3_hsi_extract.log`, `receipts/d3_hsi_embedding.json`.

- 2026-09-03 D3A: accepted. The source-pixel-disjoint 512x4096 HSI split uses
  query top-left rows 0--47 and base rows 51--142. Exact midpoint radii yield
  exactly 1 and 64 results/query. The unchanged certificate has zero bound,
  direct-decision, or final-output errors and leaves 466 pairs (0.0222%) and
  11,565 pairs (0.5515%) ambiguous. Evidence: `results/d3a_summary.json`,
  SHA-256 `f1b67b02...a50ff`; script SHA-256 `eb942edc...27dccd94`.

- 2026-09-03 D3B smoke: stopped before safety/formal promotion by the frozen
  cheap performance rule. Both paths exactly match the FP64 oracle with zero
  unsafe decisions, duplicates, or overflows. At 1 result/query, diagnostic
  medians are 409.824/112.864 microseconds (3.631x). At 64 results/query,
  they are 466.992/392.192 microseconds (1.191x), below the required 1.35x.
  No sanitizer, stress, or eight-process formal timing was run, and no HSI
  performance claim is allowed. Evidence: `results/d3b_t{1,64}_smoke.json`,
  `raw/d3b_smoke.log`.

- 2026-09-03 F0 fixed precision cascade: rejected as a universal mechanism.
  All six audio/video/HSI cells preserve exact output with zero false FP32
  direct decisions, and five cells send only 2.13%--15.88% of INT8-ambiguous
  pairs to FP64. The tie-expanded low-radius audio cell, however, sends
  7,628/11,225 ambiguous pairs (67.96%) to FP64, failing the predeclared 25%
  all-cell gate. The `1e-3` guard is not retuned. This rejection does not erase
  the measured high-output opportunity: the three target-64 cells send 5.22%,
  3.89%, and 8.02% to FP64, but those are mechanism-opportunity measurements,
  not latency results. Evidence: `results/f0_summary.json`, SHA-256
  `b832f13a...54979`; script SHA-256 `6599d667...c9436`.

- 2026-09-03 F1 preflight: correctly aborted before launch. Physical GPU0 was
  occupied by user `xyh`, PID 2,225,280, using 17,132 MiB at 100% utilization
  for an unrelated video-generation job. The TensorJoin flock was available,
  demonstrating that the campaign lock cannot substitute for live process
  inspection. No F1 GPU command ran, no result file was created, and no other
  process was touched. Evidence: `raw/f1_preflight_blocked.log`, SHA-256
  `4601a9b7...5bc9d`. F1 source is additive and syntax-checked, SHA-256
  `058cc566...0b95a`; resume only after GPU0 is actually free.

- 2026-09-03 F1 smoke: accepted for safety and formal promotion on all three
  target-64 cells. The cascade exactly matches the direct-difference FP64
  oracle and F0 work counts (366/1,061/928 FP64 pairs for audio/video/HSI).
  Diagnostic `two_stage/cascade` medians are 1.559x, 1.674x, and 1.898x;
  `guarded_fp32/cascade` medians are 2.428x, 1.952x, and 2.184x. These exceed
  every frozen smoke threshold, but remain one-process admission measurements.
  Evidence: `results/f1_{audio,video,hsi}_smoke.json`, `raw/f1_smoke.log`;
  immutable smoke source SHA-256 `058cc566...0b95a`.

- 2026-09-03 F1 safety/SASS: accepted for all three modalities. Every
  validation-only run exactly matches the oracle, each memcheck reports
  `ERROR SUMMARY: 0 errors`, and every 1,000-launch stress run retains one
  output hash. The isolated formal-cache SASS contains 48 IMMA instructions
  and FP64 arithmetic (40 DADD, 4 DMUL, 4 DFMA). Evidence:
  `results/f1_{audio,video,hsi}_{memcheck,stress1000,sass_validation}.json`,
  `raw/f1_*_{memcheck,stress1000,sass_validation}.log`,
  `artifacts/f1_formal.sass` (SHA-256 `b95b2729...b8cfd`), and
  `receipts/f1_formal_*`. Promotion source SHA-256 is
  `19fd8341...32a8e6`.

- 2026-09-03 F1 audio formal: accepted for the completed modality, not as the
  global three-modality decision. All 8/8 fresh processes are exact and favor
  the cascade. `two_stage/cascade` geometric-mean speedup is 1.588x with 95%
  process-bootstrap interval [1.580x, 1.596x]; `guarded_fp32/cascade` is
  2.439x [2.433x, 2.445x]. Marginal medians are 259.200, 398.176, and
  163.296 microseconds for two-stage, guarded FP32, and cascade. Evidence:
  `results/f1_audio_process_*.json` and
  `results/f1_audio_formal_partial_summary.json` (SHA-256
  `789e6fcf...3028`). The partial summary explicitly sets the global gate to
  false while marking the covered audio gate true.

- 2026-09-03 F1 video process 3: excluded as contaminated. An unrelated user
  process began on physical GPU0 in the same second and overlapped the entire
  `GCT` sample. The campaign aborted after the post-process occupancy check;
  the original result is preserved but cannot enter an estimator. Video
  processes 0--2 remain clean. A later resume attempt aborted before any F1
  launch because another unrelated GPU0 process was already active. Evidence:
  `raw/f1_video_process_3_contamination.log`,
  `results/f1_video_process_3.json`, `raw/f1_formal_campaign.log`,
  `raw/f1_formal_campaign_resume1.log`, and
  `raw/f1_formal_resume1_blocked.log`. No foreign process was modified.

- 2026-09-03 F1 formal completion: accepted under the frozen target-64,
  resident-input contract. After a 30-second empty-GPU quiescence check, the
  one allowed clean video-process-3 replacement, video processes 4--7, and HSI
  processes 0--7 completed without foreign-process overlap. All 24 admitted
  process records are exact and both comparisons favor the cascade in 8/8
  processes per modality. Geometric-mean `two_stage/cascade` speedups are
  1.588x [1.580x, 1.596x] for audio, 1.690x [1.677x, 1.701x] for video, and
  1.909x [1.901x, 1.917x] for HSI. `guarded_fp32/cascade` speedups are 2.439x
  [2.433x, 2.445x], 1.959x [1.948x, 1.971x], and 2.210x [2.203x, 2.216x].
  Every predeclared lower-bound gate passes. The excluded contaminated video
  record remains excluded. Evidence: `results/f1_summary.json`, SHA-256
  `c3aa7f42...0507b`; `raw/f1_formal_campaign_resume2.log`, SHA-256
  `a685f6b8...3f7e`; source SHA-256 `19fd8341...32a8e6`.

- 2026-09-03 G2 source/build audit: GDS-Join and MiSTIC are MIT-licensed
  exact self-join sources; FaSTED is a lower-quality mixed-precision reference
  with no license file at the pinned ICPP revision. All three compile to SM120
  cubins after toolchain/configuration-only portability changes. Attempt 0 did
  not compile because `/usr/bin/time` is absent; attempt 1 used a broken nvcc
  symlink; attempt 2 established that CUDA 13.1 requires C++17 for GDS-Join and
  MiSTIC; attempt 3 built both with C++17. These attempts are retained and do
  not establish output correctness or speed. Evidence:
  `GATE0_G2.md`, `raw/g2_external_build_audit*.log`,
  `receipts/external_baseline_sources.json`,
  `receipts/g2_external_builds_sha256.txt`, and
  `receipts/g2_external_binary_metadata.txt`.

- 2026-09-03 G2A: accepted for G2B admission. The direct-difference FP64
  oracle contains 262,144 directed IDs including self at a strict mid-gap
  radius. GDS-Join, MiSTIC, and TensorJoin each reproduce the exact raw uint64
  hash in two isolated runs with zero missing, extra, duplicate, or invalid
  IDs. TensorJoin's forced capacity sequence is 32K, 65K, 131K, 262K; its
  work counts repeat exactly, with 204,555 INT8-ambiguous and 6,350 FP64-refined
  pairs. No G2A wall time is performance evidence. Evidence:
  `DECISION_G2A.md`, `results/g2a_summary.json`, SHA-256
  `92b7975a...25c76`, and `raw/g2a_*.log`.
- 2026-09-03 G2A2: accepted the proof-bounded upper-triangle production path
  for one full G2B smoke. Two isolated executions each evaluated 8,390,656
  unique upper-triangle elements, returned 133,120 unique upper pairs, and
  expanded to the exact 262,144 directed-ID oracle hash. All stage decisions,
  duplicate/lower-triangle checks, and overflow counters passed. No G2A2 wall
  time is performance evidence. Evidence: `DESIGN_G2B.md`, `DECISION_G2A2.md`,
  `results/g2a2_summary.json`, SHA-256 `f523909b...f1d3`, and
  `raw/g2a2_tensorjoin_*.log`.
- 2026-09-03 G2B full smoke: accepted exactness/resource admission, not
  performance. GDS FP64, MiSTIC FP64, and TensorJoin each materialized the same
  3,926,078 directed IDs with raw uint64 hash `13cae87e...63495`. TensorJoin
  used 108 proof-bounded upper-tile batches, refined 69,358 of 1,831,461
  ambiguous upper pairs in FP64, and had zero overflow. A CPU direct-float64
  audit found no false positive among all 1,993,039 accepted upper pairs. The
  pinned historical FP64-GDS count 3,926,074 is incompatible with this frozen
  source run and remains unresolved counterevidence, not an oracle. Evidence:
  `DECISION_G2B_SMOKE.md`, `results/g2b_smoke_summary.json`, SHA-256
  `b68f1d56...a101c`, and `raw/g2b_*`.

- 2026-09-03 G2B public-denominator screen: accepted only for allocating safety
  and formal work. Two direction-reversed rounds produced exact output for GDS,
  MiSTIC, and TensorJoin. TensorJoin/MiSTIC paired speedups were 4.994x and
  5.002x (4.998x geometric mean), above the cheap-screen gates. Attempt 0
  stopped before a benchmark because `/usr/bin/time` was absent; the optional
  dependency was removed without changing the denominator. No performance
  claim is taken from the two-round screen. Evidence: `DECISION_G2B_SCREEN.md`,
  `results/g2b_public_screen_summary.json`, SHA-256 `c55e9322...a101c`.

- 2026-09-03 G2B safety: accepted. Full memcheck reports `ERROR SUMMARY: 0
  errors`; two exact iterations exercise all three precision stages. The
  sustained run completed 1,000/1,000 full iterations and 3,000 core launches
  with invariant output/stage counts and zero overflow. Attempt 0's correct
  output but 4 MiB PyTorch caching-allocator teardown leak is retained; the
  repair changed cleanup only and did not weaken full leak checking. Evidence:
  `DECISION_G2B_SAFETY.md`, `results/g2b_tensorjoin_safety_manifest.json`.

- 2026-09-03 pinned FaSTED public validation: accepted as quality context only.
  The clean adapter run is structurally valid and emits 3,926,610 sorted unique
  IDs. Relative to the frozen exact output, it has 3,925,480 intersections, 598
  false negatives, and 1,130 false positives (precision 0.99971222, recall
  0.99984769, F1 0.99977995). Its first wrapper attempt was terminated by a
  monitor bug that reclassified one already-owned stale GPU PID as foreign; all
  failed artifacts are retained. Target-PID ownership was made monotonic while
  never-owned foreign PID detection remained active. Evidence:
  `results/g2b_public_validation_fasted_adaptercheck.json`, SHA-256
  `c27a9f0e...18b5a`, and `results/g2b_public_fasted_validation_attempt0_diagnosis.json`.

- 2026-09-03 G2B formal: accepted under the frozen full-output public
  denominator. All 32/32 four-method processes were clean and admitted; all 24
  exact-method outputs reproduce the 3,926,078-ID hash. TensorJoin's median is
  0.926153 s versus 4.732177 s for MiSTIC and 10.927797 s for GDS. It wins 8/8
  rounds against each; paired geometric-mean speedups are 5.083x over MiSTIC
  with 95% seeded round-bootstrap interval [5.030x, 5.133x], and 11.783x over
  GDS with interval [11.716x, 11.846x]. FaSTED's median is 0.229410 s but it is
  non-exact and excluded from acceptance. Evidence: `DECISION_G2B_FORMAL.md`,
  `results/g2b_public_formal_summary.json`, SHA-256 `893e4b63...b8c30`;
  manifest SHA-256 `d1d85dad...3a61`; 216-file evidence ledger SHA-256
  `cda1cfc5...34f`, locally reverified in full.

- 2026-09-03 G3A host analytic-certificate opportunity gate: accepted for a
  separate GPU proof candidate, not as GPU proof or timing evidence. The model
  audited all 8,390,656 G2A upper pairs with zero reconstructed-expression
  bound violations, interval-containment violations, or unsafe direct
  decisions. It leaves 101,920 pairs ambiguous, 0.9965x the frozen G2A2
  fixed-pad count and below the 127,846 gate. The largest observed expression
  error consumes 1.4025% of its pairwise bound. Evidence: `DECISION_G3A.md`,
  `results/g3a_analytic_certificate.json`, SHA-256 `4f7b4f9c...edf09e`;
  raw log SHA-256 `feea7b32...21756`.

- 2026-09-03 G3B/G3B-R1 GPU analytic-certificate admission: accepted for
  the frozen G2A plus seven-case adversarial proof contract, not performance.
  The parent Gate-0 audit found that the first G3B protocol had omitted its
  adversarial requirement. Five metamorphic cases passed before the all-zero
  case exposed an exact-zero-to-subnormal host residual-bound defect. Separate
  R1 preserves zero and rounds positive subnormal upper bounds to float32
  smallest normal. Seven adversarial/metamorphic cases then pass with zero
  unsafe decisions and exact final output, including all-zero FTZ and the
  `512*127^2 = 8,258,048` integer extreme. Two isolated G2A runs reproduce
  90,946 direct, 102,079 ambiguous, 133,120 final upper, and the exact 262,144
  directed-ID hash. R1 has the same normalized GPU instruction stream as G3B:
  16 INT8/INT32 IMMA, 64 approximate square roots, 140 registers/thread, and
  zero stack/local bytes. Full memcheck reports zero errors/leaks; 1,000/1,000
  two-buffer launches have invariant counts/hashes and zero allocator residue.
  The original domain failure and later wrapper-only cache-path error are
  retained. Evidence: `DECISION_G3B.md`, `results/g3b_r1_final_summary.json`,
  SHA-256 `19674be5...3e0f0d`; R1 cubin SHA-256 `dd32e979...114ac`; normalized
  instruction SHA-256 `8a59d826...e040`; 110-file evidence-ledger SHA-256
  `85446916...f6ca4`.

- 2026-09-03 G3C-A opportunity: accepted for a separate GPU FP32 candidate,
  not proof or timing. The live G3B-R1 kernel reproduces the frozen 102,079
  ambiguous-ID hash. A host `gamma_511` interval has zero containment or unsafe
  decision violations and leaves 40 pairs for FP64, versus 3,176 under the
  historical fixed `1e-3` guard. Evidence:
  `results/g3c_a_host_fp32_interval.json`, SHA-256
  `e4ec9ea0...a35f6b`; `PROTOCOL_G3C_A.md`.

- 2026-09-03 G3C-B initial GPU candidate: correct but superseded for scale
  selectivity before safety promotion. Two isolated G2A processes reproduce
  exact output and leave 97 pairs for FP64. A first generated-code audit records
  a false negative caused by comparing `cuobjdump` address targets with an old
  `nvdisasm` symbolic-label normalization; the full G3B cubin is byte-identical,
  and a same-normalizer re-audit passes. All seven adversarial outputs are exact,
  but `scale_2^-8` leaves 49,452/103,266 pairs for FP64 because
  `max(magnitude,1)` is not scale-equivariant. Evidence:
  `results/g3c_b_gpu_cascade_process_{0,1}.json`,
  `results/g3c_b_generated_code_audit{,_r1}.json`, and
  `results/g3c_b_adversarial.json`. No G3C-B safety or timing claim is allowed.

- 2026-09-03 G3C-B-R1 final correctness/safety admission: accepted, not timed.
  The separate R1 source removes only the unit floor from the final relative
  margin while retaining two `4096*tiny` absolute FTZ floors. Two isolated G2A
  processes return the exact 262,144-ID hash and route 97/102,079 (0.0950%)
  first-stage ambiguous pairs to FP64. The downscale case also leaves 97, below
  its frozen 10,326 gate, and all seven cases have zero unsafe decisions and
  exact output. The G3B cubin remains `dd32e979...114ac`; the G3C cubin is
  `db750295...20670`, uses 32 registers/thread, and has zero stack/local bytes.
  Full three-stage memcheck reports zero errors/leaks. A sustained test completes
  1,000 cascades/3,000 core launches with invariant six-stage hashes, zero
  mismatches/overflow, and zero allocator residue. Evidence: `DECISION_G3C.md`,
  `results/g3c_b_r1_final_summary.json`, SHA-256
  `82aa22b3...47f8d`; source SHA-256 `637e9455...f706d`.

- 2026-09-03 G3D complete resident-GPU pipeline attribution: accepted. The
  unchanged G3B-R1 two-stage keeper refines all 102,079 ambiguity pairs in
  FP64; the exact G3C-B-R1 candidate inserts the certified FP32 stage and
  refines 97. Eight direction-balanced fresh processes, each with 100 retained
  complete-pipeline CUDA-event observations, give a 2.669585x paired geometric
  mean speedup with seeded 95% process-bootstrap interval
  [2.658105x, 2.677861x] and 8/8 wins. Separate 1,000-invocation continuous
  batches give 2.867997x and 8/8 wins. All pre/post outputs reproduce the
  133,120-ID upper hash, and every fresh cache contains exactly the accepted
  G3B, FP32-filter, and FP64-refinement cubins. The original screen driver
  cleanly completed `KC`, then stopped before launching `CK` when an unrelated
  `sglang::scheduler` appeared during quiescence; the blockage is retained and
  Resume1 changes outer scheduling only. G3D excludes ingest, dynamic count
  discovery, D2H output, and host sort, so it is mechanism attribution rather
  than public end-to-end evidence. Evidence: `DECISION_G3D.md`,
  `results/g3d_formal_summary.json`, SHA-256 `2778c76e...f6de`; formal manifest
  `adfa81b3...2c1f`; runtime audit `84f6d9aa...c147`.

- 2026-09-03 G4A/G4A-R1 dynamic-count router attribution: accepted on the
  revised same-model GPU1 contract.  The original GPU0 screen passed, but three
  formal attempts were excluded when unrelated dense/sparse benchmarks and
  later an SGLang server acquired GPU0.  The isolation path either blocked
  before launch or terminated only the TensorJoin process group; all partial
  campaigns and clean-but-unused slots are retained under `rejected/` and no
  foreign process was modified.  `PROTOCOL_G4A_R1.md` changes only the physical
  device UUID, lock, and evidence prefix, and requires a fresh screen plus
  formal campaign.  The GPU1 screen passed in 2/2 processes.  The formal eight
  direction-balanced processes all reproduced the exact 133,120 upper-pair
  output and the dynamic count tuples `102079 -> 102079 -> 133120` for the
  keeper and `102079 -> 97 -> 133120` for the candidate.  After charging actual
  scalar D2H count reads, their synchronizations, Python/Triton launches, and
  dynamic stage extents, the paired process-median geometric-mean speedup is
  2.194639x with seeded 95% process-bootstrap interval [2.159275x, 2.213376x]
  and 8/8 wins.  Separate 1,000-invocation dynamic sequences give 2.226875x and
  8/8 wins.  Every fresh runtime cache contains exactly the accepted G3B/G3C/
  FP64 cubins.  This closes G3D's fixed-count shortcut but still excludes input
  ingest and final pair-ID D2H/sort.  Evidence: `DECISION_G4A_R1.md`,
  `results/g4a_r1_formal_summary.json`, SHA-256 `1a18a5ed...ffd209`; formal
  runtime audit `437a87c3...9799`; and 708-file ledger
  `receipts/g4a_r1_final_evidence_sha256.txt`, SHA-256
  `b42c41a2...3f088`.

- 2026-09-03 G4B/G4B-R1 public breadth opportunity: the first implementation
  is rejected and the separate ragged-safe revision is accepted for timing.
  G4B completed all SIFT-128 and CIFAR-GIST-512 cells, then stopped on the first
  Fashion-MNIST-784 cell with 8 unsafe G3B accepts, 4 unsafe G3B rejects, 8
  final extras, and 4 misses. An isolated in-process same-input A/B localized
  the first failure to the G3B final K tile: the D=784 tail has 16 valid lanes,
  but the inherited static mask enabled 64. Changing only the address/mask to
  `block_start+offsets_k < K` eliminates every unsafe decision. A minimal
  Compute Sanitizer probe reported zero allocation-level errors and is retained
  as inconclusive because the caching allocator can keep logical cross-tensor
  accesses inside a larger valid allocation. G4B-R1 uses a separate source and
  reruns all 27 cells. Every SIFT-128/CIFAR-GIST-512/Fashion-784 x
  N=1024/2048/4096 x k=1/16/64 cell exactly matches its direct-FP64 oracle with
  zero unsafe decision, duplicate, invalid ID, or overflow. Both non-CIFAR
  N=4096 sources pass all three residual-FP64 thresholds and both low/medium
  G3B selectivity cells. Evidence: `DECISION_G4B.md`, `DECISION_G4B_R1.md`,
  `results/g4b_r1_opportunity_summary.json`, SHA-256
  `4cc069e2...60371`; rejected result
  `results/g4b_opportunity_fashion784.json`; diagnostic SHA-256
  `dce93811...c5a8`. This is breadth/correctness, not timing.

- 2026-09-03 G4C public-anchor dynamic timing: accepted. The deterministic
  compact subset takes the maximum frozen scale and density (`N=4096,k=64`)
  on all three public sources. A six-slot screen passed on all datasets. The
  formal campaign then admitted 24/24 isolated fresh processes, each with 200
  retained calls and a separate 1,000-call sustained loop per variant. The
  exact three-stage router reduces measured FP64 work from 39,810 to 83 pairs
  on SIFT, 109,499 to 117 on CIFAR, and 27,349 to 63 on Fashion. Paired
  process-median geometric-mean speedups over the identical G3B plus all-FP64
  keeper are 1.270875x [1.267100x,1.274316x], 2.328925x
  [2.299469x,2.358065x], and 1.709810x [1.687169x,1.735672x], respectively;
  all three have 8/8 median and 8/8 sustained wins. Every dynamic count and
  exact pre/post output is invariant. The formal fresh caches reproduce the
  per-dimension screen-audited `sm_120a` G3B/G3C/FP64 cubins with zero
  stack/local bytes or static LDL/STL. Evidence: `DECISION_G4C.md`,
  `results/g4c_formal_summary.json`, SHA-256 `43ed0a57...9cf3`; runtime audit
  `b1714742...a106`. The timing scope includes actual D2H count reads,
  synchronizations, and host launches but excludes ingest, resident setup, and
  final output transfer/sort.

- 2026-09-03 H0 exact multi-vector certificate opportunity: accepted as a
  CPU-only structural gate. On 128 disjoint query objects by 512 base objects,
  the symmetric-Chamfer certificate has zero token/object containment
  violations, unsafe decisions, or final mismatches. ESC-50/PANNs object
  ambiguity is 0.1053%/0.5432% and global FP64 competitive-cell refinement is
  0.0712%/0.3501% at targets 1/8. UCF101/R3D-18 is
  0.1434%/1.2527% ambiguous and 0.1132%/1.0027% refined. All four frozen gates
  pass. Evidence: `DECISION_H0_MULTIVECTOR_CERTIFICATE.md`,
  `results/h0_multivector_certificate.json`, SHA-256
  `7d8b8e68...a68af31`; runner `eaca708d...830191`. This admits H1 GPU design;
  it is not performance, generated-code, or definitive novelty evidence.

## H1 pre-run: exact GPU multi-vector operator screen (2026-09-03)

- **State:** frozen before implementation or GPU execution.
- **Question:** can H0's exact object-level certificate algebra become a
  resident-GPU operator that beats exhaustive direct-FP64 evaluation under the
  identical variable-cardinality symmetric-Chamfer threshold-join contract?
- **Keeper:** exhaustive direct-FP64 token distances plus exact object
  aggregation and canonical final IDs.
- **Candidate:** full INT8 Tensor Core token intervals, object-first compaction,
  then padded 8x8 direct-FP64 refinement only for ambiguous object panels.
- **Workloads:** frozen H0 ESC-50/PANNs and UCF101/R3D-18 splits, thresholds for
  target results/query 1 and 8, physical GPU1 on `gpu-host-8`.
- **Correctness/safety gate:** zero oracle/output/containment/direct-decision
  errors and overflows; keeper and candidate Compute Sanitizer memcheck pass.
- **Performance screen gate:** 10 warmups plus 50 alternating observations;
  candidate median speedup at least 1.25x with no losing cell.  Formal
  confidence and sustained evidence are explicitly out of scope.
- **Design/protocol:** `DESIGN_H1_MULTIVECTOR_GPU_SCREEN.md` and
  `PROTOCOL_H1_MULTIVECTOR_GPU_SCREEN.md`.
- **Rejected launch/compile smoke:** a bounded candidate invocation compiled
  and returned IDs, but its immediate preflight showed 52.7 GiB already used on
  physical GPU1 by another user's eight-GPU process.  It is excluded from all
  correctness, safety, and performance evidence.  No foreign process was
  stopped.  The runner was then strengthened to resolve GPU1's UUID and refuse
  every launch while any foreign compute PID is present; execution remains
  pending an idle card.
- **Guard verification:** the revised runner was invoked under the GPU1 lock
  while the external process remained resident and refused before experiment
  launch, naming physical GPU1's UUID/PID.  Raw log
  `raw/h1_foreign_process_refusal.log`, SHA-256
  `bdb0839eb516a8820418c8acf1d32332c64b2501dc60f28932a085ff07b96245`.
  This is safety-control evidence only; H1 remains undecided.

## H1-R1 pre-run device revision (2026-09-03)

- At 09:08:46 GMT, physical GPU3 and GPU4 on `gpu-host-8` were empty while
  GPU0/1/2/5/6/7 had live processes.  H1-R1 selects GPU3 and creates a separate
  GPU3 advisory lock.
- Only the physical device, lock, and artifact names change.  The same-model
  hardware, semantic contract, kernel sources, correctness/safety ladder, and
  timing gate are unchanged.  See `PROTOCOL_H1_R1_GPU3.md`.
- **Attempt 0 rejected before measurement:** the full audio correctness run
  reached candidate exact-panel compilation after the independent CPU oracle,
  then Triton rejected a one-element block value stored through a scalar
  pointer.  No correctness result was written and no timing was attempted.
  The repair changes only that output pointer to an explicit one-lane block;
  the failed raw log is retained as
  `raw/h1_r1_correctness_attempt0_compile_fail.log`.

## H1-R1 result: exact GPU multi-vector screen (2026-09-03)

- **Decision:** pass all frozen correctness, safety, and performance screen
  gates; admit H2.  See `DECISION_H1_R1.md`.
- **Exactness:** four/four cells have zero token/object containment violation,
  unsafe direct decision, duplicate, overflow, keeper mismatch, or candidate
  mismatch.  Candidate, GPU keeper, and CPU oracle canonical hashes agree.
- **Safety:** keeper/candidate on D2048 fixed-cardinality and D512 ragged actual
  subsets each report Compute Sanitizer `ERROR SUMMARY: 0 errors`.
- **Outer-wall medians:** audio target 1: 35.693 ms -> 0.549 ms (65.063x);
  audio target 8: 35.703 ms -> 0.953 ms (37.477x); video target 1:
  9.860 ms -> 0.457 ms (21.570x); video target 8: 9.864 ms -> 0.720 ms
  (13.707x).  Every cell has 50/50 candidate wins and invariant output hashes.
- **Mechanism:** captured token-certificate cubins contain 16 static signed
  INT8 `IMMA.16832.S8.S8.SAT` instructions; audited instances use 95--96
  registers with zero stack/local bytes.
- **Scope limit:** this compares against exhaustive direct FP64 on a small
  resident 128x512 object matrix.  It is not evidence against a certified
  FP32/TF32 exact baseline and is not yet a system or novelty claim.
- **Evidence hashes:** correctness `06461064...8dca4`; memcheck summary
  `f022cc90...5d09`; timing `c98d0e33...1f3173`; generated-code audit
  `da4c39f5...cea1b`.

## H2A pre-run: certified FP32 strong-baseline kill test (2026-09-03)

- **State:** frozen before new-kernel execution.
- **Question:** does the accepted H1 INT8 candidate remain at least 1.25x
  faster than a same-contract direct-FP32 interval scan with identical exact
  FP64 object-panel repair?
- **Frozen scope:** same four H1 cells and outer-wall denominator on physical
  GPU3; 10 warmups and 50 alternating observations per variant/cell.
- **Stop:** any containment/output/safety failure, any losing cell, speedup
  below 1.25x, or fewer than 50/50 wins rejects H2A and blocks tree work.
- **Design/protocol:** `DESIGN_H2A_FP32_STRONG_BASELINE.md`,
  `PROTOCOL_H2A_FP32_STRONG_BASELINE.md`.

## H2A result: certified FP32 strong-baseline screen (2026-09-03)

- **Decision:** pass all frozen exactness, safety, and performance gates; admit
  a separate H2B pedantic SGEMM/cuBLAS plus larger-streaming gate.  See
  `DECISION_H2A.md`.
- **Exactness:** four/four cells have zero token/object containment violation,
  unsafe direct decision, baseline/candidate/oracle mismatch, duplicate,
  nonfinite bound, or overflow.  Three certified-FP32 cells require no FP64
  repair; video target 8 repairs only two objects and 44 valid token cells.
- **Safety:** both bounded actual-data baseline memchecks report
  `ERROR SUMMARY: 0 errors`; accepted H1 candidate safety remains an immutable
  dependency.
- **Outer-wall medians:** audio target 1: 6.141 ms -> 0.587 ms (10.455x);
  audio target 8: 6.109 ms -> 0.940 ms (6.499x); video target 1:
  1.931 ms -> 0.460 ms (4.202x); video target 8: 1.994 ms -> 0.723 ms
  (2.758x).  Every cell has 50/50 paired candidate wins.
- **Mechanism:** both full-workload direct-FP32 baseline cubins use 40
  registers, 1,024 bytes shared memory, and zero stack/local bytes.  Their
  normalized streams each contain 167 `FADD`, 34 `FFMA`, 33 `FMUL`, and zero
  MMA-family instructions.
- **Scope limit:** custom direct-FP32 Triton keeper, small resident object
  matrix, one process, no formal CI/sustained/streaming/index/external-system
  evidence.  Passing H2A does not waive H2B.
- **Evidence hashes:** correctness `d773c3cf...c9be`; memcheck summary
  `d10f7dab...da4`; timing `a97efa25...cb4`; generated-code audit
  `c00576dc...ecfa`.

## H2B-P0 pre-run: pedantic-SGEMM certificate opportunity (2026-09-03)

- **State:** numerical opportunity contract frozen before execution; no H2B
  result exists yet.
- **Question:** can a dimension-aware worst-case FP32 dot-product error envelope
  remain selective enough to justify a direct pedantic-cuBLAS GPU keeper?
- **Scope:** the same four H2A cells; host direct-FP64 geometry only.  P0 uses a
  pessimistic envelope that charges both allowed midpoint displacement and its
  enclosing interval radius.
- **Stop:** any soundness failure, more than 5% ambiguous objects in any cell,
  or more than 2% in either target-8 cell blocks the GPU implementation.
- **Design/protocol:** `DESIGN_H2B_PEDANTIC_SGEMM.md`,
  `PROTOCOL_H2B_P0_PEDANTIC_OPPORTUNITY.md`.

## H2B-P0 result: pedantic-SGEMM certificate opportunity (2026-09-03)

- **Decision:** pass the CPU opportunity gate and admit H2B-P1 GPU correctness;
  see `DECISION_H2B_P0.md` and `PROTOCOL_H2B_P1_PEDANTIC_CORRECTNESS.md`.
- **Pessimistic ambiguity:** audio target 1/8 leaves 6/19 objects
  (0.00916%/0.0290%); video target 1/8 leaves 0/8
  (0%/0.0122%).  Every cell is far below its frozen 5%/2% gate.
- **Envelope:** maximum two-radius half width is 0.0009773 at D2048 and
  0.0002446 at D512; all explicit token/object containment and direct-decision
  checks are zero.
- **Boundary:** CPU direct-FP64 ideal-center geometry only.  No cuBLAS kernel,
  target-GPU containment, generated-code identity, sanitizer, stress, or timing
  claim is admitted.
- **Evidence:** result/raw SHA-256 `7b514ab...21e4`; runner SHA-256
  `57c7e9c7...3676`.

## H2B-P1 pre-run: direct-cuBLAS correctness gate (2026-09-03)

- **State:** frozen before implementation or GPU execution.
- **Keeper:** direct `cublasGemmEx` FP32 token dots with queried pedantic math
  mode and explicit pedantic compute type, frozen theoretical error intervals,
  object-first decisions, and unchanged exact-FP64 repair.
- **Gate:** full actual-data and adversarial dot/token/object containment,
  direct-decision, output-ID, capacity, and repeatability checks must all pass.
- **Boundary:** P1 correctness cannot be promoted as generated-code, safety, or
  performance evidence.
- **Protocol:** `PROTOCOL_H2B_P1_PEDANTIC_CORRECTNESS.md`.

## H2B-P1-R1 pre-run device revision (2026-09-03)

- The GPU3 preflight found external PID 4169406 using about 28 GiB at
  92--96% utilization.  No H2B kernel was launched and no external process was
  modified.
- Physical GPU1 was idle, is the identical RTX PRO 6000 Blackwell Server
  Edition/SM120 model, and is selected under the existing GPU1 campaign lock.
- Only device/UUID/lock and resulting P1 source/evidence hashes change.  All
  numerical and validation contracts remain frozen.  See
  `PROTOCOL_H2B_P1_R1_GPU1.md`.
- **Attempt 0 rejected before GEMM:** the typed FFI bound the unsuffixed
  `cublasSetPointerMode` symbol, while cuBLAS 13 exports
  `cublasSetPointerMode_v2`.  Initialization stopped before any experimental
  GEMM/kernel launch and no result file was written.  The repair changes only
  the two pointer-mode symbol names; the failed raw log is retained as
  `raw/h2b_p1_correctness_attempt0.log`.

## H2B-P1 result: direct-cuBLAS correctness gate (2026-09-03)

- **Decision:** pass full actual-data, adversarial, and two-process
  repeatability gates; admit P2 runtime-kernel/SASS resolution.  See
  `DECISION_H2B_P1.md`.
- **API contract:** cuBLAS 13.1.0 reports pedantic math mode 2; each GEMM uses
  FP32 A/B/C, compute type 69 (`CUBLAS_COMPUTE_32F_PEDANTIC`), default
  algorithm -1, and `NVIDIA_TF32_OVERRIDE=0`.
- **Actual data:** audio target 1/8 leaves 2/9 ambiguous objects; video target
  1/8 leaves 0/4.  All four final outputs equal both the independent FP64
  oracle and accepted H1 candidate, with zero bound/decision/capacity issue.
- **Adversarial:** seven cases at each of D512/D2048 pass, including zero,
  identity, sign, cancellation, one-hot, small-scale, and ragged objects.
- **Repeatability:** two fresh processes reproduce all four actual-data output
  hashes and ambiguity counts.
- **Boundary:** API configuration is not generated-code proof.  No P2 SASS,
  P3 sanitizer/stress, P4 timing, or streaming evidence is yet admitted.
- **Evidence hashes:** correctness `68f7e587...7e99`; repeats
  `6970b41c...fc27` / `75d22742...da30`; kernel `33103f09...bf43`; runner
  `d747b7c2...eef1`.

## H2B-P2 pre-run: runtime cuBLAS kernel/SASS audit (2026-09-03)

- **State:** frozen after P1 and before profiler execution.
- **Shapes:** full token dot matrices 640x2560x2048 and 662x2684x512 under
  the exact P1 API/math/compute contract on physical GPU1.
- **Gate:** bind each API call to its actual runtime kernel and selected-function
  SASS/resources; require no TF32 or lower-precision MMA evidence.  Unresolved
  binary acquisition is not a pass.
- **Boundary:** all profiler durations are diagnostic and excluded from P4
  outer-wall performance.
- **Protocol/runner:** `PROTOCOL_H2B_P2_CUBLAS_SASS.md`,
  `src/run_h2b_p2_trace.py`.

## H2B-P2 result: runtime cuBLAS kernel/SASS audit (2026-09-03)

- **Decision:** pass the actual selected-function precision gate and admit P3
  sanitizer/stress; see `DECISION_H2B_P2.md`.
- **Runtime binding:** NSYS binds the audio `(640,2560,2048)` call to
  `cutlass_80_simt_sgemm_256x128_8x4_tn_align1` and the video
  `(662,2684,512)` call to `cutlass_80_simt_sgemm_128x64_8x5_tn_align1`.
- **Selected SASS:** NCU exports 4,288/2,496 static instructions with
  1,152/576 FP32 `FFMA` and zero `F2FP.TF32`, `HMMA`, `IMMA`, `MMA`,
  `QMMA`, `QGMMA`, or `WGMMA` in both exact functions.
- **Resources:** audio/video use 212/130 registers per thread and
  49.15/30.72 KiB dynamic shared memory per block; NCU reports zero local
  spilling requests.
- **Attempt 0 retained:** the first NSYS invocation used a relative `env`
  executable and was rejected before CUDA launch; it contributes no result.
- **Boundary:** profiler evidence proves the two selected-function instruction
  paths, not safety or latency.  Parent static/JIT image linkage was not
  exposed by the trace and remains unresolved.
- **Evidence:** audit SHA-256 `c4145a0e...94d77`; normalized selected SASS
  `4e7a7c2e...0753` / `fa5dd1a9...6623`; NCU reports
  `de5a2e31...98e` / `5c7dc3e5...48fc`.

## H2B-P3 pre-run: full-operator safety and stability (2026-09-03)

- **State:** frozen before sanitizer or stress execution.
- **Safety:** full actual-data target-8 operator under Compute Sanitizer
  `memcheck` and `synccheck` for both audio and video, with tool error exit
  codes and exact P1 output/count/ambiguity checks.
- **Stress:** two independent device buffers, 1,000 complete invocations per
  dataset/target cell, 4,000 retained iteration records total.
- **Stop:** any sanitizer error, crash, P1 count/hash/ambiguity drift,
  duplicate, overflow, or multi-signature cell rejects P3 and blocks timing.
- **Protocol/runner:** `PROTOCOL_H2B_P3_SAFETY_STRESS.md`,
  `src/run_h2b_p3_safety_stress.py` (SHA-256 `41422276...9f86`).

## H2B-P3 attempt 0 rejected; P3-R1 pre-run (2026-09-03)

- **Rejected attempt:** the full audio target-8 operator returned the exact P1
  1,024-ID hash and ambiguity count 9, but full leak checking reported 12
  outstanding PyTorch allocator/cuBLAS-workspace allocations totaling
  153,224,192 bytes.  The original any-error rule rejects the attempt; no
  later original subgate ran.
- **R1-only change:** disable PyTorch CUDA allocation caching for instrumented
  processes and explicitly release state/cache/cuBLAS workspaces before exit;
  keep full leak checking and all operator/data/output gates unchanged.
- **R1 runner:** SHA-256 `5624418b...a441`; rejected log SHA-256
  `60eb6d14...a8610`.
- **Protocol:** `PROTOCOL_H2B_P3_R1_SAFETY_STRESS.md`.

## H2B-P3-R2 pre-run device revision (2026-09-03)

- P3-R1 stopped before GPU execution because external PID 4186371 occupied
  physical GPU1 with approximately 45 GiB at 99% utilization.  It was not
  modified.
- R2 changes only the device/UUID/lock to idle physical GPU4, the identical
  RTX PRO 6000 Blackwell Server Edition / SM120 model.  All R1 safety/stress
  contracts remain frozen.
- **Runner/protocol:** SHA-256 `63b19aab...45ba`;
  `PROTOCOL_H2B_P3_R2_GPU4.md`.

## H2B-P3-R2 rejected; P3-R3 access-safety pre-run (2026-09-03)

- **R2 rejection:** explicit cleanup reduced outstanding allocations to three
  / 8,520,704 bytes, all rooted at PyTorch's process-owned
  `cublasCreate_v2` handle.  Exact output passed and no access error was
  reported, but R2's leak-free gate fails.  No later R2 subgate ran.
- **R3 scope revision:** memcheck now gates device-memory access errors with
  leak reporting disabled; the rejected full-leak log remains the explicit
  resource-lifetime caveat.  Do not call R3 leak-free.
- All full-data, synccheck, exact-output, two-buffer 4,000-invocation stress,
  GPU4, and stop rules remain unchanged.
- **Evidence/protocol:** rejected R2 log SHA-256 `6ae8d7e7...1f11c`;
  `PROTOCOL_H2B_P3_R3_ACCESS_SAFETY.md`.

## H2B-P3-R3 result: access safety and two-buffer stress (2026-09-03)

- **Decision:** pass the bounded device-access/synchronization and stability
  gate; admit P4 timing.  See `DECISION_H2B_P3_R3.md`.
- **Sanitizer:** both full target-8 audio/video operators report zero memcheck
  access errors and zero synccheck errors; all four instrumented outputs match
  P1 exactly.
- **Stress:** four/four cells retain one exact signature across 1,000 complete
  invocations while alternating two state buffers, 4,000 records total.
- **Postflight:** physical GPU4 returns to 14 MiB, 0%, with no process on its
  UUID.
- **Material caveat:** this is not leak-free.  The rejected full-leak R2 log
  retains three PyTorch-owned cuBLAS-handle allocations / 8,520,704 bytes.
- **Evidence:** safety summary SHA-256 `3a02d1a5...d245`; stress
  `bd3c4b1c...1225`; runner `63b19aab...45ba`.

## H2B-P4 pre-run: pedantic-cuBLAS outer-wall kill test (2026-09-03)

- **State:** frozen before any timing observation.
- **Denominator:** complete host outer wall from scan/dot through dynamic count,
  exact repair, final ID transfer, and canonical sort; one-time setup and
  post-return ID hashing excluded.
- **Fresh processes:** 8 process units, 10 warmups and 100 direction-balanced
  observations/variant/cell; process-paired bootstrap with 20,000 resamples.
- **Sustained:** two reversed 500+500-call blocks per cell, 1,000 retained
  observations per variant/cell.
- **Gate:** every process median wins, every paired 95% lower bound exceeds
  1.25x, every sustained block/combined ratio exceeds 1.25x, and every output
  signature remains exact.
- **Protocol/runner:** `PROTOCOL_H2B_P4_OUTER_WALL.md`,
  `src/run_h2b_p4_timing.py` (SHA-256 `b6ea11bc...74e0`).

## H2B-P4 result: Phase-P early-stop rejection (2026-09-03)

- **Decision:** reject H2B Phase P and stop streaming/tree work; see
  `DECISION_H2B_P4.md`.
- **Exact result:** all 800 retained complete outer-wall calls match frozen
  output/count signatures.  Median keeper/candidate speedups are 1.169x and
  0.666x for audio target 1/8, and 1.016x and 0.750x for video target 1/8.
- **Stop trigger:** no cell exceeds 1.25x, while both target-8 cells favor the
  pedantic keeper.  Paired candidate wins are 100/100, 0/100, 76/100, and
  0/100 respectively; order-stratified medians agree.
- **Not executed:** fresh slots 1--7, process bootstrap, sustained blocks,
  larger streaming, and tree/index work, exactly as required by the stop rule.
- **Diagnosis:** the pedantic keeper refines only 2/9/0/4 objects, versus the
  H1 candidate's 70/358/94/822; candidate FP64 panel repair dominates at
  target 8.
- **Evidence:** rejection SHA-256 `a61fbf98...2fe3`; slot-0 result/log
  `c8ac7d4e...c9c` / `68db0aa9...ba96`.
# G5 unified public artifact (started 2026-09-03)

| Experiment | Hypothesis | Contract | Keeper/candidate | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---|---|---|---|
| `tensorjoin_20260903_g5_unified_public_cifar60k` | The generated-code-audited G4C three-stage router can replace the older G2B candidate under the complete CIFAR-GIST-512 60K public denominator while retaining a decisive exact-system advantage. | `DESIGN_G5_UNIFIED_ARTIFACT.md`; `PROTOCOL_G5_COMPATIBILITY_SCREEN.md`; `PROTOCOL_G5_P1_R1_GPU2.md`; `PROTOCOL_G5_P3_SAFETY_STRESS.md`; `PROTOCOL_G5_P4_SCREEN.md` | Keepers: freshly measured exact FP64 MiSTIC and GDS-Join. Candidate: batched analytic INT8 certificate -> certified FP32 -> FP64 router. | All 1,800,030,000 upper pairs receive INT8 certificate work; 1,827,007 reach FP32 and 1,734 reach FP64. | Same 3,926,078 sorted directed IDs and hash `13cae87e...5963495`; same source, stage counts, cubin/PTX hashes across correctness/audit/safety/timing; zero overflow; GPU2 isolation. | `results/g5_{compatibility_tensorjoin_p1_r1_a0,p2_generated_code_audit_p1,p3_safety_manifest,p4_screen_summary}.json`; `raw/g5_*`; `artifacts/g5_*`; `DECISION_G5_P4.md`. | `rejected at P4`; P1--P3 pass, but paired speedups over MiSTIC are 4.092x and 0.951x, violating the every-round 1.50x gate despite 1.972x geometric mean. Formal/SIFT/Fashion stopped. |

# G6 latest available baselines (started 2026-09-04)

| Experiment | Hypothesis | Contract | Baselines | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---:|---|---|---|
| `tensorjoin_20260904_g6_latest_baselines` | The existing TensorJoin conclusion remains defensible after adding the newest reproducible GTS and cuVS comparators, while explicitly separating the unavailable RT-HiSS artifact from measured evidence. | `PROTOCOL_G6_LATEST_BASELINES.md`; `PROTOCOL_G6_R1_CONTEXT.md`; `RUN_CARD_G6.md` | GTS commit `3bac1b7`; `cuvs-cu13==26.8.1`; RT-HiSS literature-only gap | 0 for an exact admission | Same G2A/G2B source, radius, directed-ID encoding, float64 oracle, public host-to-sorted-host denominator, GPU1 isolation | `results/g6_{gts,cuvs}_*`; `raw/g6_*`; `artifacts/g6/*`; `DECISION_G6_LATEST_BASELINES.md` | No new exact keeper. GTS: G2A +2; G2B 4 FP/0 FN, 1111.520 s. cuVS: G2A -4; G2B 24 FP/26 FN, 3.551750 s. Both G2B times are one-run non-exact context only; RT-HiSS remains unmeasured without a public artifact. |


## G14 static-positive diagnostic (2026-09-05)

| Experiment | Hypothesis | Contract | Keeper/candidate | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---|---|---|---|
| `tensorjoin_20260905_static_positive_g14_cifar60k` | Phase timing and fixed placement can distinguish a persistent audited-runner cost from an unlocalized historical slowdown. | `g14_static_positive_diagnostic_20260905/PROTOCOL.md`; 12 independent processes, default/node-0 ABBA placement, original 60K host-to-canonical-host output. | Fresh MiSTIC and additive G2B/G5 diagnostic adapters; unchanged kernels. | Zero change in routing work versus each original. | Exact 3,926,078-ID hash, GPU isolation, G5 P4 binary identity, phase closure. | G14 `results/campaign.json`, `results/summary.json`, `raw/campaign.log`, `artifacts/raw_evidence_manifest.json`. | Diagnostic complete: all 12 admitted; G5 0.911–0.951 s and 4.99–5.88 same-block MiSTIC/G5 ratios; preparation/movement/sort 94.12–94.54%. Historical slow-mode cause inconclusive; no novelty/formal promotion. |


## G15/G16 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G15 strong control | CPU blocking removes preparation overhead; compare TC fairly with a complete FP32-first operator. | `g15_strong_control_20260905/PLAN_AND_GATE0.md`, `BASELINE_DESIGN.md`, results cpu/correctness/precision/public_screen, DECISION/CLAIM_EVIDENCE. | CPU integration local win; complete FP32-first decisively beats CPU-prepared TC. Preserve the failed TC-specific gate. |
| G16 GPU preparation | Build the unchanged G5 metadata on GPU, and grant the FP32 control equal GPU preparation. | `g16_gpu_preparation_20260905/PROTOCOL.md`, `ADDENDUM_FAIR_GPU_NORMS.md`, results metadata/safety/precision/public_screen/closure_checks, raw_evidence_manifest. | Narrow same-contract screen passes 1.499724/1.267190 versus faster control. No novelty/formal/sustained/multidataset promotion. |


## G17 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G17 RT-HiSS pair-contract admission | Preserve the native algorithm while exporting and independently auditing complete original IDs. | `g17_rthiss_pair_contract_20260905/PROTOCOL.md`, additive zero32 protocol, frozen hashes, 16 adapter runs, 1 same-launch diagnostic, code/safety/closure receipts. | Bounded decoder/structure/safety/code checks pass. Native CIFAR4096 has 2 FN + 2 FP despite equal counts; zero32 has 2 FP. Numeric failures retained. No exact-reference, performance or novelty promotion; separately designed two-sided repair next. |


## G18 evidence supplement (2026-09-05)

| Experiment | Hypothesis | Frozen contract / evidence | Decision |
|---|---|---|---|
| G18 RT-HiSS conservative two-sided repair | A bounded FP32 prefix/full-sum predicate plus selective original-order FP64 terminal can restore the frozen reference IDs without full-FP64 candidate refinement. | G18 PROTOCOL / NUMERICAL_DESIGN / explicit A1-A3 addenda; nine D512 inputs; 20 GPU3 main slots; exact hashes, four sanitizer runs, predicate stress, selected-code and closure receipts. | Bounded same-output admission: zero missing/extra IDs; real4096 uses 184/16,777,216 FP64 terminals (0.0010967255%), boundary_zero32 6.0546875%. No complete-cost timing or novelty admission. All failures and old evidence retained; next diagnostic-free artifact plus explicit engine-lifecycle fairness. |
