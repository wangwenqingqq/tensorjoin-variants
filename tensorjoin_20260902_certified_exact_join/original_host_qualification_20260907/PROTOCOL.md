# Original-host frozen G16 qualification

Experiment: tensorjoin_20260907_original_host_g16d512.
Predeclared before any new GPU launch. No original kernel, proof, benchmark or
past record is replaced. Parent: frozen_gpu_boundary_20260907/delivery_a1.json.

## Execution card

- Topology: local Mac -> configured tiaoban -> gpu-host-8, root account.
- Workspace: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
- Device: original physical GPU2, GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245,
  RTX PRO 6000 Blackwell Server Edition, driver590.48.01. Fresh inventory finds
  it idle. GPU0 and GPU7 foreign processes remain untouched.
- Environment: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python; verify live versions
  in raw/preflight_a1.txt. No new cubin compilation.
- Code identity: frozen_manifest.json, four retained G16 cubins and24 associated
  metadata/IR files. New host code sources are hashed in artifacts/host_sources.json.
- Input/oracle: exact prior fixtures, N60000 D512 finite FP32 |x|<=1;
  epsilon161/256 and T25921/65536; inclusive fixed terminal. Full NPY SHA and
  expected directed canonical hash are unchanged from the sealed prior campaign.
- Locks: /tmp/tensorjoin_gpu2_campaign.lock and /tmp/tensorjoin_g5_gpu2.lock.
- Admission: 30s no foreign compute process before each child, monitor during
  execution; sampled GPU memory below4GiB; 30min per-process timeout. Abort and
  retain errors, terminating only the child process group started by this guard.
- Commands/PIDs/logs: recorded before each launch in raw/p8_* and results/*_guard.json.
- Rollback: no system changes; stop only owned child; preserve raw records.

## Required gates and ordering

1. Boundary62128 pairs, exact metadata105 rows,4 reordered passes,3 distinct
   vector addresses and8 immutable-input graph replays, same-terminal comparison.
2. Full retained public input:108 batches,1,800,030,000 upper candidates;
   3,926,078 directed output IDs and canonical SHA256
   13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495.
3. Separate memcheck, synccheck, racecheck and initcheck on the same bounded
   fixture with one reorder pass. Each requires tool-clean and numerical pass.
4. Actual stage2 append/reject composition on the prior18528-pair margin fixture.
5. Cross-process original code identity, cross-host output equivalence, exact
   residual metadata replay and original-file integrity before qualification.

No performance is promoted by these gates. Runtime parameter offsets/sizes,
launch128 threads and shared-memory sizes remain fixed. Known metadata row
admission and arbitrary-radius caveats remain; no invalid accesses are launched.

A separate performance record must freeze its own output contract, source and
binary binding, baseline precision/method, timing boundary, raw process order,
statistics and sustained rule. Correctness qualification is necessary, not a
substitute for that record. The historical5.083x mixed-artifact result remains
excluded; no paper/Overleaf/novelty claim changes are authorized by this record.
