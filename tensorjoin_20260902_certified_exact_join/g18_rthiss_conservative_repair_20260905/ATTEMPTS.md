# Append-only attempt ledger

- Pre-GPU build A0 used the wrong CMake cache key `DATASET_DIM=512`; upstream
  actually consumes `DIM`, so that build compiled the default D18. Source and
  flags inspection caught the unused-variable warning before any GPU launch.
  The complete build/log/cache remain in `build_a0`, `raw/configure.log` and
  `raw/build.log`; this artifact is NOT admitted. A separate `build_r1` uses
  `-DDIM=512`. Freeze explicitly asserts `-DDATASET_DIM=512` in generated flags,
  along with gradual-underflow and default-kernel resource admission. No old
  benchmark, data, threshold or measured result is replaced.

- The initial continuation deployment encountered a jump-host connection reset.
  The first rsync failed, so the `&&` chain did not launch a GPU child. The same
  deployment was retried without changing topology or scientific parameters.
- A foreign Physion process occupied GPU 2 during the original signed31 slot's
  preflight. No child or numerical observation occurred. A1 preserves the two
  earlier clean passes and the failed preflight, then resumes the unmeasured
  slots only. See ADDENDUM_OCCUPANCY_A1.md and both campaign receipts.

- A2 zeros31 returned the exact961-ID reference output but failed the extra
  cross-run point-map byte-identity proxy. That combined guard remains failed;
  A3 explicitly revises the proxy to semantic map validation, not weaker output.
- Native map diagnostic d0 returned correct output and a different valid native
  permutation, but its guard saw an unattributed retiring PID1805357 (`[No data]`)
  and did not admit isolation. Do not relabel it clean or infer a foreign owner.
  Diagnostic r1 keeps its exact direct child unreaped for1 second after exit,
  allowing the unchanged guard to attribute a retiring CUDA PID; a180-second
  watchdog can kill only that owned child. It does not change GPU computation
  or safety policy and supplies no latency claim. Two new native-control IDs
  are required; d0 stays unadmitted.
