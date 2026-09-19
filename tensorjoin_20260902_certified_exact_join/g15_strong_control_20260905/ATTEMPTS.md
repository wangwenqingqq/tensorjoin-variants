# Attempt ledger

- `g15b_validation_a0`: deployment/launch failure, not a numerical experiment.
  The first rsync connection closed before new baseline files were delivered.
  A shell command separated by semicolons still entered the guard, and the
  child then reported the missing runner. GPU memory/use remained zero for
  that child, no compute PID was admitted, and return code was 2. Preserve
  `../raw/g5_g15b_validation_a0*` and its guard record. This is not a timed
  sample and is not a mechanism rejection.
- Before `g15b_validation_a1`, retry transport and verify all 23 frozen
  source/design files on the destination. Subsequent deploy/launch chains
  use fail-closed command chaining. A1 passed all seven real/adversarial
  fixtures with zero interval, direct-decision, rectangular mapping or final
  output mismatch. Formula and kernel were not modified to obtain that pass.

Safety scope: planned memcheck is device-access checking with leak checking
disabled; synccheck covers tested synchronization. The prior H2B library-handle
leak finding remains a limitation, not a new leak-free claim for this adapter.

- Precision summary A0 stopped on unequal naive SASS hashes. Inspection found
  exactly 72 differing BRA/BSSY operands, all absolute relocated PCs of the
  same 2424-instruction function. R1 rebases only such validated control-flow
  destinations to function-relative offsets, preserving all other operands.
  Original raw exports, source and failed log remain. No GPU/kernel/radius
  change or profiler recollection was made to obtain a matching identity.
