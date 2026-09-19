# Superseded diagnostic evidence

These runs precede the final v1b threshold ABI and/or final harness.
The labels probe_r1, probe_cuda131 and validate_r1 used bundled ptxas 12.9,
not the system CUDA 13.1 compiler. Later pre-v1b runs used 13.1 but still had
the scalar-threshold ABI hazard. Their first-call timings include JIT/load.
Retain for provenance; do not use them for final correctness or speed claims.
