# G5 P1 R1: Move the Unstarted Campaign to Physical GPU2

Date: 2026-09-03

## Why R1 is required

The first P1 guard entered its required 30-second GPU1 quiescence interval but
observed a foreign process using 552 MiB before the TensorJoin child started.
The guard failed closed.  No candidate kernel, public timer, or output was
produced, and no foreign process was signalled.

Retained invalid-preflight evidence:

| Artifact | SHA-256 |
|---|---|
| `raw/g5_p1_compatibility_a0_preflight.log` | `c536ada5b234eb9fb4ff6dad11cb5e018002735a2af5408224b921d3198ca7c0` |
| `raw/g5_p1_compatibility_a0_occupancy.jsonl` | `255b7f50df8ea3431176dca49d562e94b9de4737bc52d47bddd544d992877970` |

## Active replacement contract

- Physical GPU2, UUID `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245`.
- Same RTX PRO 6000 Blackwell Server Edition model, host, driver, toolkit,
  clocks policy, source, data, specialization, denominator, and gates as P0.
- New lock: `/tmp/tensorjoin_g5_gpu2.lock`.
- Every R1 label and artifact is new; the invalid attempt is never overwritten
  or counted.
- Abort only the owned G5 process group if a foreign GPU2 process appears.

This host-placement change occurs before measurement and does not change the
candidate mechanism or acceptance thresholds.

