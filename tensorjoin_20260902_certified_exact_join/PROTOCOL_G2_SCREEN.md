# G2 public-path cheap-screen record

Date: 2026-09-03

Parent frozen protocol: `PROTOCOL_G2.md`.

Design and exact timing boundaries: `DESIGN_G2_TIMING.md`.

| Experiment | Hypothesis | Contract | Keeper/candidate | Expected count delta | Invariants | Raw evidence | Decision |
|---|---|---|---|---:|---|---|---|
| `tensorjoin_20260903_g2b_public_screen` | Output-sensitive INT8 certification plus narrow exact refinement retains enough Tensor Core benefit to beat both exact tree/index keepers after all public-path construction, transfer, materialization, and sorting costs. | CIFAR-10-GIST 60000x512 float32 source, epsilon 0.62890625, sorted directed uint64 IDs, public denominator in `PROTOCOL_G2.md`, physical GPU0, two fresh direction-balanced rounds. | GDS FP64 and MiSTIC FP64 / TensorJoin exact mixed precision. | 0 | Frozen count/hash; zero invalid/duplicate/missing-symmetry pairs; zero capacity overflow; clean GPU isolation. | Pending. | `designed`; screen thresholds frozen in `DESIGN_G2_TIMING.md`. |

The screen is not a substitute for the eight-round estimator and cannot support
a paper performance claim.

