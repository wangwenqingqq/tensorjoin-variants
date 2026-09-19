# Design G4B-R1: ragged-safe public breadth matrix

Date frozen: 2026-09-03

G4B-R1 inherits the complete numeric, output, ownership, live-set, ready-graph,
and verification contract from `DESIGN_G4B.md`. It changes one mechanism: the
G3B operand load index and mask for a final partial K tile.

## Delta mechanism

The rejected G4B source increments operand pointers by `BLOCK_K=64` but masks
with the unchanged `offsets_k < K`. That is valid only when K is divisible by
64. G4B-R1 instead forms `k = block_start + offsets_k` inside every loop
iteration and uses `k < K` for both operands. Valid lanes retain exactly the
same row-major and K-major addresses; invalid tail lanes load zero.

| Native D | K blocks | Final valid lanes | Final masked lanes |
|---:|---:|---:|---:|
| 128 | 2 | 64 | 0 |
| 512 | 8 | 64 | 0 |
| 784 | 13 | 16 | 48 |

The CTA remains one 64x64 upper tile with four warps and three compiler-owned
stages. INT8 codes, INT32 accumulation, float32 interval arithmetic, residual
bounds, thresholds, counters, and buffers are unchanged. The correction adds
no new persistent state and changes no output or stage contract.

## Comparator and stop rule

- Keeper for multiple-of-64 controls: rejected G4B classifications may be
  compared diagnostically, but G4B-R1 must independently pass the exact oracle.
- Ragged comparator: the admitted in-process A/B result at Fashion
  `N=1024,D=784,k=1`.
- Reject on any of the original 27-cell exactness/structure failures.
- Reject generic timing if the unchanged cross-dataset selectivity thresholds
  fail.

This is an implementation correction and extensibility enabler, not a new
algorithmic novelty claim. Generated-code, sanitizer, performance, and
end-to-end admission remain separate gates.
