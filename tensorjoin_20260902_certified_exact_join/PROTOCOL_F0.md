# Protocol F0: Three-Precision Cascade Opportunity

Experiment ID: `tensorjoin_20260903_precision_cascade_f0`

## Frozen inputs

Reuse the exact query/base splits and tie-aware nominal-1/64 mid-gap radii from:

- D1 learned audio (`512x4096x2048`);
- D2 learned video (`512x4096x512`);
- D3 hyperspectral patches (`512x4096x1984`).

## Cascade simulation

1. Recompute the unchanged `1e-4`-padded INT8 certificate and collect only its
   ambiguous pairs.
2. For an ambiguous pair, first mark bitwise-identical FP32 vectors as exact
   zero-distance accepts.
3. Otherwise compute direct-difference FP32 squared distance. Reuse the
   keeper's frozen `1e-3` boundary guard: accept below `threshold-guard`, reject
   above `threshold+guard`, and send only the band to direct FP64 refinement.
4. Compare every direct decision and final pair ID with blocked direct FP64.

F0 computes FP32 distances densely only for implementation simplicity, but its
work ledger charges them only for INT8-ambiguous IDs. It excludes GPU queueing,
gathering, kernel launches, and latency.

## Pass/stop rule

Every one of the six modality/radius cells must have zero false direct
decisions and final mismatches, and at most 25% of INT8-ambiguous pairs may
reach FP64. A pass admits a fused-kernel smoke test; it does not prove the
`1e-3` guard or establish speed. A failure stops this fixed cascade rather than
tuning the guard per dataset.
