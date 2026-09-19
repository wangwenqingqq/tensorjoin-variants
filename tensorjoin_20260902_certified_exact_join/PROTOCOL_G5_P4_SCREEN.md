# G5 P4: Full-Public External-System Cheap Screen

Date: 2026-09-03

## Admitted inputs

- G5 P1 exact result SHA-256:
  `7fcdd2786db413ee9f56347c27eceb9542e41e8fc644e9dd3a55d9ec59bff2a5`.
- G5 P2 generated-code audit SHA-256:
  `ffcfdf9c4949756ad1de8974f7e4453378dc26690abba3c292a551553c48482b`.
- G5 P3 safety manifest SHA-256:
  `6c6b0dfc52c33f91c6bd2d95331254838de5379c24299c3c88b87fafc81895d0`.
- Candidate runner SHA-256:
  `a8653469a0e5bae82b64cd3d9a82c476df741783735c1b9d699e2f27b2db13a5`.
- GDS wrapper/library SHA-256:
  `786053df96026bc0daeda146324de0f26464e2cc629f1c220cee8113238f4a96` /
  `e0d31f51a8a2ed844b988b2c2b1c08ed9757a55dfceda338c787d999b34b54bb`.
- MiSTIC wrapper/binary SHA-256:
  `83f4141ec87c74a1f288ce82e706bd40327367a849118278e046bdc8fe75339e` /
  `20739abefe2fc21cfb93ec70b0c9cddbde20b76dc896243c72336981c402326a`.
- Guard runner SHA-256:
  `8d7de27698313a31777d08b315d7fbafde9fd8a0cca5f50c8fd54eced3009126`.

All five frozen arithmetic/quantization upstream hashes remain those in
`DESIGN_G5_UNIFIED_ARTIFACT.md`.

## Measurement

Run on isolated physical GPU2 with one fresh process per slot and a 30-second
quiescence interval before every process.  Method orders are:

```text
round 0: gds, mistic, tensorjoin
round 1: tensorjoin, mistic, gds
```

Every method uses the same CIFAR-GIST-512 60K source, epsilon, exact output,
and `pageable host float32 -> sorted canonical host uint64 IDs` denominator.
TensorJoin uses a fresh Triton cache in each slot; compilation remains outside
the public timer.  GDS and MiSTIC are freshly measured; no G2B time is reused.

## Gate

P4 passes only if:

1. all six child and guard records are clean, isolated, and exact;
2. every TensorJoin stage count and source hash matches P1;
3. every TensorJoin cubin/PTX hash matches P1/P2/P3;
4. the same external library/binary hashes are used in both rounds;
5. TensorJoin / faster-exact-keeper speedup is at least 1.50x in each matched
   round and its paired geometric mean is at least 1.60x.

This is a diagnostic screen and cannot directly support paper-facing
performance prose.  A pass admits the frozen eight-round formal campaign; a
failure stops G5.

