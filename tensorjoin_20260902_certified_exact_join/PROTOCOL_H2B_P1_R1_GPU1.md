# Protocol Revision: H2B-P1-R1 Device Move to Physical GPU1

Date: 2026-09-03

## Reason

The frozen GPU3 target was rechecked before any H2B-P1 GPU execution.  It had
an external Python process (PID 4169406) using about 28 GiB at about 92--96%
GPU utilization.  The H2B runner was not launched and the external process was
not modified.

Physical GPU1 was simultaneously idle at 14 MiB and 0% utilization.  It is the
same `NVIDIA RTX PRO 6000 Blackwell Server Edition` model and SM120 target.

## Allowed revision

Only these fields change before the first P1 GPU launch:

- physical device: GPU3 -> GPU1;
- expected UUID: resolved live for GPU1;
- advisory lock: `.tensorjoin_gpu3_campaign.lock` ->
  `.tensorjoin_gpu1_campaign.lock`;
- P1 source/evidence hashes that necessarily encode the physical-device guard.

Data, split, semantics, error constants, cuBLAS API/math/compute types,
correctness/adversarial gates, output paths, and stop rules are unchanged.
Timing is not part of P1.  A later performance phase must rerun both keeper and
candidate from scratch on one frozen idle device; it may not mix H2A GPU3
timings with P1 GPU1 evidence.

## Evidence

The rejected-device preflight is retained in
`raw/h2b_p1_gpu3_foreign_preflight.txt`.  This is safety-control evidence only.
