# Protocol D2: Open Video-Embedding Breadth Gate

Experiment family: `tensorjoin_20260903_ucf101_r3d18_d2`

## Stage and purpose

D2 is a breadth test, not a paper claim. It asks whether D1's exact fused
Tensor-Core mechanism survives a different modality and embedding family.
All data/extraction choices below are frozen before feature measurement.

## Open data and model

- Dataset: UCF101, 13,320 AVI files. The UCF endpoint is the canonical source,
  but its 6,932,971,618-byte RAR was measured at only about 70 KB/s from the
  experiment host. Acquisition therefore uses the fixed community ZIP that
  reports an unmodified conversion of the official videos, pinned to Hugging
  Face commit `04d4e5ca1dc93606cb58752b0c08331e598743a4`. The direct endpoint
  resets from this host, so transport uses `hf-mirror.com`. Require linked size
  6,957,373,664 and linked SHA-256
  `eb77e54dfafd9c77e7b086f5e5ef7738cabd16a701b3f5c301a1f6c81fc4756e`,
  then record HTTP headers and a sorted relative-path manifest.
- Model: torchvision `r3d_18` with
  `R3D_18_Weights.KINETICS400_V1`; record the exact checkpoint URL, file
  SHA-256, torchvision/PyTorch versions, and extraction-script SHA-256.
- Representation: replace only the final classification layer by identity and
  retain its 512-D penultimate embedding. No fine-tuning.

## Deterministic clip contract

- Enumerate all AVI files by bytewise-sorted relative path.
- Decode RGB frames with the environment's fixed OpenCV build. Select 16
  consecutive frames centered on the decoded video midpoint; edge indices are
  clipped, and unreadable videos are recorded rather than silently replaced.
- Apply the selected weight object's official resize, center-crop,
  normalization, and channel layout. Run deterministic FP32 inference with
  TF32 disabled, then L2-normalize each embedding and store FP32 values.
- Process every decodable video. The output cache includes relative path,
  action class, decoded-frame count, selected-frame indices, and embedding.
  Refuse fewer than 4,608 valid videos.

## Geometry gate (D2A)

- Seed `20260903`; parse each UCF filename's `gXX` source group, then sample
  512 query videos and 4,096 base videos from disjoint source groups without
  class filtering or replacement. This prevents clips cut from the same long
  source video from crossing the two sides.
- Direct blocked FP64 squared differences define the oracle.
- Evaluate tie-aware mid-gap radii at nominal 1 and 64 results/query. Report
  actual counts and all exact ties.
- Reuse per-vector signed-INT8 quantization and the padded accept/reject/
  ambiguous certificate without retuning its `1e-4` pad.
- Pass: zero unsafe direct decisions and final mismatches, ambiguity <=10% at
  both radii. Failure stops D2B rather than tuning the certificate on UCF101.

## End-to-end gate (D2B)

- Shape `512x4096x512`; resident-input/output timing; physical GPU0 only.
- Keeper: D1's exhaustive non-TF32 FP32 scan with `1e-3` boundary guard and
  direct FP64 refinement.
- Candidate: D1's fused INT8 IMMA scan, compact outputs, and direct FP64
  refinement, specialized only for D=512. No threshold-dependent tuning.
- Before formal timing: exact pair IDs, zero unsafe decisions/duplicates/
  overflows, clean memcheck, one output hash over 1,000 launches, and SASS
  evidence for INT8 IMMA plus FP64 refinement.
- Formal timing: both radii, 20 warmups/100 observations, eight fresh processes,
  `AB,BA` repeated. Pass requires exactness, >=7/8 wins, and a deterministic
  process-bootstrap 95% lower speedup bound >=1.5x at both radii.

## Stop boundary

A D2 pass admits the index-aware stage; it does not establish novelty or a
competitive database-system result. The index stage must freeze an external
same-output comparator and account for traversal, candidate generation,
Tensor-Core certification, exact refinement, and compact materialization.
