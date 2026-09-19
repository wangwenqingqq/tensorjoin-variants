# Protocol D0: Learned Audio Embedding Breadth Kill Test

Experiment ID: `tensorjoin_20260902_panns_certificate_d0`

## Question

Does the per-vector INT8 certificate remain selective on a real pretrained
audio representation, or was A0's result specific to a handcrafted 1024-D
feature?

This is a certificate-geometry gate, not a GPU performance experiment.

## Frozen source and representation

- Dataset: the existing official ESC-50 checkout pinned to commit
  `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
- Model implementation: `qiuqiangkong/panns_inference`, commit recorded in
  `receipts/panns_inference_commit.txt`.
- Checkpoint: Zenodo record 3987831, `Cnn14_mAP=0.431.pth`, 327,428,481 bytes,
  official MD5 `541141fa2ee191a88f24a3219fff024e`.
- Each five-second ESC-50 clip is converted to mono FP32, resampled once from
  its source rate to 32 kHz, and split into five non-overlapping one-second
  segments.
- PANNs Cnn14 runs in evaluation mode and emits one 2048-D embedding per
  segment. Each embedding is L2-normalized without dataset centering.
- The cache must contain exactly 10,000 finite, nonzero vectors. Query and base
  samples are selected by seed `20260902` from disjoint original clips:
  `M=512`, `N=4096`, `D=2048`.

## Numerical contract

- Oracle: exhaustive squared Euclidean distance in NumPy FP64.
- Candidate geometry: unchanged per-vector symmetric INT8 quantization and
  residual-norm triangle bounds from A0.
- Thresholds are the globally calibrated order statistics yielding averages of
  1, 8, and 64 exact results per query.
- INT32 safety is checked as `D * 127^2 <= INT32_MAX`.

## Pass/stop rule

Pass only if all three thresholds have:

1. zero lower/upper containment violations;
2. zero direct false accepts and false rejects;
3. exact final classification after ambiguous FP64 refinement;
4. maximum ambiguity fraction at most 5% and median ambiguity fraction at most
   1%.

A pass admits generalized C0B execution at `D=2048`. A failure is retained as
negative evidence and stops PANNs from being used as a favorable breadth claim;
the thresholds or feature normalization must not be tuned after observing the
result.

## Reproducibility and safety

- Extraction uses physical GPU0 only after an idle check and while holding
  `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`.
- No clocks, power limits, or unrelated processes are changed.
- The embedding cache, checkpoint, scripts, source commit, environment, and
  result files are checksummed.
- D0 certificate evaluation is CPU-only with `CUDA_VISIBLE_DEVICES=""`.
