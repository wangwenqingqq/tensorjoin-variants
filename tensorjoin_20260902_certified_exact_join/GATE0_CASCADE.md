# Gate 0: Exact Three-Precision Cascade

Date: 2026-09-03

## Verdict

Multi-stage filter-and-verify is established prior art, so
**INT8 -> FP32 -> FP64 is not by itself a novelty claim**. Exact set-similarity
joins already use filter-verification pipelines, and vector systems commonly
rerank compressed candidates in higher precision. FaSTED also establishes
mixed-precision Tensor-Core vector joins, although its output is not exact.

The defensible systems extension is narrower: a fused, output-sensitive
precision cascade in which every stage has an explicit no-false-decision
contract, exact duplicate/tie handling, compact GPU queues, and a measured
router across audio, video, and scientific tensors.

## Nearest evidence

- [GPU exact-join evaluation](https://www.sciencedirect.com/science/article/pii/S030643791930537X)
  treats filter-verification and workload-dependent winners as established.
- [Bitmap Filter](https://www.sciencedirect.com/science/article/pii/S0306437919305010)
  shows another exact multi-filter GPU join; additional stages alone are not
  novel.
- [NVIDIA cuVS](https://docs.nvidia.com/cuvs/getting-started/introduction/vector-search)
  documents compressed candidate generation followed by more accurate
  reranking, but generally in approximate-retrieval contracts.
- [FaSTED](https://github.com/bwcurless/FaSTED) uses FP16 inputs, FP32 distance,
  and Tensor Cores for all-pairs Euclidean search, reporting nonzero result loss.

## Cheap kill

Run `PROTOCOL_F0.md` before coding a new kernel. If unchanged guarded FP32 does
not eliminate at least 75% of the INT8-ambiguous pairs on every frozen
audio/video/HSI radius, a third stage is unlikely to repay queueing and another
full-vector pass. If it passes, the next gate is a fused FP32 direct-difference
kernel plus a bit-level error analysis; opportunity counts are not performance.
