# G10 primary-source ledger

Checked 2026-09-04. This ledger records the sources actually inspected, not a
claim of exhaustive literature coverage. Dates below are publication or arXiv
submission dates, not search-engine crawl dates. No external code was run.

| ID | Primary source | Inspected evidence | Relevance and limit |
|---|---|---|---|
| S01 | [VA-file, VLDB 1998](https://www.vldb.org/conf/1998/p194.pdf) | Section 4, printed pp. 202–203: approximation scan, distance lower/upper bounds, candidate refinement | Quantized high-dimensional search followed by exact candidate distances is established. This is nearest-neighbor search, not our GPU self-join implementation. |
| S02 | [Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates, 1997](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf) | Abstract and adaptive-predicate description | Precision depending on predicate uncertainty and exact fallback is established. The demonstrated predicates are geometric orientation/incircle tests, not D512 joins. |
| S03 | [Fast Floating-Point Filters for Robust Predicates, BIT 2023](https://link.springer.com/article/10.1007/s10543-023-00975-x) | Sections 2, 3 and 4.1: general polynomial filters, staged sign/uncertainty interface, exact stages | A squared-distance threshold is a polynomial predicate. The source supplies a strong conceptual/adapted-method control, not a ready GPU D512 performance comparator. [Author preprint](https://arxiv.org/abs/2208.00497). |
| S04 | [QDOT / A Framework for Error-Bounded Approximate Computing, SISC 2022](https://epubs.siam.org/doi/10.1137/21M1406994) | Abstract and [2021 preprint](https://arxiv.org/abs/2105.00115): deterministic approximate-dot bounds and precision parameter selection | Generic error-bounded mixed-precision arithmetic is not new. Relative numerical-error control is not the same as complete exact join output. |
| S05 | [FaSTED, ICPP 2025](https://arxiv.org/html/2508.21230v1) | Sections 2.1–2.2, 3.1–3.3, 4.5–4.6 | Direct same-task prior art for TC high-dimensional distance self-join, tiled data reuse and mixed precision. It evaluates accuracy loss rather than giving our deterministic predicate contract. Do not equate this distinction with a novelty pass. |
| S06 | [Exact and/or Fast Nearest Neighbors / Certified Cosine, 2019](https://arxiv.org/html/1910.02478v2) | Sections 2–4 and evaluation caveat in Section 5 | Search-region certificates can guarantee a nearest-neighbor answer; the user can require a scan on certification failure. Budget-limited approximate results in its evaluation must not be described as unconditionally certified. This is search completeness, not low-precision arithmetic certification. |
| S07 | [RaBitQ, SIGMOD 2024](https://arxiv.org/html/2405.12497v1) | Theorem 3.2, Eq. 15 and its failure-probability discussion | Quantization with a theoretical error interval is established. The cited interval is high-probability, not an unconditional deterministic predicate certificate. |
| S08 | [TRIM, 2025 preprint](https://arxiv.org/html/2508.17828v1) | Sections 3.1–3.2: reconstructed PQ landmarks, strict triangle bound, separately defined p-relaxed bound | Quantized reconstruction plus residual/triangle bounds already appears in vector search. Strict and probabilistically relaxed bounds must be distinguished. No claim is made that its complete ANN traversal gives exhaustive join output. |
| S09 | [RT-HiSS, submitted 2026-09-02](https://arxiv.org/abs/2609.01975) | Abstract: RT-core candidate generation, CUDA refinement, output-size upper bound, batching and compressed masks | A current direct GPU similarity-search/self-join systems comparator. Our [pinned artifact audit](DECISION_G7_ARTIFACT_ADMISSION.md) still has only a count-based tiny smoke; same-contract pair exactness and performance remain pending. |
| S10 | [DGEMM on Integer Matrix Multiplication Unit, 2023](https://arxiv.org/abs/2306.11975), [Ozaki Scheme II, 2025](https://arxiv.org/abs/2504.08009) | Abstracts and method identities | Integer-unit emulation and decomposition are established families. These are not assertions that the papers implement G9's exact two-limb schedule or distance certificate. |
| S11 | [Tribase, SIGMOD 2025, author-hosted PDF](https://madsys.cs.tsinghua.edu.cn/publication/tribase-a-vector-data-query-engine-for-reliable-and-lossless-pruning-compression-using-triangle-inequalities/SIGMOD25-xu.pdf) | Indexed author-PDF abstract: refined cluster index and triangle-inequality pruning for ANNS | Additional nearby database work. Full PDF opening failed in this audit; no detailed theorem or numerical-contract claim is based on it. |

## Search routes and limits

The audit searched these families, then inspected primary sources rather than
relying on blog summaries: certified similarity join and quantization; QDOT;
exact range search with quantization/error bounds; adaptive robust predicates;
verified TC matrix multiplication; VA-file; distance lower/upper bounds; modern
TC and RT-core high-dimensional similarity join. Additional exact-title checks
covered FaSTED, TRIM, RaBitQ, Tribase and integer-GEMM emulation.

The 2001 *Adaptable Similarity Search Using Vector Quantization* author PDF
was found, but direct opening failed. It is a follow-up lead, not a necessary
premise of the decision. The GPredicates 2019 publisher page was located but
did not expose enough method text here; no detailed GPU scheduling claim is
derived from its title. Two guessed version-specific arXiv HTML pages failed;
their existing v1/abstract pages were inspected instead. Retrieval failure is
not evidence that an artifact or result does not exist.

No single inspected paper was shown to reproduce the entire current
INT8/FP32/FP64 self-join implementation with the same numerical and output
contract. Conversely, failure to find that exact combination is not proof of
novelty. The decision rests on identifiable component overlap and the absence
of demonstrated non-incremental interaction, not on a claim of literal copying
or universal impossibility.
