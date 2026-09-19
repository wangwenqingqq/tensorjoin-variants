# Source Ledger

Last checked: 2026-09-03

## Prior art

1. Thomas D. Ahle and Francesco Silvestri. *Similarity Search with Tensor Core
   Units*. arXiv:2006.12608. https://arxiv.org/abs/2006.12608
2. Benoit Gallet and Michael Gowanlock. *Leveraging GPU Tensor Cores for Double
   Precision Euclidean Distance Calculations*. arXiv:2209.11287.
   https://arxiv.org/abs/2209.11287
3. Brian Curless and Michael Gowanlock. *Fast and Scalable Mixed Precision
   Euclidean Distance Calculations Using GPU Tensor Cores*. ICPP 2025;
   arXiv:2508.21230. https://arxiv.org/abs/2508.21230
4. FaSTED reference implementation. https://github.com/bwcurless/fasted
5. Jianyang Gao and Cheng Long. *RaBitQ: Quantizing High-Dimensional Vectors
   with a Theoretical Error Bound for Approximate Nearest Neighbor Search*.
   arXiv:2405.12497. https://arxiv.org/abs/2405.12497
6. *Exact and/or Fast Nearest Neighbors* (Certified Cosine).
   arXiv:1910.02478. https://arxiv.org/abs/1910.02478
7. NVIDIA cuVS, vector search and refinement documentation.
   https://docs.nvidia.com/cuvs/getting-started/introduction/vector-search
8. James Diffenderfer, Daniel Osei-Kuffuor, and Harshitha Menon. *A
   Framework for Error-Bounded Approximate Computing, with an Application to
   Dot Products*. SIAM Journal on Scientific Computing 44(3), 2022.
   https://doi.org/10.1137/21M1406994
9. Muhammad Aamir Cheema et al. *A Survey on Query Processing in Vector
   Databases*. 2026 manuscript, Section 6 similarity joins.
   https://xiejiadong.github.io/files/paper/vector_survey.pdf
10. Michael Gowanlock and Ben Karsin. *Accelerating the Similarity Self-Join
    Using the GPU*. Journal of Parallel and Distributed Computing 133, 2019.
    Exact GDS-Join implementation: https://github.com/mgowanlock/gpu_self_join
11. Brendan Donnelly and Michael Gowanlock. *MiSTIC: A High-Performance
    Similarity Self-Join for Multidimensional Data on the GPU*. HiPC 2024.
    https://jan.ucc.nau.edu/mg2745/publications/Donnelly_Gowanlock_HiPC2024.pdf
12. MiSTIC reference implementation.
    https://github.com/bwd29/self-join-MiSTIC
13. Faizan A. Khattak and Mantas Mikaitis. *Accurate Models of NVIDIA Tensor
    Cores*, version 4. arXiv:2512.07004. https://arxiv.org/abs/2512.07004
14. NVIDIA. *Parallel Thread Execution ISA*, including integer MMA and
    `sqrt.approx` semantics. https://docs.nvidia.com/cuda/parallel-thread-execution/
15. Jiadong Xie, Yingfan Liu, and Jeffrey Xu Yu. *A Survey on Query
    Processing in Vector Databases*. 2026 manuscript.
    https://xiejiadong.github.io/files/paper/vector_survey.pdf
16. Laxman Dhulipala et al. *MUVERA: Multi-Vector Retrieval via Fixed
    Dimensional Encodings*. NeurIPS 2024. https://arxiv.org/abs/2405.19504
17. Yao Tian et al. *GEM: A Native Graph-based Index for Multi-Vector
    Retrieval*. 2026. https://arxiv.org/abs/2603.20336
18. Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. *Efficient
    Multi-Vector Dense Retrieval Using Bit Vectors*. ECIR 2024.
    https://arxiv.org/abs/2404.02805
19. Michael Leybovich and Oded Shmueli. *Efficient Approximate Search for
    Sets of Vectors*. 2021. https://arxiv.org/abs/2107.06817
20. Edwin H. Jacox and Hanan Samet. *Metric Space Similarity Joins*. ACM
    TODS 33(2), 2008. https://doi.org/10.1145/1366102.1366104
21. Tony Joseph et al. *AMES: Approximate Multi-modal Enterprise Search via
    Late Interaction Retrieval*. 2026. https://arxiv.org/abs/2603.13537
22. Qian Xu et al. *Tribase: A Vector Data Query Engine for Reliable and
    Lossless Pruning Compression using Triangle Inequalities*. SIGMOD 2025.
    https://doi.org/10.1145/3709743
23. Yitong Song et al. *TRIM: Accelerating High-Dimensional Vector
    Similarity Search with Enhanced Triangle-Inequality-Based Pruning*. 2025.
    https://arxiv.org/abs/2508.17828

## Dataset

1. ESC-50 official repository and dataset documentation.
   https://github.com/karolpiczak/ESC-50
2. Pinned source commit:
   `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`
3. Dataset license: Creative Commons Attribution-NonCommercial 3.0. The ESC-10
   subset is Creative Commons Attribution 4.0. This experiment is research-only
   and does not redistribute the dataset.
4. Cifar60K public download page used by the FaSTED/GQR experiments.
   https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html
5. Cifar60K archive and extracted fvec hashes are recorded in
   `receipts/g2_cifar60k_sha256.txt` and
   `receipts/g2_cifar60k_shape.json`. No license statement was found on the
   source page, so the dataset is internal research input and is not
   redistributable by this project.

## Evidence classification

- The prior-art statements above are literature findings.
- The triangle-inequality interval is an established mathematical fact applied
  to this execution design.
- Novelty is an inference from the searched literature, not proof that no
  unpublished or unindexed work exists.
- Performance and ambiguity claims remain unknown until measured under the
  frozen contracts.
