# Additional projected-pair certificate

The inherited input/range and terminal-predicate scope in
inherited/NUMERICAL_SCOPE.md remains in force. For the represented PCA matrix P,
||P||_2^2 <= scale2. The archived FP64 projection q approximates xP coordinatewise
within PROJ_PAD=2^-24. Let f be q rounded to binary32. Define each coordinate's
maximum delta = max_rows |q-f| + PROJ_PAD, expanded upward, and span as the
maximum difference between represented f values, also expanded upward.

The true projected squared distance can be smaller than the exact squared
distance between f values by at most
 E_conversion = 4 sum_k delta_k (span_k + delta_k).
For <=32 direct binary32 subtractions and FMA accumulation of nonnegative
squares, gamma_128 * sum_k span_k^2 is a conservative absolute arithmetic
allowance, with gamma_128=128*u/(1-128*u), u=2^-24. A further 2^-20 absolute pad
covers tiny underflow terms and CPU rounding of these scalar quantities. No
cancellation-prone Gram formula or Tensor Core approximation is used here.

Each GPU block computes the minimum accumulated projected squared distance
over all valid 64-by-64 pairs. Let s be that minimum. Then
 L2=max(0,s-E_conversion-E_arithmetic-2^-20)/scale2
is the additional original-space squared-distance lower bound, under the
same numerical engineering assumptions as the inherited projection bound.
Reject a tile only if L2 > T+2^-20. Diagonal tiles are never rejected.

The kernel may stop after 8, 16 or 24 dimensions when every pair already clears
the same conservative test. Partial sums remain lower bounds; the global
32-coordinate error allowance is retained. Otherwise all 32 coordinates are
used. Invalid rows in the final ragged tile are masked to +infinity before the
minimum. Omitted tiles are always passed to the inherited complete solver.

Before training, verify the GPU minima against CPU FP64 direct computations on
random and adversarial cases, including ragged blocks, near-threshold pairs,
and partial/full evaluation equivalence. After selection, check full-data
reference occupancy, and compare actual complete output arrays at admission.
Empirical checks complement the conservative argument; neither the old system
nor this addition is presented as an unconditional exact-real theorem.
