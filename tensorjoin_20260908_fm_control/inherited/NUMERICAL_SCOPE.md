# Scope and conservative margins

All bound inputs are finite binary32 with |x|<=1 and D<=1024. Secondary integer
datasets are rescaled by an exact power of two. Original coordinates are never
replaced by learned/projected coordinates in the final distance predicate.

For direct FP64 squared distances, subtracting, squaring and reducing <=1024
terms of magnitude <=4 has a conservative standard roundoff allowance much
smaller than 2^-20: using u=2^-53 and gamma_(4D+16) gives <2^-28 in the worst
admitted range. The retained terminal PTX likewise converts binary32 to FP64,
subtracts, squares/FMA-reduces nonnegative terms and compares with binary32 T.
The original terminal error is therefore also below the separate 2^-20 pad.
Nonzero binary32 differences/squares are well within normal FP64 range.

We use direct FP64 differences (no Gram cancellation) and pad squared values
by +/-2^-20 before sqrt. Roots are expanded by nextafter. Ball radii use an
upper enclosure, center separations a lower enclosure. Subtraction of radii
has an additional 2^-30 distance pad; squaring the lower result subtracts
2^-20. Centers are arbitrary represented binary32 vectors; the mean need not
be the exact centroid for the ball argument to hold.

Original AABBs use exact min/max of represented coordinates, direct FP64 gaps,
and subtract 2^-20 from their squared gap sum. Pivot distance ranges use the
same padded roots and the reverse triangle inequality. Only the maximum
single-pivot separation is used, never an invalid sum over pivots.

PCA is used as a represented linear map P with verified |P_ij|<=1. For its
actual FP64 coefficients, ||P||_2^2 <= 1 + ||P^T P-I||_F. Add 2^-20 to the
computed right side to cover Gram and Frobenius rounding (32 columns,
D<=1024). Enclose each projected coordinate by +/-2^-24; this greatly exceeds
the FP64 dot-product forward error for the admitted range. Divide the projected
AABB squared lower bound by the resulting upper norm-squared allowance.

Reject a tile only if the resulting squared lower bound is strictly greater
than T+2^-20. Diagonal/self tiles must survive. Whole-reference occupancy checks
are a separate empirical validation for every tested layout/threshold/bound.
They are not used to select geometrically rejected tiles.

This is a conservative source-level engineering argument under ordinary FP64
CPU arithmetic; not a new numerical theorem. The retained F8/F16 comparator
and its unchanged Tensor Core numerical premise are inherited from the prior
experiment. Full output agreement is checked against the same frozen FP64
reference. Do not describe the full system as unconditional exact-real join.
