# Method and interpretation

This tests a supervised conditional flow in **solution-state space**. The input
is a tile's data summaries and a query threshold; the scalar state predicts
whether that 64-by-64 tile contains any true join entry. The initial state is
a deterministic coarse estimate derived from the same data, not a Gaussian.
The endpoint target is -1 for an empty tile and +1 for an occupied tile, from
archived complete FP64 reference results on training block pairs.

For a paired example (condition c, initial state a, target y), sample t uniformly,
set z_t=(1-t)*a+t*y, and regress v(c,z_t,t) to y-a. At inference actually solve
 dz/dt=v(c,z,t) from a. No oracle labels enter inference. This is a finite
supervised conditional interpolation experiment, not a claim of exact density
transport to a discrete distribution, nor of guaranteed exact neural outputs.
General CFM accepts non-Gaussian sources and paired endpoints; see
[Tong et al., sections 3.1–3.2](https://arxiv.org/html/2302.00482v4).
The deterministic coarse state, summaries, binary occupancy target and
certification policy are our TensorJoin design choices, not results of that paper.

A matched one-pass residual predictor learns y from the identical conditions and
initial state. All layers are trained. FM's state/time evolution is executed,
including its actual inference cost in every query. The condition-only affine
term of the first layer is cached within each inference chunk because it does
not depend on state or time; all subsequent vector-field evaluations are real.

Model predictions only allocate a stronger certificate to a fixed fraction of
coarse-bound survivors. No prediction alone can reject a tile. The certificate
checks projected distances between the actual 4096 point pairs, with conservative
projection, FP32 arithmetic and threshold margins. Any uncertified tile passes
to the unchanged arithmetic/refinement/output path. Compare all-certificate and
heuristic/random policies to separate certificate benefits from neural routing.
