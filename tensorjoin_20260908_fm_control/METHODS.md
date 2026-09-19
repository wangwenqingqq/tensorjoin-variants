# Method provenance and experimental interpretation

Flow matching regresses a time-dependent vector field using sampled probability
paths and their conditional velocities. Deployment integrates the learned ODE.
See [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

We use the zero-noise minibatch OT-CFM construction: match a batch of source
points u to target points g, sample t, interpolate z_t=(1-t)u+t*g, and regress
v_theta(z_t,t) to g-u. CFM permits general source/target distributions; here
the direction is data to Gaussian. The batch assignment approximates the
coupling used for learning, not the global OT plan. See sections 3.2.2–3.2.3
of [Tong et al., Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/html/2302.00482v4).

Our choices of Gaussianization, network size, data normalization, layout rules,
training budget and certified-pruning model selection are experimental design
choices for TensorJoin. The cited papers do not establish that these choices
improve this join task. No transformed-space distance can authorize pruning.

The one-pass endpoint regressor shares the same OT pair batches and similar
parameter count. It learns a different regression problem and has no ODE.
Comparing it with FM tests these two specified constructions; it does not rank
all one-pass models against all possible flow-matching models.
