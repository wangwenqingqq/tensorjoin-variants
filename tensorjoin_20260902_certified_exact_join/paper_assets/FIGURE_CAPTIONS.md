# Paper-Facing Figure Captions

Date: 2026-09-03

## Figure 1: Evidence-separated teaser

**Certified routing concentrates high precision on the decision boundary, while
a separately scoped scalable implementation establishes external-system
impact.**  In the exact G5 full-scale mechanism run, only 0.1015% of 1.800B
upper pairs advance from INT8 Tensor Cores to certified FP32, and 1,734 pairs
advance to exact FP64 before reconstruction of 3,926,078 directed IDs.  Under
the separate G2B pageable-host-input-to-sorted-host-output contract, TensorJoin
has a 0.926 s median and is 5.083x and 11.783x faster than exact MiSTIC and
GDS-Join, respectively; the 0.229 s FaSTED point has 598 false negatives and
1,130 false positives.  Panels (a) and (b) use different frozen implementation
scopes and do not constitute a shared-binary claim.

## Figure 2: Exact precision-routing query plan

**TensorJoin executes precision escalation as a dynamic exact query plan.**
The full-scale G5 implementation quantizes 60,000x512 FP32 vectors, schedules
440,391 upper 64x64 tiles in 108 batches, classifies each INT8 Tensor-Core score
with an analytic interval, and reads the actual ambiguous counts before
launching certified FP32 and exact FP64 work.  Direct accepts bypass later
precision stages and join the accepted-ID path; host reconstruction mirrors
off-diagonal pairs, retains self pairs, concatenates IDs, and sorts the final
3,926,078-entry canonical output.  All populations are measured in G5 P1;
compilation is outside the depicted public timing scope.

## Main external-results table

**The scalable G2B implementation outperforms both exact external systems under
the complete public denominator, whereas the faster FaSTED point changes the
output set.**  Times are the marginal p10, median, and p90 across eight clean
admitted processes on CIFAR-GIST-512 60K.  Relative exact-system results use
paired round geometric means and 100,000-replicate 95% bootstrap intervals.
