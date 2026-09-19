# H0 Multi-Vector Certificate Decision

Date: 2026-09-03

## Conclusion

**PASS the structural gate and admit an H1 GPU screen.**  The same per-token
INT8 residual intervals can be propagated through symmetric Chamfer aggregation
without losing exactness, and the resulting object router is selective on both
public-derived learned audio and learned video objects.

This is the strongest remaining novelty route because it changes the query
contract from single-vector join to variable-cardinality multi-vector join.
The novelty result is still conditional: H0 does not establish a first-work
claim, and the broad components are prior art.

## Frozen evidence

The direct-FP64-difference oracle covers 128 disjoint query objects by 512 base
objects per dataset.  Strict mid-gap thresholds yield exactly 1 and 8 object
results per query.

| Dataset | Tokens/object | Target | Ambiguous objects | Object ambiguity | Competitive FP64 cells / all token cells |
|---|---:|---:|---:|---:|---:|
| ESC-50/PANNs | 5 | 1 | 69 / 65,536 | 0.1053% | 1,167 / 1,638,400 = 0.0712% |
| ESC-50/PANNs | 5 | 8 | 356 / 65,536 | 0.5432% | 5,736 / 1,638,400 = 0.3501% |
| UCF101/R3D-18 | 4--7 | 1 | 94 / 65,536 | 0.1434% | 2,012 / 1,776,808 = 0.1132% |
| UCF101/R3D-18 | 4--7 | 8 | 821 / 65,536 | 1.2527% | 17,816 / 1,776,808 = 1.0027% |

All four cells have zero token- or object-interval containment violations,
zero unsafe direct accepts/rejects, zero final decision mismatches, disjoint
object sets, and safe INT32 accumulator ceilings.  Every predeclared 5%
object-ambiguity and 2.5% global-refinement gate passes.

## Mechanism interpretation

The useful new behavior is not simply quantized MaxSim.  Pairwise uncertainty
is lifted through `min` and `mean` to decide almost every object pair directly.
For the residual ambiguous objects, row-wise winner envelopes discard token
cells that cannot affect either directed Chamfer term.

The local refinement density inside an ambiguous object remains high
(64.45%--77.61%).  Therefore H1 must compact **object pairs first** and refine
small dense token panels; a global fine-grained sparse-token kernel is not the
default design.  This is a concrete routing law exposed by H0.

## Scope and caveats

- H0 is CPU-only opportunity evidence; no GPU kernel, SASS, sanitizer, stress,
  or latency result exists.
- The low-precision stage still scans every token cell.  Only FP64 work is
  reduced in H0.
- Audio objects are five one-second segments from each ESC-50 clip.  Video
  objects are four to seven UCF101 clip embeddings sharing a source-group ID.
  These are meaningful existing groupings, not a universal multi-vector data
  model.
- The literature search supports only a conditional gap.  Multi-vector
  retrieval, exact vector-set search, metric joins over Hausdorff objects, and
  accelerator MaxSim reranking already exist.

## Next action

Design H1 around bucketed object-pair tiles:

1. immutable FP64 full-token-panel keeper;
2. candidate INT8 token panel plus object-interval aggregation;
3. compact ambiguous object pairs, then refine their small dense token panels
   in FP64;
4. charge variable-length padding, dynamic counts, materialization, and final
   object IDs;
5. pass exactness and sanitizer before any timing promotion.

Do not spend effort on another single-vector pivot/cluster tree before H1.  The
current tree line is both structurally weak under E0 and novelty-crowded by GTS,
Tribase, and related lossless pruning systems.

## Evidence

- Result: `results/h0_multivector_certificate.json`, SHA-256
  `7d8b8e68c3ecaa6bb38de133e83afefcab4ddefa4b375572ba09fe3e9a68af31`.
- Raw log: `raw/h0_multivector_certificate.log`, SHA-256
  `b47bcfec4ee391cecab7ad70b704174383495be09175890006c23367fd6eac9c`.
- Runner: `src/run_h0_multivector_certificate.py`, SHA-256
  `eaca708df9ee2de45567f05044ba23b0674eea4f87ce0e3644ba44f9fd830191`.
- Host: `gpu-host-8`; CPU-only; 2026-09-03.

