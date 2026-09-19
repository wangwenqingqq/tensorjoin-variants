# G12 Strong Baseline: Factor Tiles -> Predicate -> Output DISTINCT

This is a comparator construction, not our claimed new algorithm.

## Query and identity

Let A_w = Qw x Bw, E = union_w A_w, and P(q,b) be a deterministic predicate
depending only on the endpoint vectors. Let delta remove duplicate endpoint
pairs. Under the frozen existential/set contract:

```
sigma_P(delta(union_all_w A_w))
  = delta(union_all_w sigma_P(A_w)).
```

Proof: either side contains `(q,b)` exactly when at least one A_w contains the
pair and P(q,b) is true. Witness multiplicity does not change either condition.
This is an elementary set/relational-algebra identity, not a new theorem.

The identity does not eliminate pair arithmetic. It moves global endpoint
deduplication after an endpoint-only predicate. It cannot be generalized
unchanged to bag outputs, witness-dependent predicates after projection,
non-deterministic functions, or top-k truncation.

## Execution

```text
construct/receive witness endpoint lists
for each witness w:
    for each query tile I in Qw:
        for each database tile J in Bw:
            gather/pack vectors for I and J
            evaluate the squared-distance panel
            immediately test the threshold
            insert only passing endpoint pairs into output hash set
export sorted unique output pairs
```

Only tile position and the endpoint lists are needed to identify a candidate
pair. There is no need to form a global candidate-pair list, mask, ownership
table, or CSR before evaluating distances. An actual high-performance grouped
GPU implementation could stream bounded descriptor batches; building a
global descriptor for every tile is also not a semantic requirement.

## Full ledger, including costs not removed

Let F count factor endpoint IDs plus witness metadata, W=sum|Qw||Bw|, Z be
unique passing output pairs, and W+ be passing witness occurrences.

| Item | Baseline cost | Can the proposed story claim to remove it? |
|---|---|---|
| Original graph and vectors | Retained; index construction separately charged | No |
| Factor lists | O(F) storage; construction and intersection charged | No: standard factorization |
| Global pre-predicate candidate IDs/CSR | Not required by this plan | No: an absent cost is not a contribution |
| Score/input tile workspace | O(t^2+tD); per-active-worker for parallel execution | No: ordinary blocking |
| Output set and export | O(Z) retained endpoint keys plus sort/export storage | Not for free; output size and DISTINCT remain |
| Score evaluations | W, with additional tile padding on a TC implementation | Some repeats could be removed, but G11 common-reference W/E is only about 1.36 |
| Output hash attempts | W+, not W or E in general | Must count repeated accepts and hash contention |
| Vector loads and packing | Depend on tile geometry, ordering and reuse | Real potential cost; no new differentiated mechanism identified yet |
| Numerical decision/refinement | Required by the separately frozen contract | Cannot be waived by this algebraic rewrite |

The memory bound O(F + tD + t^2 + Z), excluding shared graph/vector inputs,
is a logical live-set bound for a serial reference. It does not assert a
GPU runtime's allocator footprint, number of parallel tiles, or total data
movement. For dense output, Z can equal E and this advantage can disappear.

## Comparison suite for any future differentiated candidate

1. This standard streamed factorized plan, including accepted-output DISTINCT.
2. Deduplicated eligibility CSR + sparse/gather evaluation (one distance per E).
3. Block-mask/grouped dense scoring with all mask/packing construction charged.

These plans trade arithmetic, locality and metadata. No universal winner is
asserted. A later experiment must measure the strongest same-contract plans,
not present an unknown implementation as an already defeated comparator.

## Current novelty diagnosis

The proposal "keep groups instead of building all candidate pairs" is already
achievable by the comparator above. It is not a new cost-removal mechanism.
Replacing its compute backend with Tensor Cores is engineering until a
differentiated mechanism and decisive same-contract advantage are established.

The tempting alternative "compute only factor summaries" is not generally
exact: a nonlinear pair threshold cannot be recovered from arbitrary aggregate
feature sums. The G12 negative control makes this information-loss boundary
explicit; it does not rule out richer bounds or workload-specific structure.
