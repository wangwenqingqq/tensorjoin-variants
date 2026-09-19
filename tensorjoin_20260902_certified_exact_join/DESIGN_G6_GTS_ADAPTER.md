# G6 GTS pair-export adapter design

Date: 2026-09-04

Purpose: compatibility and evidence collection only.  This adapter does not
optimize GTS and does not alter its float32 search semantics.

## Target and mechanism

- Hardware: RTX PRO 6000 Blackwell Server Edition, SM120, physical GPU1.
- Toolkit: CUDA 13.1.115.  Build with C++17 because CUDA 13.1 CCCL rejects the
  upstream C++14 setting; target `sm_120`.
- Hypothesis: a result-export-only patch can expose the pair IDs already marked
  by upstream `dataProcessRnn` without changing candidate generation, pruning,
  distance accumulation, or threshold decisions.
- This is not a performance optimization.  Added compaction and D2H work is
  deliberately inside the common public denominator.

## Semantic and layout contract

- Input dtype/layout: contiguous row-major float32, `N x D`, loaded into
  pageable host memory before the timer and copied to device after the timer.
- Metric: upstream GTS L2 path with float32 accumulation, square root, and
  float32 epsilon.
- Query IDs: `0..N-1` in order.
- Output: sorted host `uint64`, where each key is `query_id * N + data_id`.
- The adapter never changes a GTS acceptance flag.  It only maps accepted flag
  slots to source and query IDs.

## Shape and launch roles

| Component | Grid/CTA role | State written |
|---|---|---|
| Upstream index/search | unchanged from GTS | tree, traversal list, acceptance flags |
| `compactAcceptedPairs` | one grid-stride thread per leaf-slot; 256 threads/CTA | one output key for every flag equal to one |

The supported frozen shapes are G2A `4096 x 512` and G2B `60000 x 512`.
There is no ragged dimension dispatch, tensor-core instruction, shared-memory
tile, swizzle, cluster, or software pipeline in the new kernel.

## Ownership and live state

| Object | Owner | Lifetime / last use |
|---|---|---|
| GTS `p_list` leaf flags and node/query maps | upstream search batch | compact kernel completion |
| Batch output buffer | adapter search function | batch D2H completion |
| Device atomic counter | compact kernel | counter D2H completion |
| Host pair vector | adapter main | canonical sort/write completion |

Per compacting thread, the live state is one slot index, one flag, leaf/query
map offsets, one node, one data ID, one query ID, and one uint64 key.  No
persistent accumulator or cross-CTA reduction state exists.  Output positions
are uniquely owned through one `atomicAdd` per accepted flag.

## Ready graph and synchronization

```text
dataProcessRnn completion
  -> device synchronize (upstream boundary)
  -> reduce flag count
  -> allocate exact-size batch buffer and reset counter
  -> compactAcceptedPairs launch
  -> device synchronize / error check
  -> copy counter and exact keys D2H
  -> append to host vector
  -> next traversal batch may reuse p_list
```

No `p_list` overwrite is allowed before the compact kernel and D2H copy finish.

## Verification ladder and rejection

1. Show an exact diff against the upstream commit and prove all edits are build
   compatibility, raw input, output export, receipts, or error handling.
2. Compile for SM120 and retain compiler output plus source/binary hashes.
3. Run G2A under the GPU1 lock and compare every canonical pair ID with the
   frozen float64 oracle.
4. Only after G2A exactness, run memory-access sanitizer if capacity permits.
5. Admit G2B diagnostic timing only after the earlier gates pass.

Reject the adapter if compacted count differs from upstream flag reduction,
the atomic counter overflows capacity, any invalid/duplicate ID appears, the
exact count/hash differs, or sanitizer reports an access/synchronization error.
If the mismatch comes from upstream float32 arithmetic, classify GTS as a
non-exact empirical comparator under this contract rather than repairing it
inside the baseline.

