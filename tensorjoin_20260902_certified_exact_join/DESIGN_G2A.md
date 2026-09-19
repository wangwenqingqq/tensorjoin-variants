# G2A Design Card: Scalable Canonical-Output Admission

Experiment ID: `tensorjoin_20260903_tensorjoin_cifar4096_g2a`

## Role and hypothesis

G2A is a correctness and capacity-mechanism gate, not a performance result. It
tests whether the accepted F1 precision cascade can leave the fixed
`512x4096`/65K-output prototype contract and return the same canonical self-join
IDs as exact external systems.

The single mechanism under test is an **overflow-detecting geometric-growth
append buffer**. It preserves the F1 numerical stages while replacing the fixed
65K allocation and 32-bit pair encoding. It does not add a full all-pairs
status tensor or precompute the oracle count.

## Frozen input and output

- Input: the immutable G2A 4,096x512 Cifar60K subset from
  `data/g2a_cifar4096/vectors_f32.npy`.
- Threshold: `effective_epsilon_d2` from the immutable G2A metadata.
- Oracle: 262,144 sorted directed IDs including self, SHA-256 recorded in
  `receipts/g2a_artifacts_sha256.json`.
- Pair representation: signed int64 on device, serialized/hashed as little-endian
  uint64 after validation. This representation remains valid for the full
  60,000x60,000 contract.

## Numerical stages

1. Per-vector signed INT8 codes, FP32 scales/reconstructed norms, and upward
   rounded FP32 residual-norm bounds, with the unchanged `1e-4` certificate
   pad.
2. Direct-difference FP32 evaluation of INT8-ambiguous IDs with the unchanged
   `1e-3` guard and exact equality path.
3. Direct-difference FP64 evaluation only for the residual guard band.

The G2A threshold and data are new; therefore all direct decisions and final
IDs are revalidated against the independent SciPy FP64 oracle. F1 correctness
does not transfer by assumption.

## Capacity state machine

- Start each execution at `8*N = 32,768` slots for each compact stream.
- Run the fused INT8 scan and read accept/ambiguity/overflow counters.
- On any overflow, discard all compact outputs, double capacity, reset counters,
  and rerun the unchanged scan.
- Stop only when both direct-accept and ambiguity streams fit, or fail closed if
  capacity would exceed `N*N`.
- The accepted capacity is then reused by FP32 and FP64 stages; any later
  overflow is a hard failure, not another tuning opportunity.

This deliberately forces the retry path at G2A scale if the data require it.
G2B may use a separately frozen sampling-based capacity estimate, but cannot
claim the G2A repeated scans as its performance path.

## Ownership and phase-live set

| Phase | Owner | Live state | Last use |
|---|---|---|---|
| quantization | host preflight | INT8 codes, FP32 scales/norms/errors | device upload |
| INT8 certificate scan | one 64x64 Triton program per tile | INT32 dot accumulator and bound operands | compact direct/ambiguous append |
| capacity control | host orchestration | five scalar counters and current capacity | successful scan selection |
| FP32 ambiguity filter | one Triton program per ambiguous ID | one 256-D reduction slice | accept or residual append |
| FP64 verification | one Triton program per residual ID | one 256-D reduction slice | final accept append |
| canonicalization | host validation | compact int64 IDs only | sorted oracle comparison/hash |

There is no `N x N` status object. The largest candidate-specific live objects
are the two capacity-bounded pair-ID buffers.

## Predicted work and falsification

- Invariant useful work: one successful dense INT8 scan plus direct FP32/FP64
  work selected by the certificate; exact final IDs.
- G2A-only added work: one or more discarded INT8 scans caused by the forced
  small initial capacity.
- Removed from the old prototype: fixed 65K admission boundary and int32 pair-ID
  range limit.
- Falsify if the final capacity exceeds `N*N`, any retry produces inconsistent
  counts, any stage overflows after admission, any unsafe direct decision is
  observed, or either isolated execution differs from the oracle/hash.

## Required gate

Two isolated executions must have zero missing/extra/duplicate/invalid IDs,
the same raw canonical hash as the oracle, identical work counts and accepted
capacity, and no foreign GPU process. Timing is diagnostic only.
