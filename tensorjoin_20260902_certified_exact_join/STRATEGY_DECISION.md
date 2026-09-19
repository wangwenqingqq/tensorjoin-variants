# Research Strategy Decision

Date: 2026-09-03

## Bottom line

The original single-vector campaign did not find a defensible top-tier novelty
thesis. It established a credible middle strategy: an **exact, output-sensitive
mixed-precision vector join operator** that converts low-precision Tensor-Core
scores into certified decisions and routes unresolved pairs by
tie/threshold regime. The frozen full-scale G2B campaign beats two external
exact GPU systems by a decisive same-contract margin. This promotes the work
from an in-house kernel observation to an evidence-backed database/systems
direction. G4B-R1 and G4C now establish public breadth and dynamic-router
benefit across three native-dimensional anchors. That single-vector novelty
remains an extension thesis. H0 opened a more novel conditional route: an
**exact variable-cardinality multi-vector similarity join** whose pairwise
certificates compose through Chamfer/MaxSim aggregation. H0's CPU structural
gate, H1-R1's exact GPU screen, and H2A's certified direct-FP32 comparison pass
on learned audio and video, but the decisive H2B kill test fails against a
runtime-resolved pedantic SGEMM/cuBLAS exact keeper. The upper multi-vector
performance thesis is therefore closed; streaming and tree/index work are
blocked. The recommended route returns to the evidence-backed single-vector
exact precision router as an extensible systems contribution, without claiming
a new exact-join principle.

## Upper strategy: new exact Tensor-Core join principle

**Status: Gate 0 not passed.** Tensor-Core similarity join is established by
[Ahle and Silvestri](https://arxiv.org/abs/2006.12608); mixed-precision brute
force Euclidean join is implemented by
[FaSTED](https://arxiv.org/abs/2508.21230); deterministic error-bounded dot
products exist in [QDOT](https://doi.org/10.1137/21M1406994); and exact
high-dimensional retrieval certificates exist in
[Certified Cosine](https://arxiv.org/abs/1910.02478). Exact certification plus
Tensor Cores may be a useful combination, but the components make a broad
"first exact low-precision search" claim unsafe.

Reopen only if a non-obvious certificate or execution primitive appears that
is not reducible to established error bounds plus filter/refine, and it yields
a decisive same-contract advantage.

## New upper candidate: exact multi-vector certificate algebra

**Status: H2B Phase P rejected; do not pursue as the paper thesis.** A current vector
query survey treats multi-vector retrieval and similarity join as separate
families, and the inspected closest work does not provide an exact all-pairs
threshold join over variable-cardinality dense-vector objects under
Chamfer/MaxSim semantics. Broad claims remain unsafe: MUVERA, PLAID, EMVB, IGP,
GEM, and AMES cover multi-vector retrieval and accelerator reranking; exact
metric and set joins cover broader object families.

The surviving mechanism lifts certified per-token intervals through directed
`min` and `mean`, decides object pairs from the resulting interval, and refines
only winner-competitive token cells in ambiguous objects. H0 passes on
ESC-50/PANNs and UCF101/R3D-18: only 0.105%--1.253% of object pairs remain
ambiguous and 0.071%--1.003% of all token cells require FP64, with zero observed
decision mismatch. The local ambiguous panels remain 64%--78% dense, implying
an object-first compaction and small-dense-panel refinement law rather than a
global sparse-token kernel.

H1-R1 supplies that exact GPU operator. Across the frozen 128x512 object
cells it is 13.707x--65.063x faster than exhaustive direct FP64 after charging
variable-length padding, dynamic count synchronization, and final canonical
IDs, with zero mismatch and clean bounded memcheck. The screen also confirms
signed INT8 Tensor Core lowering in SASS. H2A then replaces the weak comparator
with a certified direct-FP32 exact scan and retains 2.758x--10.455x outer-wall
speedups, 50/50 wins per cell, exact output, and clean bounded memcheck. This
closes the custom direct-FP32 kill test. H2B P0--P3 then confirms that a direct
pedantic-cuBLAS keeper is exact, executes an audited FP32 SIMT path, and passes
bounded access/stress checks. P4 nevertheless measures only 1.169x/0.666x on
audio target 1/8 and 1.016x/0.750x on video; both target-8 cells favor the
keeper. Under the predeclared stop rule this blocks fresh slots 1--7,
sustained, streaming, and any tree/index layer. H0--H2A remain useful scoped
mechanism/ablation evidence, not a surviving headline.

## Middle strategy: extensible exact precision router

**Status: promoted research direction with public breadth, but not yet
submission-ready.** Measured
evidence now spans PANNs audio, R3D-18 video, and hyperspectral patches. The
certificate is exact and selective on all three. The two-stage kernel passes
formal learned-audio timing, while video and HSI expose the same high-output
failure. F0 then localizes a coherent mechanism: ordinary target-64 cells need
FP64 for only 3.89%--8.02% of INT8 ambiguity, whereas a tie/near-zero cell needs
67.96%. F1 passes the fixed three-modality target-64 smoke, safety, and formal
gates. It yields 1.588x/1.690x/1.909x over the unchanged two-stage path and
2.439x/1.959x/2.210x over guarded FP32 for audio/video/HSI, with every
predeclared confidence lower bound passing. That is evidence for a regime-aware operator rather than a list of unrelated
application ports. F1 alone remains a shape-local in-house result; G2B adds one
full-scale same-contract comparison against two external exact systems, while
G4B-R1/G4C add a 27-cell public correctness/selectivity matrix and formal
dynamic timing on SIFT-128, CIFAR-GIST-512, and Fashion-MNIST-784.

Required promotion path:

1. Preserve F1 and G2B as accepted frozen mechanism/external-system results;
   do not retune them or replace failed attempts.
2. Preserve G4B's ragged-tail failure and G4B-R1/G4C's accepted breadth and
   timing evidence; do not retrofit the failed G3B source.
3. Preserve H0, H1-R1, and H2A as accepted structural/GPU screens; do not
   retrofit their frozen semantics or replace their recorded keepers.
4. H2B has failed against the runtime-resolved pedantic SGEMM/cuBLAS exact
   baseline. Lock the current work as a single-vector operator paper; do not
   run the blocked streaming phase or invent a weak tree contribution.
5. Lock the one-sentence thesis and contribution IDs before prose, then build
   the teaser, pipeline figure, and main external-results table.

## Lower strategy: modality ports and one more tuned kernel

**Status: do not pursue as the paper thesis.** Audio/video/HSI breadth alone is
incremental, and D2B/D3B already show that a fixed kernel does not transfer as a
universal performance result. More datasets, per-modality guards, or tree
parameter sweeps would add engineering without repairing novelty. Retain them
only as evaluation once the middle mechanism survives.

## Current resource decision

F1 is complete and accepted. The external audit found that GDS-Join and MiSTIC
are buildable exact keepers on SM120; FaSTED is buildable but approximate and
license-constrained. `GATE0_G2.md` and `PROTOCOL_G2.md` now freeze a 4,096-point
canonical-ID admission gate before any full Cifar60K campaign. More modality
datasets, another conventional tree, and multi-GPU tuning are not admitted.

G2A has now passed that admission gate: both exact external keepers and the
scalable candidate reproduce one canonical 262,144-ID FP64 oracle twice, while
the candidate's forced overflow path grows beyond the former 65K boundary.
This raises the middle strategy from buildable to subset-scale externally
validated correctness. It still does not supply a performance or full-scale
claim; G2B is the decisive remaining gate.

The follow-on G2A2 gate also passed: a proof-bounded upper-triangle schedule
reproduced the same oracle twice while evaluating each unordered pair once and
without an output-count estimate or retry. This admits one full Cifar60K
resource/correctness smoke, but still supplies no performance evidence.

The full G2B exactness/resource smoke has now passed as well: TensorJoin,
GDS-Join FP64, and MiSTIC FP64 produced byte-identical 3,926,078-ID outputs on
the frozen 60K source. The historical 3,926,074 count is retained as unresolved
provenance counterevidence. That smoke was followed by accepted full memcheck, 1,000-iteration sustained
stability, and an eight-round same-public-denominator campaign. TensorJoin won
8/8 rounds against both exact keepers: 5.083x over faster-keeper MiSTIC with
95% paired round-bootstrap interval [5.030x, 5.133x], and 11.783x over GDS-Join
[11.716x, 11.846x]. FaSTED is faster but misses 598 exact pairs and adds 1,130
false pairs. `DECISION_G2B_FORMAL.md` therefore admits a scoped external exact
performance claim. The remaining promotion gates are cross-cell breadth and
mechanism attribution, not another single modality port.

G3A/G3B-R1 now remove one important paper weakness without changing the
accepted G2B keeper. The host analytic interval first passed all 8,390,656 G2A
upper pairs. A parent-level acceptance audit then caught the missing adversarial
gate: the first all-zero case exposed a host preprocessing domain defect rather
than a GPU decision mismatch. Separate R1 repairs only normal-or-zero residual
bound storage. It passes all seven same-shape adversarial/metamorphic cases, two
independent G2A runs, a generated-code audit, full memcheck, and 1,000-launch
stability. Its normalized GPU instruction stream is identical to G3B. This is a
generated-code-aware exact certificate under one frozen SM120/shape/domain and
seven falsification cases, not a theorem over all float32 inputs. It does
**not** retroactively prove the timed G2B implementation or the FP32 middle
stage.

G3C-B-R1 now proves the missing middle stage under the same scoped discipline.
The host opportunity gate first left 40/102,079 G3B-R1 ambiguous pairs for
FP64. The deliberately padded GPU implementation leaves 97 (0.0950%), with
exact output in two isolated processes. Its seven-case adversarial gate exposed
and then repaired a non-scale-equivariant final relative margin: at `2^-8`
scale the original candidate left 49,452 pairs for FP64, whereas separate R1
leaves 97 while preserving the absolute FTZ floor. R1 passes generated-code
inspection, full three-stage memcheck, and 1,000 complete cascades. This is
correctness, generated-code, and safety evidence—not a speedup—and it does not
retroactively prove the separately timed G2B implementation.

G3D then passes the missing same-contract performance-attribution gate. On the
frozen `4096x512` resident-GPU contract, the exact G3C-B-R1 pipeline is 2.670x
faster than the unchanged G3B-R1 two-stage path, with a 95% process-bootstrap
interval [2.658x, 2.678x] and 8/8 wins. Eight separate 1,000-invocation batches
give 2.868x and 8/8 wins. Every process retains exact pre/post output and the
same audited G3B, FP32-filter, and FP64-refinement cubins. This supports the
router mechanism, but its validated fixed launch extents deliberately exclude
dynamic count discovery, ingest, D2H output, and sorting; it is not a public
end-to-end result and does not retroactively prove the G2B implementation.

G4A-R1 closes the fixed-count part of that boundary.  The original GPU0 screen
passed, but three formal attempts were preserved and excluded after unrelated
dense/sparse and SGLang workloads acquired GPU0.  A separately frozen revision
moved only the physical device, UUID, lock, and evidence prefix to an idle
same-model GPU1, then reran screen and formal from scratch.  Across eight fresh
processes, the host-dispatched candidate remains 2.195x faster after charging
actual scalar D2H count reads, their synchronizations, Python/Triton launches,
and observed launch extents; the 95% process-bootstrap interval is
[2.159x, 2.213x], with 8/8 median and 8/8 sustained wins.  The 1,000-invocation
dynamic sequences give 2.227x.  Exact output, observed count tuples, runtime
cubin identity, and isolation all pass.  This establishes a deployable
shape-local dynamic router, but still excludes ingest and final pair-ID
D2H/sort and does not transfer proof to the full-scale G2B kernel.

G4B then exposed and preserved a real extensibility failure: the inherited G3B
K mask was sound only when D was divisible by 64, producing 8 unsafe accepts
and 4 unsafe rejects at Fashion D=784. A same-process one-variable A/B localized
the final partial-K load, and separate G4B-R1 reran all 27 public cells with a
ragged-safe source. All cells are exact. At N=4096, SIFT and Fashion pass every
residual-FP64 and low/medium G3B-selectivity threshold, so the result is a
coherent dimension-density-scale mechanism test rather than more modality
packaging.

G4C converts that geometry into formal cross-dataset dynamic timing. On the
predeclared N=4096,target-64 anchors, the candidate is 1.271x on SIFT-128,
2.329x on CIFAR-GIST-512, and 1.710x on Fashion-784, with respective 95%
process-bootstrap intervals [1.267x,1.274x], [2.299x,2.358x], and
[1.687x,1.736x]. Every dataset has 8/8 median and 8/8 sustained wins; all 24
fresh processes reproduce exact outputs, dynamic counts, and screen-audited
per-dimension cubins. This closes the breadth and dynamic-orchestration gates,
but the scope remains resident-GPU and only the maximum frozen scale/density
cell is timed per dataset.

Accordingly, the single-vector middle strategy is evidence-complete as a strong
exact GPU operator direction. A conventional certificate-aware block/tree
index is no longer the highest-value next step: E0 is structurally negative,
while GTS and Tribase make the novelty space crowded. H1-R1 established an
exact GPU multi-vector prototype against exhaustive FP64, and H2A retains a
2.758x--10.455x advantage against its custom certified-FP32 comparator, but
H2B now supplies the decisive negative answer against the strongest admitted
keeper. Despite passing exactness, selected-SASS, and bounded safety/stress
gates, the upper candidate fails complete outer-wall timing. The next action is
paper-thesis/contribution locking around G2B plus G4, followed by paper-facing
external-results and mechanism tables—not H2B scaling, a tree experiment,
another modality port, or an isolated micro-kernel tune.

G5 subsequently tested the highest-value remaining strengthening route rather
than merely assuming the split could be closed.  A single full-scale
G4C-derived artifact passes exact output, selected-cubin/PTX semantic audit,
full-leak memcheck, and 1,000-iteration two-buffer stress.  Its two clean
full-public rounds against freshly measured MiSTIC/GDS-Join yield `4.092x` and
`0.951x` over MiSTIC.  The `0.951x` round violates the frozen every-round
`1.50x` gate, so G5 is rejected before formal timing and SIFT/Fashion expansion.
This negative result does not erase G2B or G4, but it prevents unifying their
claims.  The paper path remains G2B external impact plus separately scoped G4
mechanism/breadth evidence, with the implementation boundary stated directly.
