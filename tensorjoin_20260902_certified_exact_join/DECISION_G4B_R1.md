# Decision G4B-R1: public breadth opportunity admitted

Date: 2026-09-03

## Decision

**Admit the ragged-safe exact router for cross-dataset timing.** G4B-R1 passes
all 27 public dataset/scale/density cells with exact output and passes both
predeclared cross-dataset selectivity gates. This establishes breadth of the
mechanism and exposes its implementation boundary; it is not a performance or
novelty result by itself.

The rejected G4B source and its Fashion failure remain immutable. G4B-R1 uses a
separate kernel source and reruns every multiple-of-64 control and ragged cell
from scratch.

## Exactness and routing result

- Datasets/dimensions: SIFT-128, CIFAR-GIST-512, Fashion-MNIST-784.
- Nested public-source scales: `N=1024,2048,4096`.
- Strict FP64 mid-gap target degrees: `k=1,16,64`.
- Exact cells: 27/27; zero unsafe G3B/G3C decisions, final mismatch, duplicate,
  invalid/lower-triangle ID, or overflow.
- Each source file, sampled row prefix, threshold metadata, oracle file/raw IDs,
  and final output has a retained SHA-256.

At `N=4096`, the G3B ambiguity and residual-FP64 fractions are:

| Dataset | G3B ambiguity / all upper pairs, k=1..64 | FP64 / G3B ambiguity, k=1..64 |
|---|---:|---:|
| SIFT-128 | 0.0083%--0.4745% | 0.1631%--0.2882% |
| CIFAR-GIST-512 | 0.0257%--1.3050% | 0.0926%--0.1306% |
| Fashion-MNIST-784 | 0.0093%--0.3259% | 0.1280%--0.2304% |

SIFT and Fashion therefore pass all three residual thresholds, exceeding the
required two of three, and both pass both low/medium-density G3B selectivity
cells, exceeding the required one.

## Ragged-dimension boundary

G4B's failed `D=784` run is an important negative result. The original G3B
kernel's static K mask was valid only for dimensions divisible by 64. The
in-process A/B reproduced 8 unsafe accepts and 4 unsafe rejects, while the
one-variable ragged-safe mask produced zero of either. G4B-R1 then completed
all nine Fashion cells exactly. This is a correctness repair and deployment
boundary, not an algorithmic contribution by itself.

## Evidence and allowed wording

- Opportunity summary:
  `results/g4b_r1_opportunity_summary.json`, SHA-256
  `4cc069e20b52412d267e5bb945f11362df7dbcf443fbbd5c7a96d4ddc097d1de`.
- R1 kernel SHA-256:
  `439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6`.
- Protocol SHA-256:
  `9f7a44ee7bee8a0a656ddfa3cfeb88cef3802d432829eba6018e03b58ea0fe1a`.
- Clean isolation: three of three dataset processes admitted on physical GPU1;
  no foreign or postflight process.

Allowed: the audited exact cascade remains selective across the tested three
public datasets, three native dimensions, three scales, and three output
densities.

Not allowed: arbitrary-dataset generality, performance, full-scale performance,
ingest-inclusive performance, or a new low-precision-join novelty claim.
