#!/usr/bin/env python3
"""Exact synthetic counterexample to mandatory full candidate materialization."""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "PROTOCOL_G12_FACTORIZED_BASELINE.md"
FROZEN_SHA = "96206c0db252774e0c02de20a19389e118ae192af00f543374b686d2d6c766ef"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def pair_hash(pairs):
    return hashlib.sha256(np.asarray(sorted(pairs), dtype="<u8").tobytes()).hexdigest()


def key(q, b):
    # IDs are typed by relation: high 32 bits query ID, low 32 bits base ID.
    return (int(q) << 32) | int(b)


def fixtures():
    rng = np.random.Generator(np.random.PCG64(20260904))
    all_cases = []
    for name, nq, nb in [("mixed_overlap", 23, 29), ("full_overlap_12", 64, 80),
                         ("ragged_disjoint", 33, 41), ("empty_relation", 23, 29)]:
        q = rng.integers(-8, 9, size=(nq, 8), dtype=np.int64)
        b = rng.integers(-8, 9, size=(nb, 8), dtype=np.int64)
        b[0] = q[0]
        if name == "mixed_overlap":
            factors = []
            for i in range(9):
                qi = sorted(rng.choice(nq, size=int(rng.integers(1, nq + 1)), replace=False).tolist())
                bi = sorted(rng.choice(nb, size=int(rng.integers(1, nb + 1)), replace=False).tolist())
                if i == 0:
                    qi, bi = sorted(set(qi) | {0}), sorted(set(bi) | {0})
                factors.append((qi, bi))
            factors += [factors[0], ([], [0]), ([0], [])]
        elif name == "full_overlap_12":
            factors = [(list(range(nq)), list(range(nb))) for _ in range(12)]
        elif name == "ragged_disjoint":
            factors = [(list(range(i, min(i + 7, nq))), list(range(i % 3, nb, 2 + i % 3)))
                       for i in range(0, nq, 7)] + [([], [0, 1])]
        else:
            factors = [(list(range(nq)), []), ([], list(range(nb)))]
        all_cases.append((name, q, b, factors))
    return all_cases


def oracle(q, b, factors, threshold):
    # The oracle deliberately expands eligibility; it is NOT the strong plan.
    multiplicity = collections.Counter((i, j) for qi, bi in factors for i in qi for j in bi)
    answer = set()
    accepted_occurrences = 0
    for (i, j), m in multiplicity.items():
        dist = sum((int(x) - int(y)) ** 2 for x, y in zip(q[i], b[j]))
        if dist <= threshold:
            answer.add(key(i, j))
            accepted_occurrences += m
    return answer, accepted_occurrences, len(multiplicity)


def streamed_plan(q, b, factors, threshold, tile):
    """Retain factors, one score tile, and passing output keys only."""
    answer = set()
    evaluations = accepts = high_water = 0
    assert q.shape[1] == b.shape[1] == 8
    assert np.max(np.abs(q)) <= 8 and np.max(np.abs(b)) <= 8
    # Bound on all squared distances is 8*(2*8)^2 = 2048, well within int64.
    for qi, bi in factors:
        for qs in range(0, len(qi), tile):
            rows = qi[qs:qs + tile]
            qtile = q[rows]
            qnorm = np.sum(qtile * qtile, axis=1)
            for bs in range(0, len(bi), tile):
                cols = bi[bs:bs + tile]
                btile = b[cols]
                bnorm = np.sum(btile * btile, axis=1)
                score = qnorm[:, None] + bnorm[None, :] - 2 * (qtile @ btile.T)
                evaluations += int(score.size)
                high_water = max(high_water, int(score.size))
                ir, ic = np.nonzero(score <= threshold)
                accepts += len(ir)
                for i, j in zip(ir, ic):
                    answer.add(key(rows[int(i)], cols[int(j)]))
                # No queue/cache/list of rejected or all eligible endpoint pairs.
    return answer, {"score_evaluations_W": evaluations, "output_hash_attempts_Wplus": accepts,
                    "score_tile_slots_high_water": high_water,
                    "retained_output_keys_Z": len(answer),
                    "global_eligible_pair_buffer_present": False}


def negative_controls():
    # Same factor sum and individual norms, different orientation and answers.
    qa = np.array([[1, 0], [-1, 0]], dtype=np.int64)
    qb = np.array([[0, 1], [0, -1]], dtype=np.int64)
    b = np.array([[1, 0], [-1, 0]], dtype=np.int64)
    assert np.array_equal(qa.sum(axis=0), qb.sum(axis=0))
    assert np.array_equal((qa * qa).sum(axis=1), (qb * qb).sum(axis=1))
    aa = [(i, j) for i in range(2) for j in range(2) if sum((qa[i] - b[j]) ** 2) <= 0]
    ab = [(i, j) for i in range(2) for j in range(2) if sum((qb[i] - b[j]) ** 2) <= 0]
    assert len(aa) == 2 and len(ab) == 0
    bag = [(0, 0), (0, 0)]
    assert len(bag) == 2 and len(set(bag)) == 1
    return {"bag_multiplicity_not_preserved_by_distinct": {"bag_rows": 2, "distinct_rows": 1},
            "factor_summary_collision": {"same_sum": True, "same_per_vector_norms": True,
                                         "query_vectors_A": qa.tolist(), "query_vectors_B": qb.tolist(),
                                         "base_vectors": b.tolist(), "threshold": 0,
                                         "matching_pairs_A": aa, "matching_pairs_B": ab}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "results/g12_factorized_baseline_cpu_a0.json")
    args = ap.parse_args()
    assert not args.output.exists(), "Refuse to overwrite an existing run"
    assert sha(PROTOCOL) == FROZEN_SHA, "Protocol changed after freeze"
    records, fixture_ledger = [], []
    for name, q, b, factors in fixtures():
        encoded = json.dumps({"q": q.tolist(), "b": b.tolist(), "factors": factors}, sort_keys=True)
        w = sum(len(qi) * len(bi) for qi, bi in factors)
        fixture_ledger.append({"fixture": name, "shape": [len(q), len(b), 8],
                               "factors": len(factors), "W": w,
                               "fixture_sha256": hashlib.sha256(encoded.encode()).hexdigest()})
        for threshold in (-1, 0, 64, 4096):
            expected, wplus, e = oracle(q, b, factors, threshold)
            for tile in (1, 7, 16):
                for order in ("forward", "reverse"):
                    ordered = factors if order == "forward" else list(reversed(factors))
                    actual, counters = streamed_plan(q, b, ordered, threshold, tile)
                    assert actual == expected, (name, threshold, tile, order)
                    assert counters["score_evaluations_W"] == w
                    assert counters["output_hash_attempts_Wplus"] == wplus
                    assert counters["score_tile_slots_high_water"] <= tile * tile
                    assert counters["retained_output_keys_Z"] == len(expected)
                    row = {"fixture": name, "threshold": threshold, "tile": tile, "order": order,
                           "eligible_pairs_E_oracle_only": e, "output_sha256": pair_hash(actual),
                           "exact_ids_match": True, **counters}
                    records.append(row)
                    print(json.dumps(row), flush=True)
    assert len(records) == 96
    result = {
        "experiment": "tensorjoin_20260904_g12_streamed_factorized_semantics_cpu_a0",
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "source_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
        "scope": "Synthetic exact integer semantics and logical live-set counters only",
        "gpu_timing_correctness_sanitizer_sass": "N/A; no GPU or arbitrary-FP32 test",
        "fixtures": fixture_ledger, "runs": records, "negative_controls": negative_controls(),
        "summary": {"runs": len(records), "all_exact_matches": True,
                    "mandatory_pre_predicate_E_materialization": "disproved for the frozen contract",
                    "GPU_speedup": "unknown", "novelty_pass": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
