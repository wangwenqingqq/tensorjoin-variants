#!/usr/bin/env python3
"""Audit real graph eligibility; this is not a GPU candidate or timing benchmark."""
import argparse
import collections
import datetime
import gc
import gzip
import hashlib
import io
import json
import pathlib
import platform
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "PROTOCOL_G11_RELATIONAL_CENSUS.md"
PROTOCOL_SHA256 = "97e18ef1af988fc93755c8d084733e6cf982fd221b97580d473c91cf337e4ad2"
N, NQ, NB = 169343, 4096, 16384
SEEDS = (20260904, 20260905)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def adjacency(edges, key_column):
    order = np.argsort(edges[:, key_column], kind="stable")
    sorted_edges = edges[order]
    keys, starts = np.unique(sorted_edges[:, key_column], return_index=True)
    ends = np.r_[starts[1:], len(edges)]
    return {int(k): sorted_edges[a:b, 1 - key_column]
            for k, a, b in zip(keys, starts, ends)}


def census(edges, out_adj, in_adj, seed, pattern):
    started = time.perf_counter()
    perm = np.random.Generator(np.random.PCG64(seed)).permutation(N)
    query, base = perm[:NQ].astype("<i4"), perm[NQ:NQ + NB].astype("<i4")
    qmap, bmap = np.full(N, -1, np.int32), np.full(N, -1, np.int32)
    qmap[query], bmap[base] = np.arange(NQ), np.arange(NB)
    endpoint_column = 0 if pattern == "common_reference" else 1
    endpoint, witness = edges[:, endpoint_column], edges[:, 1 - endpoint_column]
    qgroups, bgroups = collections.defaultdict(list), collections.defaultdict(list)
    for w, i in zip(witness[qmap[endpoint] >= 0], qmap[endpoint][qmap[endpoint] >= 0]):
        qgroups[int(w)].append(int(i))
    for w, i in zip(witness[bmap[endpoint] >= 0], bmap[endpoint][bmap[endpoint] >= 0]):
        bgroups[int(w)].append(int(i))
    live = sorted(qgroups.keys() & bgroups.keys())
    # Every pair has at most N distinct witnesses, so uint32 cannot overflow.
    assert N < np.iinfo(np.uint32).max
    counts = np.zeros((NQ, NB), dtype=np.uint32)
    fat = np.zeros((NQ, NB), dtype=bool)
    expanded = factor_ids = fat_expanded = 0
    padded = {16: 0, 64: 0}
    sizes = []
    for w in live:
        qi, bi = qgroups[w], bgroups[w]
        a, b = len(qi), len(bi)
        assert len(set(qi)) == a and len(set(bi)) == b
        expanded += a * b
        factor_ids += a + b
        sizes.append((a, b))
        counts[np.ix_(qi, bi)] += 1
        if a >= 16 and b >= 16:
            fat[np.ix_(qi, bi)] = True
            fat_expanded += a * b
        for tile in padded:
            padded[tile] += ((a + tile - 1) // tile) * ((b + tile - 1) // tile) * tile * tile
    assert int(counts.sum(dtype=np.uint64)) == expanded
    eligible = counts > 0
    unique = int(np.count_nonzero(eligible))
    assert unique > 0
    fat_unique = int(np.count_nonzero(fat))
    assert not np.any(fat & ~eligible)
    # Independent row-wise adjacency unions do not use the witness rectangles.
    first, second = (out_adj, in_adj) if endpoint_column == 0 else (in_adj, out_adj)
    empty = np.empty(0, dtype=np.int32)
    row_groups = collections.Counter()
    row_counts = []
    for row, q in enumerate(query):
        candidates = []
        for w in first.get(int(q), empty):
            ids = bmap[second.get(int(w), empty)]
            valid = ids[ids >= 0]
            if len(valid):
                candidates.append(valid)
        expected = np.unique(np.concatenate(candidates)) if candidates else empty
        actual = np.flatnonzero(eligible[row]).astype("<i4")
        assert np.array_equal(expected, actual), (pattern, seed, row)
        row_groups[actual.tobytes()] += 1
        row_counts.append(len(actual))
    csr_bytes = 4 * unique + 8 * (NQ + 1)
    factor_bytes = 4 * factor_ids + 24 * len(live)
    checks = {
        "duplicate_ratio_ge_1_5": expanded / unique >= 1.5,
        "factor_bytes_le_quarter_csr": factor_bytes <= 0.25 * csr_bytes,
        "fat_unique_coverage_ge_half": fat_unique >= 0.5 * unique,
    }
    blocks = {}
    grouped = {}
    for tile in (16, 64):
        active = int(np.count_nonzero(eligible.reshape(NQ // tile, tile, NB // tile, tile).any(axis=(1, 3))))
        physical = active * tile * tile
        blocks[str(tile)] = {"active_blocks": active, "physical_pair_slots": physical,
                             "slots_per_unique_pair": physical / unique}
        # Classic grouping by identical fully expanded neighborhoods.
        grouped[str(tile)] = sum(((m + tile - 1) // tile) * (((len(k) // 4) + tile - 1) // tile)
                                 * tile * tile for k, m in row_groups.items())
    positive_counts = counts[eligible]
    multiplicity = np.bincount(positive_counts.astype(np.int64))
    result = {
        "seed": seed, "predicate": pattern, "shape": [NQ, NB],
        "query_ids_sha256": array_digest(query), "base_ids_sha256": array_digest(base),
        "eligible_bitmap_littlebit_sha256": array_digest(np.packbits(eligible, bitorder="little")),
        "active_witnesses": len(live), "witness_expanded_occurrences_W": expanded,
        "unique_eligible_pairs_E": unique, "eligibility_fraction": unique / (NQ * NB),
        "duplicate_ratio_W_over_E": expanded / unique,
        "multiplicity_histogram": {str(i): int(v) for i, v in enumerate(multiplicity) if v},
        "compact_factor_endpoint_ids": factor_ids, "compact_factor_payload_bytes": factor_bytes,
        "unique_csr_payload_bytes": csr_bytes, "flat_u64_pair_payload_bytes": 8 * unique,
        "factor_over_csr_bytes": factor_bytes / csr_bytes,
        "fat_factor_expanded_occurrences": fat_expanded, "fat_factor_unique_pairs": fat_unique,
        "fat_factor_unique_coverage": fat_unique / unique,
        "literal_witness_panel_slots": {str(k): v for k, v in padded.items()},
        "literal_witness_slots_per_unique_pair": {str(k): v / unique for k, v in padded.items()},
        "fixed_layout_duplicate_free_block_control": blocks,
        "equal_neighborhood_row_groups": len(row_groups),
        "equal_neighborhood_grouped_panel_slots": {str(k): v for k, v in grouped.items()},
        "empty_queries": row_counts.count(0),
        "eligible_per_query_quantiles": np.quantile(row_counts, [0, .5, .9, .99, 1]).tolist(),
        "opportunity_checks": checks, "cell_opportunity_pass": all(checks.values()),
        "correctness": {"all_rows_independent_union_checked": NQ, "sum_counts_equals_W": True,
                        "fat_is_subset": True, "no_uint32_overflow_by_distinct_witness_bound": True},
        "diagnostic_cpu_wall_seconds_not_performance": time.perf_counter() - started,
    }
    print(json.dumps(result), flush=True)
    del counts, fat, eligible, positive_counts
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=pathlib.Path, default=ROOT / "data/g11_ogbn_arxiv/arxiv.zip")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results/g11_relational_census_cpu_a0.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert digest(PROTOCOL) == PROTOCOL_SHA256, "Protocol changed after freeze"
    start = time.perf_counter()
    with zipfile.ZipFile(args.archive) as z:
        members = z.namelist()
        edge_member = next(n for n in members if n.endswith("raw/edge.csv.gz"))
        node_member = next(n for n in members if n.endswith("raw/num-node-list.csv.gz"))
        node_count = int(gzip.decompress(z.read(node_member)).strip())
        raw = gzip.decompress(z.read(edge_member))
        input_edges = np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.int32)
        member_hashes = {n: hashlib.sha256(z.read(n)).hexdigest()
                         for n in (edge_member, node_member)}
    assert node_count == N
    assert input_edges.shape == (1166243, 2), input_edges.shape
    assert int(input_edges.min()) >= 0 and int(input_edges.max()) < N
    edges = np.unique(input_edges, axis=0)
    out_adj, in_adj = adjacency(edges, 0), adjacency(edges, 1)
    metadata = {
        "experiment": "tensorjoin_20260904_g11_relational_eligibility_cpu_a0",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "numpy": np.__version__,
        "source_sha256": digest(__file__), "protocol_sha256": digest(PROTOCOL),
        "archive_url": "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        "archive_sha256": digest(args.archive), "archive_bytes": args.archive.stat().st_size,
        "member_sha256": member_hashes, "uncompressed_edge_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "input_nodes": N, "input_directed_edges": len(input_edges),
        "deduplicated_directed_edges": len(edges), "duplicate_edges_removed": len(input_edges) - len(edges),
        "deduplicated_edges_sha256": array_digest(edges.astype("<i4")),
        "deduplicated_raw_edge_payload_bytes": int(edges.nbytes),
        "gpu_cuda_sass_sanitizer_clocks": "N/A: CPU structural census only",
        "sampling": "PCG64 permutation: first 4096 query, next 16384 base, all witnesses retained",
        "evidence_scope": "Real graph data, constructed query batches, no vector or GPU computation",
    }
    print(json.dumps({"metadata": metadata}), flush=True)
    cells = [census(edges, out_adj, in_adj, seed, predicate)
             for predicate in ("common_reference", "common_citer") for seed in SEEDS]
    admitted = [p for p in ("common_reference", "common_citer")
                if all(c["cell_opportunity_pass"] for c in cells if c["predicate"] == p)]
    summary = {
        "predicates_retained_for_algorithm_design_only": admitted,
        "novelty_status": "unestablished; no novel duplicate-free algorithm implemented",
        "gpu_admitted": False, "vector_correctness_or_speedup": "not measured",
        "all_four_cells_correct": True,
        "cpu_total_wall_seconds_diagnostic_only": time.perf_counter() - start,
    }
    result = {"metadata": metadata, "cells": cells, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
