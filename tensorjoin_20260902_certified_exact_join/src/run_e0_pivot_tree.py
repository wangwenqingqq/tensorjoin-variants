#!/usr/bin/env python3
"""Run the frozen optimistic GTS-style pivot-tree tensorization ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_pivot_tree_ceiling_e0"
QUERY_COUNT = 512
BASE_COUNT = 4096
TARGETS = (1, 64)
FANOUT = 8
LEAF_SIZE = 64
QUERY_TILE = 16
VECTOR_TILE = 64
BOUND_TOLERANCE = 1e-12


@dataclass
class Child:
    node: "Node"
    min_to_parent_pivot: float
    max_to_parent_pivot: float


@dataclass
class Node:
    members: np.ndarray
    depth: int
    pivot: int | None = None
    children: list[Child] = field(default_factory=list)
    leaf_id: int | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def select_audio(features: np.ndarray, clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260902)
    units = rng.permutation(np.unique(clips))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(clips == units[cursor]).tolist())
        cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(clips == units[cursor]).tolist())
        cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
    )


def select_video(
    features: np.ndarray, clips: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260903)
    units = rng.permutation(np.unique(groups))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(groups == units[cursor]).tolist())
        cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(groups == units[cursor]).tolist())
        cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    if len(np.intersect1d(groups[query_ids], groups[base_ids])):
        raise AssertionError("Video source-group leakage")
    if len(np.intersect1d(clips[query_ids], clips[base_ids])):
        raise AssertionError("Video clip leakage")
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
    )


def exact_squared_distances(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    output = np.empty((len(q), len(x)), dtype=np.float64)
    block = max(1, 8192 // query.shape[1])
    for start in range(0, len(q), block):
        stop = min(start + block, len(q))
        delta = q[start:stop, None, :] - x[None, :, :]
        output[start:stop] = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    return output


def midgap(squared_distance: np.ndarray, target: int) -> tuple[float, dict[str, float]]:
    squared = squared_distance.reshape(-1)
    rank = QUERY_COUNT * target - 1
    lower = float(np.partition(squared, rank)[rank])
    greater = squared[squared > lower]
    upper = float(np.min(greater))
    threshold_d2 = lower + (upper - lower) / 2.0
    return math.sqrt(threshold_d2), {
        "lower_distance_d2": lower,
        "upper_distance_d2": upper,
        "gap_d2": upper - lower,
        "threshold_d2": threshold_d2,
    }


def build_tree(base64: np.ndarray) -> tuple[Node, list[Node], list[Node]]:
    internal: list[Node] = []
    leaves: list[Node] = []

    def build(members: np.ndarray, depth: int) -> Node:
        node = Node(members=np.asarray(members, dtype=np.int32), depth=depth)
        if len(members) <= LEAF_SIZE:
            node.leaf_id = len(leaves)
            leaves.append(node)
            return node
        node.pivot = int(members[len(members) // 2])
        distance = np.linalg.norm(base64[members] - base64[node.pivot], axis=1)
        order = np.lexsort((members, distance))
        ordered_members = members[order]
        ordered_distance = distance[order]
        chunks = np.array_split(np.arange(len(members)), FANOUT)
        for chunk in chunks:
            child_members = ordered_members[chunk]
            child = build(child_members, depth + 1)
            node.children.append(
                Child(
                    node=child,
                    min_to_parent_pivot=float(ordered_distance[chunk[0]]),
                    max_to_parent_pivot=float(ordered_distance[chunk[-1]]),
                )
            )
        internal.append(node)
        return node

    root = build(np.arange(len(base64), dtype=np.int32), 0)
    internal.sort(key=lambda node: (node.depth, int(node.pivot or -1)))
    leaves.sort(key=lambda node: int(node.leaf_id or 0))
    return root, internal, leaves


def evaluate(
    modality: str,
    query: np.ndarray,
    base: np.ndarray,
) -> dict[str, object]:
    query64 = query.astype(np.float64)
    base64 = base.astype(np.float64)
    squared_distance = exact_squared_distances(query, base)
    distance = np.sqrt(squared_distance)
    root, internal, leaves = build_tree(base64)
    if sorted(len(leaf.members) for leaf in leaves) != [LEAF_SIZE] * (BASE_COUNT // LEAF_SIZE):
        raise AssertionError("Expected fixed 64-vector leaves")
    internal_pivots = np.asarray([node.pivot for node in internal], dtype=np.int32)
    pivot_distance = np.sqrt(exact_squared_distances(query, base[internal_pivots]))
    pivot_column = {int(pivot): column for column, pivot in enumerate(internal_pivots)}
    cells: dict[str, object] = {}

    for target in TARGETS:
        radius, threshold_info = midgap(squared_distance, target)
        threshold_d2 = threshold_info["threshold_d2"]
        oracle = squared_distance <= threshold_d2
        final = np.zeros_like(oracle, dtype=bool)
        active_queries_by_leaf: list[list[int]] = [[] for _ in leaves]
        rejected_pairs = 0
        bulk_accepted_pairs = 0
        bound_violations = 0
        visited_child_intervals = 0

        for query_id in range(QUERY_COUNT):
            stack = [root]
            while stack:
                node = stack.pop()
                if node.is_leaf or node.pivot is None:
                    raise AssertionError("Only internal nodes may enter traversal stack")
                dqp = float(pivot_distance[query_id, pivot_column[node.pivot]])
                for child_record in node.children:
                    child = child_record.node
                    a = child_record.min_to_parent_pivot
                    b = child_record.max_to_parent_pivot
                    lower = a - dqp if dqp < a else dqp - b if dqp > b else 0.0
                    upper = dqp + b
                    actual = distance[query_id, child.members]
                    if lower > float(actual.min()) + BOUND_TOLERANCE:
                        bound_violations += 1
                    if upper + BOUND_TOLERANCE < float(actual.max()):
                        bound_violations += 1
                    visited_child_intervals += 1
                    if lower > radius:
                        rejected_pairs += len(child.members)
                    elif upper <= radius:
                        final[query_id, child.members] = True
                        bulk_accepted_pairs += len(child.members)
                    elif child.is_leaf:
                        active_queries_by_leaf[int(child.leaf_id)].append(query_id)
                    else:
                        stack.append(child)

        useful_leaf_pairs = 0
        padded_leaf_cells = 0
        used_leaves = 0
        active_queries_per_used_leaf: list[int] = []
        for leaf, active_queries in zip(leaves, active_queries_by_leaf):
            if not active_queries:
                continue
            used_leaves += 1
            active = np.asarray(active_queries, dtype=np.int32)
            final[np.ix_(active, leaf.members)] = oracle[np.ix_(active, leaf.members)]
            useful_leaf_pairs += len(active) * len(leaf.members)
            padded_leaf_cells += (
                math.ceil(len(active) / QUERY_TILE)
                * QUERY_TILE
                * math.ceil(len(leaf.members) / VECTOR_TILE)
                * VECTOR_TILE
            )
            active_queries_per_used_leaf.append(len(active))

        mismatch = int(np.count_nonzero(final != oracle))
        total_pairs = QUERY_COUNT * BASE_COUNT
        accounted = rejected_pairs + bulk_accepted_pairs + useful_leaf_pairs
        if accounted != total_pairs:
            raise AssertionError((modality, target, accounted, total_pairs))
        pivot_cells = QUERY_COUNT * math.ceil(len(internal) / VECTOR_TILE) * VECTOR_TILE
        total_padded_cells = pivot_cells + padded_leaf_cells
        leaf_utilization = useful_leaf_pairs / max(padded_leaf_cells, 1)
        padded_fraction = total_padded_cells / total_pairs
        cell_pass = bool(
            mismatch == 0
            and bound_violations == 0
            and leaf_utilization >= 0.70
            and padded_fraction <= (0.20 if target == 1 else 0.35)
        )
        row = {
            "target_results_per_query": target,
            "actual_results_per_query": int(np.count_nonzero(oracle)) / QUERY_COUNT,
            "radius": radius,
            **threshold_info,
            "oracle_pairs": int(np.count_nonzero(oracle)),
            "output_mismatch": mismatch,
            "bound_violations": bound_violations,
            "visited_child_intervals": visited_child_intervals,
            "rejected_pairs": rejected_pairs,
            "bulk_accepted_pairs": bulk_accepted_pairs,
            "boundary_leaf_pairs": useful_leaf_pairs,
            "used_leaves": used_leaves,
            "median_active_queries_per_used_leaf": float(np.median(active_queries_per_used_leaf)),
            "pivot_cells": pivot_cells,
            "padded_leaf_cells": padded_leaf_cells,
            "leaf_panel_utilization": leaf_utilization,
            "total_padded_cells": total_padded_cells,
            "total_padded_cell_fraction": padded_fraction,
            "full_scan_pairs": total_pairs,
            "gate_pass": cell_pass,
        }
        print("CELL " + json.dumps({"modality": modality, **row}, sort_keys=True), flush=True)
        cells[str(target)] = row

    return {
        "dimension": int(query.shape[1]),
        "tree": {
            "fanout": FANOUT,
            "leaf_size": LEAF_SIZE,
            "internal_nodes": len(internal),
            "leaves": len(leaves),
            "max_depth": max(leaf.depth for leaf in leaves),
        },
        "targets": cells,
        "gate_pass": all(bool(row["gate_pass"]) for row in cells.values()),
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("E0 must be CPU-only")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite E0 evidence")
    start_time = time.time()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "audio_cache_sha256": sha256_file(args.audio_cache),
                "video_cache_sha256": sha256_file(args.video_cache),
                "script_sha256": sha256_file(Path(__file__)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with np.load(args.audio_cache, allow_pickle=False) as data:
        audio_query, audio_base = select_audio(data["features"], data["clip"])
    with np.load(args.video_cache, allow_pickle=False) as data:
        video_query, video_base = select_video(data["features"], data["clip"], data["group"])
    modalities = {
        "audio_panns_2048d": evaluate("audio_panns_2048d", audio_query, audio_base),
        "video_r3d18_512d": evaluate("video_r3d18_512d", video_query, video_base),
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "audio_cache_sha256": sha256_file(args.audio_cache),
        "video_cache_sha256": sha256_file(args.video_cache),
        "script_sha256": sha256_file(Path(__file__)),
        "modalities": modalities,
        "gate_pass": all(bool(row["gate_pass"]) for row in modalities.values()),
        "elapsed_s": time.time() - start_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
