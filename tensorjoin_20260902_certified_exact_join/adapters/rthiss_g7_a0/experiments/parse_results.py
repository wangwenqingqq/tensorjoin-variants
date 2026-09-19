import argparse
import os
import re
import sys

import pandas as pd

FIELDS = {
    "Dataset file": "dataset_file",
    "Dataset points": "points",
    "Dataset dimensions": "dims",
    "Epsilon": "epsilon",
    "Reorder mode": "reorder",
    "USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING": "prim_shared_query_shared",
    "USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING": "prim_shared_query_global",
    "USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING": "prim_global_query_shared",
    "USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING": "prim_global_query_global",
    "USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING": "non_batching",
    "MAX_KD_LEVELS": "kd_level",
    "THREADS_TO_COPY": "threads",
    "USE_UNCOMPRESSED_MASK": "uncompressed",
    "USE_PAGEABLE_MEMORY": "pageable",
    "USE_FIXED_WORK_PER_BLOCK": "fixed_work",
    "SORT_BY_WORKLOAD": "sort_by_workload",
    "USE_CANDIDATE_POINT_COPY": "candidate_copy",
    "PINNED_MEMORY_OPT": "pinned_bytes",
    "Total number of queries": "queries",
    "Time to perform only intersection test": "intersection_time",
    "Total identified neighbors": "neighbors",
    "Time for KD tree construction": "kdtree_time",
    "CUDA kernel only time": "cuda_time",
    "Time to copy results only": "copy_time",
    "Time to compress results": "compress_time",
    "Work Generation time": "workgen_time",
    "Refine time": "refine_time",
    "RT time": "rt_time",
    "TOTAL TIME (REFINE + QUERY)": "total_query_time",
    "TOTAL TIME INCLUDING EVERYTHING": "total_time",
}

NUMERIC = {
    "points", "dims", "epsilon", "kd_level", "threads", "pinned_bytes",
    "queries", "neighbors", "intersection_time", "kdtree_time", "cuda_time",
    "copy_time", "compress_time", "workgen_time", "refine_time", "rt_time",
    "total_query_time", "total_time",
    "prim_shared_query_shared", "prim_shared_query_global",
    "prim_global_query_shared", "prim_global_query_global",
    "non_batching", "uncompressed", "pageable",
    "fixed_work", "sort_by_workload", "candidate_copy",
}

VARIANT_COLUMNS = [
    "prim_shared_query_shared",
    "prim_shared_query_global",
    "prim_global_query_shared",
    "prim_global_query_global",
    "non_batching",
]

DATASET_LABELS = {
    "wave": "WEC",
    "msd": "MSD",
    "susy": "SuSy",
    "bigcross": "BigCross",
    "higgs": "Higgs",
}

DATASET_ORDER = ["WEC", "MSD", "SuSy", "BigCross", "Higgs"]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _to_number(text):
    match = NUM_RE.search(text)
    return float(match.group(0)) if match else float("nan")


def parse_file(path):
    with open(path, "r", errors="replace") as handle:
        content = handle.read()

    records = []
    for block in content.split("START OF RESULT"):
        record = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key not in FIELDS:
                continue
            record[FIELDS[key]] = value.strip()
        if "total_time" not in record or "dataset_file" not in record:
            continue
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    for column in NUMERIC:
        if column in df.columns:
            df[column] = df[column].map(_to_number)

    df["name"] = df["dataset_file"].map(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    df["Dataset"] = df["name"].map(lambda n: DATASET_LABELS.get(n, n))

    present = [c for c in VARIANT_COLUMNS if c in df.columns]
    if present:
        df["variant"] = df[present].idxmax(axis=1)
        df.loc[df[present].max(axis=1) != 1, "variant"] = "unknown"

    if "pinned_bytes" in df.columns:
        df["pinned_mib"] = (df["pinned_bytes"] / (1024 * 1024)).round().astype("Int64")

    other = df["total_time"]
    for column in ("intersection_time", "rt_time", "cuda_time",
                   "compress_time", "copy_time", "kdtree_time"):
        if column in df.columns:
            other = other - df[column].fillna(0)
    df["other_time"] = other.clip(lower=0)

    df["epsilon"] = df["epsilon"].round(6)
    return df


def order_datasets(df):
    known = [d for d in DATASET_ORDER if d in set(df["Dataset"])]
    extra = sorted(set(df["Dataset"]) - set(DATASET_ORDER))
    return known + extra


def load(path):
    df = parse_file(path)
    if df.empty:
        print("ERROR: no parsable results in {}".format(path), file=sys.stderr)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = load(args.results)
    df.to_csv(args.out, index=False)
    print("wrote {} rows to {}".format(len(df), args.out))


if __name__ == "__main__":
    main()
