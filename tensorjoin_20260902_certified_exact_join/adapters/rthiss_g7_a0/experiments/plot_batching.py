import argparse

import parse_results
import plot_style as style

LABELS = {
    "prim_shared_query_shared": "Shared-Query-Primitive (RT-HiSS)",
    "prim_shared_query_global": "Shared-Primitive",
    "prim_global_query_shared": "Shared-Query",
    "prim_global_query_global": "Shared-None",
}

ORDER = list(LABELS.values())


COLORS = ["white", "black", "white", "white"]
HATCHES = ["", "", "//", "xx"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = parse_results.load(args.results)
    if df.empty:
        return

    df = df.dropna(subset=["total_time"])
    df["series"] = df["variant"].map(LABELS)
    style.grouped_bars(df, "series", ORDER, args.out, colors=COLORS, hatches=HATCHES,
                       mono_legend=True)


if __name__ == "__main__":
    main()
