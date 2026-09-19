import argparse

import parse_results
import plot_style as style

COMPRESSED = "Compressed Result Mask (RT-HiSS)"
UNCOMPRESSED = "Uncompressed Result mask"
KEY_VALUE = "Key-Value Copy (Baseline)"

ORDER = [COMPRESSED, UNCOMPRESSED, KEY_VALUE]

COLORS = ["white", "white", "white"]
HATCHES = ["", "//", "xx"]


def classify(row):
    if row["candidate_copy"] == 1:
        return KEY_VALUE
    if row["uncompressed"] == 1:
        return UNCOMPRESSED
    return COMPRESSED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = parse_results.load(args.results)
    if df.empty:
        return

    df = df.dropna(subset=["total_time"])
    df["series"] = df.apply(classify, axis=1)
    style.grouped_bars(df, "series", ORDER, args.out, colors=COLORS, hatches=HATCHES)


if __name__ == "__main__":
    main()
