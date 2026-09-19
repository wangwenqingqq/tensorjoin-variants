import argparse

import parse_results
import plot_style as style

PROPOSED = r"\makecell{$\epsilon$-neighborhood of nearby points \\ as a primitive (RT-HiSS)}"
ONE_POINT = r"\makecell{$\epsilon$-neighborhood of each point \\ as a primitive (Baseline)}"

ORDER = [PROPOSED, ONE_POINT]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = parse_results.load(args.results)
    if df.empty:
        return

    df = df.dropna(subset=["total_time", "kd_level"])
    shallow = df.groupby("Dataset")["kd_level"].transform("min")
    df["series"] = (df["kd_level"] > shallow).map({False: PROPOSED, True: ONE_POINT})
    style.grouped_bars_by_epsilon(
        df,
        "series",
        ORDER,
        args.out,
        log_scale=True,
        colors=["white", "white"],
        hatches=["", "//"],
    )


if __name__ == "__main__":
    main()
