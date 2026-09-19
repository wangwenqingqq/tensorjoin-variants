import argparse

import numpy as np
import matplotlib.pyplot as plt

import parse_results
import plot_style as style

COMPONENTS = [
    ("intersection_time", "First Intersection Test Time"),
    ("rt_time", "RT Core Time"),
    ("cuda_time", "CUDA Core Time"),
    ("compress_time", "Compression Time"),
    ("copy_time", "Memory Copy Time"),
    ("kdtree_time", "KD-Tree Construction Time"),
    ("other_time", "CPU Time"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = parse_results.load(args.results)
    if df.empty:
        return

    df = df.dropna(subset=["total_time"])
    df = df.sort_values("epsilon").groupby("Dataset", as_index=False).first()

    datasets = parse_results.order_datasets(df)
    df = df.set_index("Dataset").loc[datasets]

    fractions = {}
    for key, label in COMPONENTS:
        values = df[key].fillna(0.0) if key in df.columns else 0.0
        fractions[label] = np.asarray(values, dtype=float) / np.asarray(
            df["total_time"], dtype=float
        )

    figure, axis = plt.subplots(figsize=(2.0 * len(datasets) + 4, 4.5))
    positions = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))

    for index, (_, label) in enumerate(COMPONENTS):
        values = fractions[label]
        axis.bar(
            positions,
            values,
            0.6,
            bottom=bottom,
            label=label,
            color=style.PALETTE[index % len(style.PALETTE)],
            hatch=style.HATCHES[index % len(style.HATCHES)],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += values

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [style.dataset_label(d) for d in datasets], fontsize=style.FONT_XTICKS
    )
    axis.set_ylabel(
        style.tex(r"\textbf{\textsc{Fraction of Total Time}}",
                  "Fraction of Total Time"),
        fontsize=style.FONT_YLABEL,
    )
    axis.set_xlabel(style.label_dataset(), fontsize=style.FONT_YLABEL)
    axis.tick_params(axis="y", labelsize=style.FONT_YTICKS)
    axis.set_ylim(0, max(1.0, float(bottom.max())))
    axis.grid(True, axis="y", linestyle=":", alpha=0.5)

    figure.tight_layout()

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        [style.legend_label(l) for l in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=style.FONT_LEGEND,
        frameon=False,
    )

    style.save(figure, args.out)


if __name__ == "__main__":
    main()
