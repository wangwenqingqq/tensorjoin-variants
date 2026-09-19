import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import parse_results

_PREAMBLE = r"""
\usepackage{libertine}
\usepackage{bold-extra}
\usepackage{makecell}
"""

def _probe_usetex():
    import tempfile

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": _PREAMBLE,
    })
    figure = plt.figure()
    try:
        figure.text(0.5, 0.5, r"\textsc{\textbf{probe}} $\epsilon$")
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            figure.savefig(handle.name)
        return True
    except Exception:
        return False
    finally:
        plt.close(figure)


_USETEX = _probe_usetex()

if not _USETEX:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })
    print("LaTeX unavailable, falling back to mathtext rendering")


def tex(latex_text, plain_text):
    return latex_text if _USETEX else plain_text


def label_response_time():
    return tex(r"\textbf{\textsc{Response Time (s)}}", "Response Time (s)")


def label_dataset():
    return tex(r"\textbf{\textsc{Dataset}}", "Dataset")


def dataset_label(name):
    return tex(r"\textit{\textbf{%s}}" % name, name)


PROJECT = "RT-HiSS"


def legend_label(name, mono=False):
    body = r"{\ttfamily %s}" if mono else r"{\bfseries\scshape %s}"
    project = r"\textsc{%s}" % PROJECT

    return tex(body % name.replace(PROJECT, project), name)


def epsilon_ticks():
    return [
        tex(r"\textbf{\textsc{$\epsilon_s$}}", r"$\epsilon_s$"),
        tex(r"\textbf{\textsc{$\epsilon_m$}}", r"$\epsilon_m$"),
        tex(r"\textbf{\textsc{$\epsilon_l$}}", r"$\epsilon_l$"),
    ]


FONT_XTICKS = 20
FONT_YTICKS = 18
FONT_YLABEL = 18
FONT_DATASET_LABEL = 18
FONT_DATASET_TICK = 14
FONT_LEGEND = 16

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
]

HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo"]


def save(figure, out):
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print("wrote {}".format(out))


def grouped_bars(df, column, order, out, log_scale=False, colors=None, hatches=None,
                 mono_legend=False):
    colors = colors or PALETTE
    hatches = hatches or HATCHES
    series = [s for s in order if (df[column] == s).any()]
    if not series:
        return
    datasets = parse_results.order_datasets(df)

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(3.6 * len(datasets), 4.0), squeeze=False
    )
    ticks = epsilon_ticks()

    for index, name in enumerate(datasets):
        axis = axes[0][index]
        subset = df[df["Dataset"] == name]
        epsilons = sorted(subset["epsilon"].unique())
        positions = np.arange(len(epsilons))
        width = 0.8 / len(series)

        for series_index, label in enumerate(series):
            values = [
                subset.loc[
                    (subset[column] == label) & (subset["epsilon"] == eps),
                    "total_time",
                ].min()
                for eps in epsilons
            ]
            axis.bar(
                positions + series_index * width - 0.4 + width / 2,
                values,
                width,
                label=label,
                color=colors[series_index % len(colors)],
                hatch=hatches[series_index % len(hatches)],
                edgecolor="black",
                linewidth=0.5,
            )

        axis.set_xticks(positions)
        axis.set_xticklabels(ticks[: len(epsilons)], fontsize=FONT_XTICKS)
        axis.set_xlabel(dataset_label(name), fontsize=FONT_DATASET_LABEL)
        axis.tick_params(axis="y", labelsize=FONT_YTICKS)
        axis.grid(True, axis="y", linestyle=":", alpha=0.5)
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        if index == 0:
            axis.set_ylabel(label_response_time(), fontsize=FONT_YLABEL)

    figure.tight_layout()

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        [legend_label(l, mono=mono_legend) for l in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(series),
        fontsize=FONT_LEGEND,
        frameon=False,
    )

    save(figure, out)


def grouped_bars_by_epsilon(df, column, order, out, log_scale=False, colors=None,
                            hatches=None, mono_legend=False):
    colors = colors or PALETTE
    hatches = hatches or HATCHES
    series = [s for s in order if (df[column] == s).any()]
    if not series:
        return
    datasets = parse_results.order_datasets(df)
    ticks = epsilon_ticks()

    df = df.copy()
    df["eps_rank"] = df.groupby("Dataset")["epsilon"].rank(method="dense").astype(int) - 1
    ranks = sorted(df["eps_rank"].unique())

    figure, axes = plt.subplots(
        1, len(ranks), figsize=(3.6 * len(ranks), 4.0), squeeze=False
    )

    for index, rank in enumerate(ranks):
        axis = axes[0][index]
        subset = df[df["eps_rank"] == rank]
        positions = np.arange(len(datasets))
        width = 0.8 / len(series)

        for series_index, label in enumerate(series):
            values = [
                subset.loc[
                    (subset[column] == label) & (subset["Dataset"] == name),
                    "total_time",
                ].min()
                for name in datasets
            ]
            axis.bar(
                positions + series_index * width - 0.4 + width / 2,
                values,
                width,
                label=label,
                color=colors[series_index % len(colors)],
                hatch=hatches[series_index % len(hatches)],
                edgecolor="black",
                linewidth=0.5,
            )

        axis.set_xticks(positions)
        axis.set_xticklabels(
            [dataset_label(name) for name in datasets],
            fontsize=FONT_DATASET_TICK,
            rotation=30,
            ha="right",
        )
        axis.set_xlabel(ticks[index], fontsize=FONT_XTICKS)
        axis.tick_params(axis="y", labelsize=FONT_YTICKS)
        axis.grid(True, axis="y", linestyle=":", alpha=0.5)
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        if index == 0:
            axis.set_ylabel(label_response_time(), fontsize=FONT_YLABEL)

    figure.tight_layout()

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        [legend_label(l, mono=mono_legend) for l in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(series),
        fontsize=FONT_LEGEND,
        frameon=False,
    )

    save(figure, out)
