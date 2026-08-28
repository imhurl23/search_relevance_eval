#!/usr/bin/env python3
"""Create publication-ready You.com normalized vs provider-native charts.

The paired effect charts use the task-bootstrap intervals already produced by
``analyze_variables.py``. The decision-space chart uses deployed-condition
means from ``variable_condition_summary.csv`` and intentionally omits cost,
which is not available in the root-span export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUTPUT = ANALYSIS / "plots" / "search_comparison"
# The Braintrust plot helpers ship as a skill outside this repository. Point
# BRAINTRUST_PLOTS_DIR at the directory holding braintrust_viz.py; the default
# is the skill's usual install location under the current user's home.
BT_VIZ = Path(
    os.environ.get("BRAINTRUST_PLOTS_DIR")
    or Path.home() / ".codex" / "skills" / "braintrust-plots" / "scripts"
)
if not (BT_VIZ / "braintrust_viz.py").exists():
    raise SystemExit(
        f"braintrust_viz.py not found under {BT_VIZ}. Install the braintrust-plots "
        "skill or set BRAINTRUST_PLOTS_DIR to the directory that contains it."
    )
sys.path.insert(0, str(BT_VIZ))

from braintrust_viz import (  # noqa: E402
    BRAND,
    NEUTRAL,
    clean_axes,
    header,
    make_figure,
    percent_formatter,
    savefig,
    use_braintrust_theme,
)


MODEL_ORDER = ["GPT-5.6 Terra", "Claude Sonnet 5"]
MODEL_SHORT = {"GPT-5.6 Terra": "GPT-5.6 Terra", "Claude Sonnet 5": "Claude Sonnet 5"}


def native_effects() -> pd.DataFrame:
    """Return You.com normalized minus native effects with paired CIs."""
    primary = pd.read_csv(ANALYSIS / "variable_main_effects.csv")
    primary = primary[
        (primary["family"] == "Native vs normalized")
        & (primary["model"].isin(MODEL_ORDER))
    ].copy()
    primary["scorer"] = "Strict answer match"

    secondary = pd.read_csv(ANALYSIS / "variable_secondary_outcome_effects.csv")
    secondary = secondary[
        (secondary["comparison"] == "Native - normalized")
        & (secondary["metric"] == "judge")
        & (secondary["model"].isin(MODEL_ORDER))
    ].copy()
    secondary["scorer"] = "Semantic judge"

    columns = ["model", "scorer", "n", "effect", "ci_low", "ci_high", "wins", "ties", "losses"]
    effects = pd.concat([primary[columns], secondary[columns]], ignore_index=True)
    # Source contrasts are native - normalized; reverse them so positive always
    # means the You.com normalized harness performed better.
    effects[["effect", "ci_low", "ci_high"]] = effects[["effect", "ci_low", "ci_high"]].mul(-1)
    effects[["ci_low", "ci_high"]] = effects[["ci_high", "ci_low"]].to_numpy()
    return effects


def plot_quality_effects(effects: pd.DataFrame) -> Path:
    order = [
        ("GPT-5.6 Terra", "Strict answer match"),
        ("GPT-5.6 Terra", "Semantic judge"),
        ("Claude Sonnet 5", "Strict answer match"),
        ("Claude Sonnet 5", "Semantic judge"),
    ]
    plotted = effects.set_index(["model", "scorer"]).loc[order].reset_index()
    y = np.array([3.2, 2.25, 0.85, -0.10])
    colors = {
        "Strict answer match": BRAND["indigo"],
        "Semantic judge": BRAND["purple"],
    }

    fig, ax = make_figure(height=6.8, left=0.22, right=0.94, bottom=0.15)
    ax.axvline(0, color=NEUTRAL["muted"], linewidth=1.4, linestyle="--", zorder=1)
    for idx, row in plotted.iterrows():
        color = colors[row["scorer"]]
        ax.errorbar(
            row["effect"], y[idx],
            xerr=[[row["effect"] - row["ci_low"]], [row["ci_high"] - row["effect"]]],
            fmt="o", markersize=11, color=color, ecolor=color,
            elinewidth=2.5, capsize=6, capthick=2.2, markeredgecolor="white",
            markeredgewidth=2, zorder=4,
        )
        ax.text(
            row["ci_high"] + 0.0025, y[idx], f"{row['effect']:+.1%}",
            va="center", ha="left", fontsize=12.5, fontweight="bold", color=NEUTRAL["ink"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}\n{s}" for m, s in order])
    ax.set_xlim(-0.032, 0.072)
    ax.set_ylim(-0.75, 3.85)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.set_xlabel("Accuracy lift from You.com normalized (percentage points)")
    ax.text(0.0015, 3.72, "You.com better", color=NEUTRAL["subtle"], fontsize=11.5)
    ax.text(-0.0015, 3.72, "Native better", color=NEUTRAL["subtle"], fontsize=11.5, ha="right")
    header(
        ax,
        "You.com improves GPT accuracy; Claude is effectively tied",
        "Paired task effects with 95% task-bootstrap intervals; positive values favor the normalized You.com harness.",
        key="n = 1,128 matched GPT tasks; n = 1,079 matched Claude tasks",
    )
    clean_axes(ax, grid_axis="x")
    path = OUTPUT / "01_native_vs_you_quality_effects.png"
    savefig(fig, path)
    plt.close(fig)
    return path


def plot_latency_effects() -> Path:
    effects = pd.read_csv(ANALYSIS / "variable_secondary_outcome_effects.csv")
    effects = effects[
        (effects["comparison"] == "Native - normalized")
        & (effects["metric"] == "latency_s")
        & (effects["model"].isin(MODEL_ORDER))
    ].set_index("model").loc[MODEL_ORDER].reset_index()

    fig, ax = make_figure(height=5.9, left=0.22, right=0.93, bottom=0.17)
    y = np.arange(len(effects))[::-1]
    colors = [BRAND["indigo"], BRAND["purple"]]
    bars = ax.barh(y, effects["effect"], height=0.54, color=colors, edgecolor="white", linewidth=2, zorder=2)
    ax.errorbar(
        effects["effect"], y,
        xerr=np.vstack([effects["effect"] - effects["ci_low"], effects["ci_high"] - effects["effect"]]),
        fmt="none", ecolor=NEUTRAL["ink"], elinewidth=2, capsize=6, capthick=2, zorder=4,
    )
    for bar, effect, high in zip(bars, effects["effect"], effects["ci_high"]):
        ax.text(
            high + 0.12, bar.get_y() + bar.get_height() / 2, f"{effect:.1f}s faster",
            va="center", ha="left", fontsize=14, fontweight="bold", color=NEUTRAL["ink"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(effects["model"])
    ax.set_xlim(0, 5.35)
    ax.set_xlabel("Latency saved per answer with You.com normalized (seconds)")
    header(
        ax,
        "You.com is faster for both models",
        "Mean paired latency savings versus provider-native search, with 95% task-bootstrap intervals.",
        key="GPT: 4.0s saved [3.6, 4.5] · Claude: 1.4s saved [0.8, 2.0]",
    )
    clean_axes(ax, grid_axis="x")
    path = OUTPUT / "02_native_vs_you_latency_savings.png"
    savefig(fig, path)
    plt.close(fig)
    return path


def plot_decision_space() -> Path:
    summary = pd.read_csv(ANALYSIS / "variable_condition_summary.csv")
    points = summary[
        summary["model"].isin(MODEL_ORDER) & summary["arm"].isin(["Normalized", "Native"])
    ].copy()
    points["label"] = points["model"].map(MODEL_SHORT) + " · " + points["arm"].replace({"Normalized": "You.com"})

    fig, ax = make_figure(height=7.2, left=0.11, right=0.92, bottom=0.15)
    color_map = {"Normalized": BRAND["indigo"], "Native": BRAND["pink"]}
    marker_map = {"GPT-5.6 Terra": "o", "Claude Sonnet 5": "s"}
    offsets = {
        ("GPT-5.6 Terra", "Normalized"): (0.18, 0.001),
        ("GPT-5.6 Terra", "Native"): (0.18, -0.006),
        ("Claude Sonnet 5", "Normalized"): (-0.18, 0.003),
        ("Claude Sonnet 5", "Native"): (-0.18, -0.007),
    }
    for _, row in points.iterrows():
        ax.scatter(
            row["latency_mean_s"], row["gated"], s=330,
            color=color_map[row["arm"]], marker=marker_map[row["model"]],
            edgecolor="white", linewidth=3, zorder=4,
        )
        dx, dy = offsets[(row["model"], row["arm"])]
        ha = "left" if dx > 0 else "right"
        ax.text(
            row["latency_mean_s"] + dx, row["gated"] + dy,
            f"{row['label']}\n{row['gated']:.1%} · {row['latency_mean_s']:.1f}s",
            ha=ha, va="center", fontsize=11.5, fontweight="bold", color=NEUTRAL["ink"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=5,
        )

    ax.annotate(
        "better accuracy\nand lower latency",
        xy=(8.9, 0.423), xytext=(10.35, 0.427),
        arrowprops={"arrowstyle": "->", "color": BRAND["indigo"], "lw": 1.7},
        color=BRAND["indigo"], fontsize=11.5, fontweight="bold", ha="center",
    )
    ax.set_xlim(8.0, 15.2)
    ax.set_ylim(0.325, 0.44)
    ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.set_xlabel("Mean end-to-end latency (seconds) — lower is better")
    ax.set_ylabel("Strict answer match — higher is better")
    header(
        ax,
        "The normalized You.com harness moves both models up and left",
        "Deployed-condition means across 1,329 rows per condition; shape identifies model and color identifies search path.",
        key="Circles: GPT-5.6 Terra   Squares: Claude Sonnet 5   Indigo: You.com normalized   Pink: provider native",
    )
    clean_axes(ax, grid_axis="both")
    path = OUTPUT / "03_quality_latency_decision_space.png"
    savefig(fig, path)
    plt.close(fig)
    return path


def plot_editorial_hero() -> Path:
    """One concise native-to-You.com story across quality and latency."""
    summary = pd.read_csv(ANALYSIS / "variable_condition_summary.csv")
    points = summary[
        summary["model"].isin(MODEL_ORDER) & summary["arm"].isin(["Normalized", "Native"])
    ].set_index(["model", "arm"])

    use_braintrust_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(top=0.69, left=0.12, right=0.96, bottom=0.20)
    fig.patch.set_facecolor("white")

    fig.text(
        0.12, 0.93,
        "You.com is faster for both models—without sacrificing quality",
        fontsize=22, fontweight="bold", color=NEUTRAL["ink"], ha="left",
    )
    fig.text(
        0.12, 0.875,
        "It lifts GPT answer match and keeps Claude effectively tied, while reducing latency for both.",
        fontsize=14.5, color=NEUTRAL["subtle"], ha="left",
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BRAND["pink"],
               markeredgecolor="white", markeredgewidth=1.5, markersize=11, label="Provider native"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BRAND["indigo"],
               markeredgecolor="white", markeredgewidth=1.5, markersize=11, label="You.com normalized"),
    ]
    fig.legend(
        handles=legend, loc="upper left", bbox_to_anchor=(0.115, 0.825),
        ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.5, fontsize=12.5,
    )

    y_by_model = {"GPT-5.6 Terra": 1, "Claude Sonnet 5": 0}
    label_by_model = {"GPT-5.6 Terra": "GPT-5.6 Terra", "Claude Sonnet 5": "Claude Sonnet 5"}

    # Quality panel: right is better.
    ax = axes[0]
    for model in MODEL_ORDER:
        y = y_by_model[model]
        native = points.loc[(model, "Native"), "gated"]
        you = points.loc[(model, "Normalized"), "gated"]
        ax.plot([native, you], [y, y], color=NEUTRAL["thin"], linewidth=5, solid_capstyle="round", zorder=1)
        ax.scatter(native, y, s=240, color=BRAND["pink"], edgecolor="white", linewidth=2.5, zorder=3)
        ax.scatter(you, y, s=240, color=BRAND["indigo"], edgecolor="white", linewidth=2.5, zorder=3)
        label_y = y + 0.17
        ax.text(native - 0.0013, label_y, f"{native:.1%}", ha="right", va="bottom", fontsize=13, color=NEUTRAL["subtle"])
        ax.text(you + 0.0013, label_y, f"{you:.1%}", ha="left", va="bottom", fontsize=14, fontweight="bold", color=BRAND["indigo"])

    ax.set_title("Answer match", loc="left", fontsize=17, fontweight="bold", pad=18)
    ax.text(1, 1.07, "higher is better", transform=ax.transAxes, ha="right", color=NEUTRAL["subtle"], fontsize=11.5)
    ax.set_xlim(0.33, 0.43)
    ax.set_ylim(-0.43, 1.43)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([label_by_model[m] for m in MODEL_ORDER])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.set_xlabel("")
    clean_axes(ax, grid_axis="x")

    # Latency panel: reverse the axis so improvement also moves left-to-right.
    ax = axes[1]
    for model in MODEL_ORDER:
        y = y_by_model[model]
        native = points.loc[(model, "Native"), "latency_mean_s"]
        you = points.loc[(model, "Normalized"), "latency_mean_s"]
        ax.plot([native, you], [y, y], color=NEUTRAL["thin"], linewidth=5, solid_capstyle="round", zorder=1)
        ax.scatter(native, y, s=240, color=BRAND["pink"], edgecolor="white", linewidth=2.5, zorder=3)
        ax.scatter(you, y, s=240, color=BRAND["indigo"], edgecolor="white", linewidth=2.5, zorder=3)
        ax.text(native, y + 0.17, f"{native:.1f}s", ha="center", va="bottom", fontsize=13, color=NEUTRAL["subtle"])
        ax.text(you, y + 0.17, f"{you:.1f}s", ha="center", va="bottom", fontsize=14, fontweight="bold", color=BRAND["indigo"])

    ax.set_title("End-to-end latency", loc="left", fontsize=17, fontweight="bold", pad=18)
    ax.text(1, 1.07, "lower is better", transform=ax.transAxes, ha="right", color=NEUTRAL["subtle"], fontsize=11.5)
    ax.set_xlim(15.0, 8.5)
    ax.set_ylim(-0.43, 1.43)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fs"))
    ax.set_xlabel("")
    clean_axes(ax, despine_left=True, grid_axis="x")

    fig.text(
        0.12, 0.08,
        "Paired inference: GPT quality improves; Claude quality is statistically tied. Both latency reductions are clear.",
        fontsize=12.5, color=NEUTRAL["subtle"], ha="left",
    )
    path = OUTPUT / "00_search_comparison_editorial_hero.png"
    savefig(fig, path)
    plt.close(fig)
    return path


def export_plot_data(effects: pd.DataFrame) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "search_comparison_plot_data.csv"
    effects.to_csv(path, index=False)
    return path


def main() -> None:
    use_braintrust_theme()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    effects = native_effects()
    paths = [
        export_plot_data(effects),
        plot_editorial_hero(),
        plot_quality_effects(effects),
        plot_latency_effects(),
        plot_decision_space(),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
