#!/usr/bin/env python3
"""Interpretable timing and question-context analysis for LiveNewsBench."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
PLOTS = ANALYSIS / "plots"
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
    categorical,
    clean_axes,
    header,
    make_figure,
    percent_formatter,
    savefig,
    use_braintrust_theme,
)

SEED = 20260826
BOOTSTRAPS = 5000
MODEL_ORDER = ["GLM-5.2", "DeepSeek V4", "GPT-5.6 Terra", "Claude Sonnet 5"]
AGE_ORDER = ["Newest: 8–9 mo", "10–11 mo", "12–13 mo", "Oldest: 14–16 mo"]
FEATURE_LABELS = {
    "age_90": "Event is 3 months older",
    "composed": "Quantitative / composed",
    "multi_year": "References multiple years",
    "temporal_scope": "Explicit temporal scope",
    "source_constrained": "Names a source / authority",
    "q_words_10": "10 additional question words",
}


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    text = frame["question"].fillna("").str.lower()
    frame["question_words"] = frame["question"].fillna("").str.split().str.len()
    frame["q_words_10"] = (frame["question_words"] - frame["question_words"].mean()) / 10
    frame["age_90"] = (frame["event_age_days"] - frame["event_age_days"].mean()) / 90
    frame["composed"] = (frame["question_type"] == "Quantitative / composed").astype(int)
    years = frame["question"].fillna("").map(lambda value: len(set(re.findall(r"\b20\d{2}\b", value))))
    frame["multi_year"] = (years >= 2).astype(int)
    frame["temporal_scope"] = text.str.contains(
        r"\b(?:as of|between|during|since|before|after|from|through|within|by the time|over the course|elapsed)\b",
        regex=True,
    ).astype(int)
    frame["source_constrained"] = text.str.contains(
        r"\b(?:according to|official|reported by|coverage|article|statement|website|database|report|results|records|data)\b",
        regex=True,
    ).astype(int)
    frame["age_bin"] = pd.cut(
        frame["event_age_days"],
        bins=[0, 275, 335, 400, 999],
        labels=AGE_ORDER,
        include_lowest=True,
    )
    return frame


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(ANALYSIS / "variable_row_level.csv")
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame["arm"] = pd.Categorical(frame["arm"], ["No search", "Normalized", "Wide", "Native"], ordered=True)
    return add_features(frame)


def clustered_mean_ci(frame: pd.DataFrame, value: str) -> tuple[float, float, float, int, int]:
    task = frame.dropna(subset=[value]).groupby("task_key", observed=True)[value].mean().to_numpy(float)
    rng = np.random.default_rng(SEED + sum(map(ord, value)) + len(frame))
    boot = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        idx = rng.integers(0, len(task), size=(size, len(task)))
        boot[start:start + size] = task[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(task.mean()), float(lo), float(hi), len(frame), len(task)


def age_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ("No search", "Normalized"):
        for model in MODEL_ORDER:
            for age_bin in AGE_ORDER:
                subset = frame[(frame["arm"] == arm) & (frame["model"] == model) & (frame["age_bin"] == age_bin)]
                for outcome in ("gated", "judge"):
                    rate, lo, hi, n_rows, n_tasks = clustered_mean_ci(subset, outcome)
                    rows.append({
                        "arm": arm, "model": model, "age_bin": age_bin, "outcome": outcome,
                        "rate": rate, "ci_low": lo, "ci_high": hi,
                        "n_rows": n_rows, "n_tasks": n_tasks,
                    })
    return pd.DataFrame(rows)


def paired_gain_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_row_id", "task_key", "model"]
    keep = [*keys, "category", "age_bin", "age_90", "composed", "multi_year",
            "temporal_scope", "source_constrained", "q_words_10", "gated", "judge"]
    normalized = frame[frame["arm"] == "Normalized"][keep].copy()
    no_search = frame[frame["arm"] == "No search"][[*keys, "gated", "judge"]].copy()
    joined = normalized.merge(no_search, on=keys, suffixes=("_normalized", "_none"), validate="one_to_one")
    joined["gated_gain"] = joined["gated_normalized"] - joined["gated_none"]
    joined["judge_gain"] = joined["judge_normalized"] - joined["judge_none"]
    return joined


def age_gains(gains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for age_bin in AGE_ORDER:
            subset = gains[(gains["model"] == model) & (gains["age_bin"] == age_bin)]
            for outcome in ("gated_gain", "judge_gain"):
                rate, lo, hi, n_rows, n_tasks = clustered_mean_ci(subset, outcome)
                rows.append({
                    "model": model, "age_bin": age_bin, "outcome": outcome,
                    "gain": rate, "ci_low": lo, "ci_high": hi,
                    "n_rows": n_rows, "n_tasks": n_tasks,
                })
    return pd.DataFrame(rows)


def adjusted_associations(frame: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    feature_terms = " + ".join(FEATURE_LABELS)
    specs = [
        ("No-search semantic accuracy", frame[frame["arm"] == "No search"].copy(), "judge"),
        ("Normalized semantic accuracy", frame[frame["arm"] == "Normalized"].copy(), "judge"),
        ("Retrieval gain", gains.copy(), "judge_gain"),
    ]
    rows = []
    for analysis, data, outcome in specs:
        for field in ("model", "category"):
            data[field + "_s"] = data[field].astype("string")
        fit = smf.ols(
            f"{outcome} ~ {feature_terms} + C(model_s) + C(category_s)",
            data=data,
        ).fit(cov_type="cluster", cov_kwds={"groups": data["task_key"]})
        for feature, label in FEATURE_LABELS.items():
            lo, hi = fit.conf_int().loc[feature]
            rows.append({
                "analysis": analysis, "feature": feature, "label": label,
                "effect": fit.params[feature], "ci_low": lo, "ci_high": hi,
                "p_value": fit.pvalues[feature], "n_rows": int(fit.nobs),
                "n_tasks": data["task_key"].nunique(), "r_squared": fit.rsquared,
            })
    return pd.DataFrame(rows)


def age_slopes_by_model(frame: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    controls = "composed + multi_year + temporal_scope + source_constrained + q_words_10 + C(category_s)"
    rows = []
    for model in MODEL_ORDER:
        specs = [
            ("No-search semantic accuracy", frame[(frame["arm"] == "No search") & (frame["model"] == model)].copy(), "judge"),
            ("Normalized semantic accuracy", frame[(frame["arm"] == "Normalized") & (frame["model"] == model)].copy(), "judge"),
            ("Retrieval gain", gains[gains["model"] == model].copy(), "judge_gain"),
        ]
        for analysis, data, outcome in specs:
            data["category_s"] = data["category"].astype("string")
            fit = smf.ols(f"{outcome} ~ age_90 + {controls}", data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data["task_key"]}
            )
            lo, hi = fit.conf_int().loc["age_90"]
            rows.append({
                "model": model, "analysis": analysis, "effect_per_90d": fit.params["age_90"],
                "ci_low": lo, "ci_high": hi, "p_value": fit.pvalues["age_90"],
                "n_rows": int(fit.nobs), "n_tasks": data["task_key"].nunique(),
            })
    return pd.DataFrame(rows)


def search_escalation(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame[frame["arm"] == "Normalized"].copy()
    quartile_order = ["Fastest", "Q2", "Q3", "Slowest"]
    normalized["latency_quartile"] = normalized.groupby("model", observed=True)["latency_s"].transform(
        lambda values: pd.qcut(values, 4, labels=quartile_order, duplicates="drop")
    )
    rows = []
    for model in MODEL_ORDER:
        for quartile in quartile_order:
            subset = normalized[(normalized["model"] == model) & (normalized["latency_quartile"] == quartile)]
            rate, lo, hi, n_rows, n_tasks = clustered_mean_ci(subset, "judge")
            rows.append({
                "model": model, "latency_quartile": quartile,
                "semantic_accuracy": rate, "ci_low": lo, "ci_high": hi,
                "latency_mean_s": subset["latency_s"].mean(),
                "searches_mean": subset["used_searches"].mean(),
                "n_rows": n_rows, "n_tasks": n_tasks,
            })
    return pd.DataFrame(rows)


def plot_age_lines(data: pd.DataFrame, *, outcome: str, arm: str, filename: str,
                   title: str, subtitle: str, ylabel: str) -> None:
    subset = data[(data["outcome"] == outcome) & (data["arm"] == arm)].copy()
    fig, ax = make_figure(width=12, height=8.0, left=.10, right=.96, top=.72, bottom=.22)
    x = np.arange(len(AGE_ORDER))
    colors = categorical(len(MODEL_ORDER))
    offsets = [-.045, -.015, .015, .045]
    for model, color, offset in zip(MODEL_ORDER, colors, offsets):
        part = subset[subset["model"] == model].set_index("age_bin").loc[AGE_ORDER]
        y = part["rate"].to_numpy(); lo = part["ci_low"].to_numpy(); hi = part["ci_high"].to_numpy()
        ax.plot(x + offset, y, marker="o", ms=9, lw=3, color=color, label=model, zorder=4)
        ax.errorbar(x + offset, y, yerr=np.vstack([y-lo, hi-y]), fmt="none", color=color,
                    lw=2, capsize=5, capthick=2, zorder=5)
    ax.set_xticks(x, AGE_ORDER)
    ax.set_ylabel(ylabel); ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    header(ax, title, subtitle, "Rates include truncations, refusals, and zero-search behavior")
    clean_axes(ax, grid_axis="y")
    savefig(fig, PLOTS / filename); plt.close(fig)


def plot_gain_lines(data: pd.DataFrame) -> None:
    subset = data[data["outcome"] == "judge_gain"].copy()
    fig, ax = make_figure(width=12, height=8.0, left=.10, right=.96, top=.72, bottom=.22)
    x = np.arange(len(AGE_ORDER)); colors = categorical(len(MODEL_ORDER)); offsets = [-.045, -.015, .015, .045]
    for model, color, offset in zip(MODEL_ORDER, colors, offsets):
        part = subset[subset["model"] == model].set_index("age_bin").loc[AGE_ORDER]
        y = part["gain"].to_numpy(); lo = part["ci_low"].to_numpy(); hi = part["ci_high"].to_numpy()
        ax.plot(x + offset, y, marker="o", ms=9, lw=3, color=color, label=model, zorder=4)
        ax.errorbar(x + offset, y, yerr=np.vstack([y-lo, hi-y]), fmt="none", color=color,
                    lw=2, capsize=5, capthick=2, zorder=5)
    ax.axhline(0, color=NEUTRAL["subtle"], ls=(0, (4, 4)), lw=1.5)
    ax.set_xticks(x, AGE_ORDER); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Normalized retrieval − no search")
    ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    header(ax, "Retrieval matters most for the newest events",
           "Paired semantic-accuracy gain by event age; 95% task-bootstrap intervals",
           "Total-system outcome: failures remain in the denominator")
    clean_axes(ax, grid_axis="y")
    savefig(fig, PLOTS / "08_retrieval_gain_by_event_age.png"); plt.close(fig)


def plot_adjusted_context(data: pd.DataFrame) -> None:
    analyses = ["Normalized semantic accuracy", "Retrieval gain"]
    subset = data[data["analysis"].isin(analyses)].copy()
    order = list(FEATURE_LABELS.values())
    fig, ax = make_figure(width=12, height=7.8, left=.25, right=.96, top=.73)
    y = np.arange(len(order)); offsets = {analyses[0]: -.14, analyses[1]: .14}
    colors = {analyses[0]: BRAND["indigo"], analyses[1]: BRAND["pink"]}
    for analysis in analyses:
        part = subset[subset["analysis"] == analysis].set_index("label").loc[order]
        effect = part["effect"].to_numpy(); lo = part["ci_low"].to_numpy(); hi = part["ci_high"].to_numpy()
        ax.errorbar(effect, y + offsets[analysis], xerr=np.vstack([effect-lo, hi-effect]),
                    fmt="o", ms=10, lw=2.5, capsize=5, capthick=2, color=colors[analysis],
                    label=analysis, zorder=5)
    ax.axvline(0, color=NEUTRAL["subtle"], ls=(0, (4, 4)), lw=1.5)
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.set_xlabel("Adjusted percentage-point association")
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.01))
    header(ax, "Question context changes both difficulty and retrieval value",
           "Linear probability models adjust for model and news category; task-clustered 95% intervals",
           "Associational context effects—not randomized treatments")
    clean_axes(ax, despine_left=True, grid_axis="x")
    savefig(fig, PLOTS / "09_adjusted_question_context_effects.png"); plt.close(fig)


def plot_search_escalation(data: pd.DataFrame) -> None:
    order = ["Fastest", "Q2", "Q3", "Slowest"]
    fig, ax = make_figure(width=12, height=8.0, left=.10, right=.96, top=.72, bottom=.22)
    x = np.arange(len(order)); colors = categorical(len(MODEL_ORDER)); offsets = [-.045, -.015, .015, .045]
    for model, color, offset in zip(MODEL_ORDER, colors, offsets):
        part = data[data["model"] == model].set_index("latency_quartile").loc[order]
        y = part["semantic_accuracy"].to_numpy(); lo = part["ci_low"].to_numpy(); hi = part["ci_high"].to_numpy()
        ax.plot(x + offset, y, marker="o", ms=9, lw=3, color=color, label=model, zorder=4)
        ax.errorbar(x + offset, y, yerr=np.vstack([y-lo, hi-y]), fmt="none", color=color,
                    lw=2, capsize=5, capthick=2, zorder=5)
    ax.set_xticks(x, order); ax.set_ylim(.45, 1.01)
    ax.set_ylabel("Normalized-search semantic accuracy"); ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    header(ax, "Slow runs are a search-escalation signal",
           "Latency quartiles are defined within each model; 95% task-bootstrap intervals",
           "Post-treatment diagnostic: difficult questions cause both longer runs and more errors")
    clean_axes(ax, grid_axis="y")
    savefig(fig, PLOTS / "10_accuracy_by_latency_quartile.png"); plt.close(fig)


def main() -> int:
    use_braintrust_theme(); PLOTS.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    rates = age_rates(frame)
    gains = paired_gain_rows(frame)
    gain_rates = age_gains(gains)
    adjusted = adjusted_associations(frame, gains)
    slopes = age_slopes_by_model(frame, gains)
    escalation = search_escalation(frame)
    features = ["dataset_row_id", "task_key", "model", "arm", "category", "event_age_days", "age_bin",
                "question_words", *FEATURE_LABELS, "gated", "judge"]
    frame[features].to_csv(ANALYSIS / "context_timing_rows.csv", index=False)
    rates.to_csv(ANALYSIS / "context_age_rates.csv", index=False)
    gain_rates.to_csv(ANALYSIS / "context_age_retrieval_gains.csv", index=False)
    adjusted.to_csv(ANALYSIS / "context_adjusted_associations.csv", index=False)
    slopes.to_csv(ANALYSIS / "context_age_slopes_by_model.csv", index=False)
    escalation.to_csv(ANALYSIS / "context_search_escalation.csv", index=False)
    plot_age_lines(rates, outcome="judge", arm="No search", filename="07_no_search_accuracy_by_event_age.png",
                   title="No-search recall improves as events get older",
                   subtitle="Semantic accuracy without search, by time from event to evaluation; 95% task-bootstrap intervals",
                   ylabel="No-search semantic accuracy")
    plot_gain_lines(gain_rates)
    plot_adjusted_context(adjusted)
    plot_search_escalation(escalation)
    print("FEATURE PREVALENCE")
    one = frame[frame["arm"] == "Normalized"].drop_duplicates("dataset_row_id")
    print(one[["question_words", "composed", "multi_year", "temporal_scope", "source_constrained"]].agg(["mean", "sum"]).to_string())
    print("\nADJUSTED ASSOCIATIONS\n", adjusted.to_string(index=False))
    print("\nAGE SLOPES BY MODEL\n", slopes.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
