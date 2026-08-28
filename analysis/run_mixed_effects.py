#!/usr/bin/env python3
"""Fit the preregistered task-random-intercept accuracy model.

The primary model is a binomial logistic mixed model with one fixed effect for
each deployed condition and a random intercept for task_key. Instrumentation-
truncated rows are excluded, matching docs/study-design.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
INPUT = ANALYSIS / "variable_row_level.csv"

CONDITION_ORDER = [
    "DeepSeek V4 | No search",
    "DeepSeek V4 | Normalized",
    "DeepSeek V4 | Wide",
    "GLM-5.2 | No search",
    "GLM-5.2 | Normalized",
    "GLM-5.2 | Wide",
    "GPT-5.6 Terra | No search",
    "GPT-5.6 Terra | Normalized",
    "GPT-5.6 Terra | Wide",
    "GPT-5.6 Terra | Native",
    "Claude Sonnet 5 | No search",
    "Claude Sonnet 5 | Normalized",
    "Claude Sonnet 5 | Wide",
    "Claude Sonnet 5 | Native",
]


def fit_model(frame: pd.DataFrame):
    frame = frame.copy()
    frame["condition_label"] = frame["model"] + " | " + frame["arm"]
    frame["condition_label"] = pd.Categorical(
        frame["condition_label"], categories=CONDITION_ORDER, ordered=True
    )
    if frame["condition_label"].isna().any():
        raise ValueError("Unexpected condition label in analysis data")

    x = pd.get_dummies(frame["condition_label"], dtype=float)[CONDITION_ORDER]
    task = pd.Categorical(frame["task_key"])
    row_index = np.arange(len(frame))
    z = sparse.csr_matrix(
        (np.ones(len(frame)), (row_index, task.codes)),
        shape=(len(frame), len(task.categories)),
    )
    model = BinomialBayesMixedGLM(
        frame["gated"].to_numpy(float),
        x.to_numpy(float),
        z,
        np.zeros(z.shape[1], dtype=int),
        vcp_p=1.0,
        fe_p=2.0,
        fep_names=CONDITION_ORDER,
        vcp_names=["task intercept SD"],
        vc_names=[str(value) for value in task.categories],
    )
    return model.fit_vb(verbose=False), frame


def condition_estimates(result, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(CONDITION_ORDER):
        log_odds = result.fe_mean[idx]
        log_odds_sd = result.fe_sd[idx]
        subset = frame[frame["condition_label"] == label]
        rows.append({
            "condition": label,
            "n_rows": len(subset),
            "n_tasks": subset["task_key"].nunique(),
            "observed_accuracy": subset["gated"].mean(),
            "conditional_probability": expit(log_odds),
            "log_odds": log_odds,
            "log_odds_sd": log_odds_sd,
            "log_odds_ci_low": log_odds - 1.96 * log_odds_sd,
            "log_odds_ci_high": log_odds + 1.96 * log_odds_sd,
        })
    return pd.DataFrame(rows)


def registered_contrasts(result) -> pd.DataFrame:
    index = {label: idx for idx, label in enumerate(CONDITION_ORDER)}
    comparisons = []
    for model in ["DeepSeek V4", "GLM-5.2", "GPT-5.6 Terra", "Claude Sonnet 5"]:
        comparisons.append((model, "Normalized vs no search", "Normalized", "No search"))
        comparisons.append((model, "Wide vs normalized", "Wide", "Normalized"))
        if model in {"GPT-5.6 Terra", "Claude Sonnet 5"}:
            comparisons.append((model, "Native vs normalized", "Native", "Normalized"))

    rows = []
    for model, comparison, arm_a, arm_b in comparisons:
        a = index[f"{model} | {arm_a}"]
        b = index[f"{model} | {arm_b}"]
        delta = result.fe_mean[a] - result.fe_mean[b]
        # statsmodels' VB approximation is mean-field, so fixed-effect posterior
        # covariances are zero and the contrast variance is the sum of variances.
        sd = np.sqrt(result.fe_sd[a] ** 2 + result.fe_sd[b] ** 2)
        low, high = delta - 1.96 * sd, delta + 1.96 * sd
        rows.append({
            "model": model,
            "comparison": comparison,
            "log_odds_difference": delta,
            "posterior_sd": sd,
            "odds_ratio": np.exp(delta),
            "odds_ratio_ci_low": np.exp(low),
            "odds_ratio_ci_high": np.exp(high),
            "approx_posterior_probability_positive": 0.5 * (
                1 + math.erf(delta / (sd * np.sqrt(2)))
            ) if sd > 0 else float(delta > 0),
        })
    return pd.DataFrame(rows)


def write_report(
    frame: pd.DataFrame,
    result,
    conditions: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    task_sd = float(np.exp(result.vcp_mean[0]))
    task_var = task_sd ** 2
    icc = task_var / (task_var + np.pi ** 2 / 3)
    lines = [
        "# Task-random-intercept mixed-effects regression",
        "",
        "## Specification",
        "",
        "A binomial logistic mixed model predicts `gated_answer_match` from the 14 deployed "
        "model-by-retrieval conditions, with a random intercept for `task_key`. The model uses "
        "a variational-Bayes fit from `statsmodels`. Instrumentation-truncated rows are excluded "
        "as preregistered. Intervals below are approximate 95% posterior intervals from the "
        "mean-field variational approximation.",
        "",
        f"- Rows: {len(frame):,}",
        f"- Unique tasks: {frame['task_key'].nunique():,}",
        f"- Excluded truncated rows: {int(pd.read_csv(INPUT)['truncated'].sum()):,}",
        f"- Estimated task-intercept SD: {task_sd:.3f}",
        f"- Latent-scale task ICC: {icc:.1%}",
        "",
        "## Registered contrasts",
        "",
        "Odds ratios above 1 favor the first arm named in the comparison.",
        "",
        "| Model | Comparison | Odds ratio | Approx. 95% interval | P(effect > 0) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.comparison} | {row.odds_ratio:.2f} | "
            f"[{row.odds_ratio_ci_low:.2f}, {row.odds_ratio_ci_high:.2f}] | "
            f"{row.approx_posterior_probability_positive:.3f} |"
        )
    lines += [
        "",
        "## Condition estimates",
        "",
        "`Conditional probability` evaluates each condition at a task random intercept of zero; "
        "it is not a population-marginal probability.",
        "",
        "| Condition | n | Observed accuracy | Conditional probability |",
        "|---|---:|---:|---:|",
    ]
    for row in conditions.itertuples(index=False):
        condition = row.condition.replace("|", "\\|")
        lines.append(
            f"| {condition} | {row.n_rows:,} | {row.observed_accuracy:.1%} | "
            f"{row.conditional_probability:.1%} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The model accounts for repeated conditions on the same task, but it does not turn "
        "post-treatment retrieval metrics into causal mediators. Native conditions remain "
        "structurally unavailable for DeepSeek and GLM. Variational-Bayes intervals may be "
        "narrower than likelihood-based intervals, so the task-bootstrap analysis remains the "
        "registered reproducible baseline.",
    ]
    (ANALYSIS / "livenewsbench-full-sonnet-v2-mixed-effects.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    raw = pd.read_csv(INPUT)
    frame = raw[(~raw["truncated"]) & raw["gated"].notna()].copy()
    result, frame = fit_model(frame)
    conditions = condition_estimates(result, frame)
    contrasts = registered_contrasts(result)
    conditions.to_csv(ANALYSIS / "mixed_effects_condition_estimates.csv", index=False)
    contrasts.to_csv(ANALYSIS / "mixed_effects_registered_contrasts.csv", index=False)
    write_report(frame, result, conditions, contrasts)
    print(result.summary())
    print("\nREGISTERED CONTRASTS\n", contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
