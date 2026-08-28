#!/usr/bin/env python3
"""Variable-level analysis and figures for the completed LiveNewsBench matrix."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


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
    diverging_cmap,
    header,
    make_figure,
    matrix_heatmap,
    percent_formatter,
    rate_heatmap,
    savefig,
    use_braintrust_theme,
)


MAIN = ANALYSIS / "livenewsbench-full-sonnet-v2.jsonl"
HIGHLIGHTS = ANALYSIS / "livenewsbench-full-sonnet-v2-highlight-mediators.jsonl"
GATING = ANALYSIS / "livenewsbench-full-sonnet-v2-gating-scores.jsonl"
RETRIEVAL = ANALYSIS / "livenewsbench-full-sonnet-v2-retrieval-mediators.jsonl"
SEED = 20260826
BOOTSTRAPS = 5000

MODEL_ORDER = ["GLM-5.2", "DeepSeek V4", "GPT-5.6 Terra", "Claude Sonnet 5"]
MODEL_MAP = {
    "zai-org/GLM-5.2": "GLM-5.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "DeepSeek V4",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-sonnet-5": "Claude Sonnet 5",
}
ARM_ORDER = ["No search", "Normalized", "Wide", "Native"]
MEDIATOR_LABELS = {
    "snippet_sufficiency": "Literal-gold coverage",
    "evidence_precision": "Evidence precision",
    "token_discounted_gain": "Token-discounted gain",
    "temporal_grounding": "Temporal grounding",
    "domain_entropy": "Domain diversity",
    "compression_redundancy": "Compression distinctness",
}


def _score_map(path: Path) -> dict[tuple[str, str], dict]:
    mapped = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            mapped[(record["condition"], record["root_span_id"])] = record.get("scores") or {}
    return mapped


def _question_type(question: str) -> str:
    q = question.lower()
    quantitative = (
        "how many", "how much", "by how many", "difference", "percentage",
        "percent", "combined value", "approximately how", "what was the total",
        "what were the", "how long", "elapsed", "additional years",
    )
    return "Quantitative / composed" if any(term in q for term in quantitative) else "Other factual"


def _parse_condition(condition: str) -> tuple[str, str]:
    model, arm = condition.split("::", 1)
    display = MODEL_MAP[model]
    arm_display = {
        "none": "No search",
        "normalized": "Normalized",
        "wide": "Wide",
        "native": "Native",
    }[arm]
    return display, arm_display


def load_rows() -> pd.DataFrame:
    highlight = _score_map(HIGHLIGHTS)
    gating = _score_map(GATING)
    retrieval = _score_map(RETRIEVAL)
    rows = []
    with MAIN.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            condition = record["condition"]
            root_id = record["root_span_id"]
            key = (condition, root_id)
            metadata = record.get("metadata") or {}
            output = record.get("output") or {}
            metrics = record.get("metrics") or {}
            scores = record.get("scores") or {}
            model, arm = _parse_condition(condition)
            question = (record.get("input") or {}).get("question", "")
            event_date = pd.to_datetime(metadata.get("date"), errors="coerce", utc=True)
            evaluated_at = pd.to_datetime(metadata.get("as_of"), errors="coerce", utc=True)
            event_age_days = (
                (evaluated_at.normalize() - event_date.normalize()).days
                if pd.notna(event_date) and pd.notna(evaluated_at) else None
            )
            category = metadata.get("benchmark_category") or metadata.get("category") or "Uncategorized"
            if category.lower() == "law and crime":
                category = "Law and crime"
            joined = {}
            joined.update(highlight.get(key, {}))
            joined.update(gating.get(key, {}))
            joined.update(retrieval.get(key, {}))
            start, end = metrics.get("start"), metrics.get("end")
            rows.append({
                "condition": condition,
                "model": model,
                "arm": arm,
                "task_key": str(metadata.get("task_key")),
                "dataset_row_id": str((record.get("origin") or {}).get("id")),
                "category": category,
                "question_type": _question_type(question),
                "question": question,
                "expected": record.get("expected"),
                "event_date": event_date.date().isoformat() if pd.notna(event_date) else None,
                "evaluated_at": evaluated_at.isoformat() if pd.notna(evaluated_at) else None,
                "event_age_days": event_age_days,
                "gated": scores.get("gated_answer_match"),
                "judge": scores.get("simpleqa_grade"),
                "qa_match": joined.get("qa_answer_match"),
                "dealbreaker_gate": joined.get("dealbreaker_gate"),
                "snippet_sufficiency": joined.get("snippet_sufficiency"),
                "evidence_precision": joined.get("evidence_precision"),
                "token_discounted_gain": joined.get("token_discounted_gain"),
                "temporal_grounding": joined.get("temporal_grounding"),
                "domain_entropy": joined.get("domain_entropy"),
                "compression_redundancy": joined.get("compression_redundancy"),
                "used_searches": output.get("used_searches"),
                "answer_words": len(str(output.get("final_answer") or "").split()),
                "latency_s": end - start if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None,
                "zero_search": bool(metadata.get("zero_search_row")),
                "search_failed": bool(metadata.get("search_fully_failed")),
                "search_degraded": bool(metadata.get("search_degraded")),
                "refused": bool(metadata.get("model_refused")),
                "truncated": bool(metadata.get("answer_truncated")),
            })
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame["arm"] = pd.Categorical(frame["arm"], ARM_ORDER, ordered=True)
    return frame


def answer_eligible(frame: pd.DataFrame, *, search_treated: bool = True) -> pd.DataFrame:
    keep = frame[~frame["refused"] & ~frame["truncated"]].copy()
    if search_treated:
        search = keep["arm"] != "No search"
        keep = keep[~search | (~keep["zero_search"] & ~keep["search_failed"])]
    return keep


def task_values(frame: pd.DataFrame, condition: str, value: str, *, search_treated: bool = True) -> pd.Series:
    subset = answer_eligible(frame[frame["condition"] == condition], search_treated=search_treated)
    return subset.dropna(subset=[value]).groupby("task_key", observed=True)[value].mean()


def paired_effect(frame: pd.DataFrame, a: str, b: str, value: str = "gated") -> dict:
    av = task_values(frame, a, value)
    bv = task_values(frame, b, value)
    keys = av.index.intersection(bv.index)
    diff = (av.loc[keys] - bv.loc[keys]).to_numpy(float)
    rng = np.random.default_rng(SEED + sum(map(ord, a + b + value)))
    samples = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        samples[start:start + size] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "a": a,
        "b": b,
        "metric": value,
        "n": len(diff),
        "effect": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
    }


def retrieval_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Difference-in-differences: which models gain more from normalized retrieval?"""
    gains = {}
    for model in MODEL_ORDER:
        normalized = answer_eligible(frame[(frame["model"] == model) & (frame["arm"] == "Normalized")])
        no_search = answer_eligible(frame[(frame["model"] == model) & (frame["arm"] == "No search")])
        a = normalized.dropna(subset=["gated"]).groupby("task_key", observed=True)["gated"].mean()
        b = no_search.dropna(subset=["gated"]).groupby("task_key", observed=True)["gated"].mean()
        keys = a.index.intersection(b.index)
        gains[model] = a.loc[keys] - b.loc[keys]

    pairs = [
        ("GLM-5.2", "GPT-5.6 Terra"),
        ("DeepSeek V4", "GPT-5.6 Terra"),
        ("GLM-5.2", "Claude Sonnet 5"),
        ("DeepSeek V4", "Claude Sonnet 5"),
        ("GLM-5.2", "DeepSeek V4"),
        ("GPT-5.6 Terra", "Claude Sonnet 5"),
    ]
    rows = []
    for first, second in pairs:
        keys = gains[first].index.intersection(gains[second].index)
        diff = (gains[first].loc[keys] - gains[second].loc[keys]).to_numpy(float)
        rng = np.random.default_rng(SEED + sum(map(ord, first + second)))
        boot = np.empty(BOOTSTRAPS)
        for start in range(0, BOOTSTRAPS, 250):
            size = min(250, BOOTSTRAPS - start)
            idx = rng.integers(0, len(diff), size=(size, len(diff)))
            boot[start:start + size] = diff[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        rows.append({
            "first_model": first, "second_model": second, "n": len(diff),
            "interaction": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
        })
    return pd.DataFrame(rows)


def cluster_mean_ci(frame: pd.DataFrame, value: str) -> tuple[float, float, float, int]:
    task = frame.dropna(subset=[value]).groupby("task_key", observed=True)[value].mean().to_numpy(float)
    rng = np.random.default_rng(SEED + sum(map(ord, value)))
    idx = rng.integers(0, len(task), size=(BOOTSTRAPS, len(task)))
    boot = task[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(task.mean()), float(lo), float(hi), len(task)


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(["model", "arm"], observed=True)
        .agg(
            n_rows=("dataset_row_id", "size"),
            n_tasks=("task_key", "nunique"),
            gated=("gated", "mean"),
            judge=("judge", "mean"),
            qa_match=("qa_match", "mean"),
            searches=("used_searches", "mean"),
            latency_mean_s=("latency_s", "mean"),
            latency_p95_s=("latency_s", lambda x: x.quantile(.95)),
            answer_words=("answer_words", "mean"),
            zero_search=("zero_search", "mean"),
            degraded=("search_degraded", "mean"),
            refused=("refused", "mean"),
            truncated=("truncated", "mean"),
        )
        .reset_index()
    )
    return result


def main_effects(frame: pd.DataFrame) -> pd.DataFrame:
    condition = {
        (model, arm): frame.loc[(frame["model"] == model) & (frame["arm"] == arm), "condition"].iloc[0]
        for model in MODEL_ORDER
        for arm in ARM_ORDER
        if ((frame["model"] == model) & (frame["arm"] == arm)).any()
    }
    rows = []
    for model in MODEL_ORDER:
        rows.append({"family": "Retrieval value", "model": model, **paired_effect(
            frame, condition[(model, "Normalized")], condition[(model, "No search")]
        )})
        rows.append({"family": "Result-count tier", "model": model, **paired_effect(
            frame, condition[(model, "Wide")], condition[(model, "Normalized")]
        )})
    for model in ("GPT-5.6 Terra", "Claude Sonnet 5"):
        rows.append({"family": "Native vs normalized", "model": model, **paired_effect(
            frame, condition[(model, "Native")], condition[(model, "Normalized")]
        )})
    return pd.DataFrame(rows)


def secondary_outcome_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired treatment effects for the audit outcome and operational outcomes."""
    condition = {
        (model, arm): frame.loc[(frame["model"] == model) & (frame["arm"] == arm), "condition"].iloc[0]
        for model in MODEL_ORDER for arm in ARM_ORDER
        if ((frame["model"] == model) & (frame["arm"] == arm)).any()
    }
    rows = []
    for model in MODEL_ORDER:
        for metric in ("judge", "latency_s", "answer_words"):
            rows.append({"comparison": "Normalized - no search", "model": model, **paired_effect(
                frame, condition[(model, "Normalized")], condition[(model, "No search")], metric
            )})
        for metric in ("judge", "used_searches", "latency_s", "answer_words"):
            rows.append({"comparison": "Wide - normalized", "model": model, **paired_effect(
                frame, condition[(model, "Wide")], condition[(model, "Normalized")], metric
            )})
    for model in ("GPT-5.6 Terra", "Claude Sonnet 5"):
        for metric in ("judge", "used_searches", "latency_s", "answer_words"):
            rows.append({"comparison": "Native - normalized", "model": model, **paired_effect(
                frame, condition[(model, "Native")], condition[(model, "Normalized")], metric
            )})
    return pd.DataFrame(rows)


def category_effects(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = answer_eligible(frame)
    rows = []
    for model in MODEL_ORDER:
        norm = eligible[(eligible["model"] == model) & (eligible["arm"] == "Normalized")]
        none = eligible[(eligible["model"] == model) & (eligible["arm"] == "No search")]
        for category in sorted(set(norm["category"]) & set(none["category"])):
            a = norm[norm["category"] == category].groupby("task_key", observed=True)["gated"].mean()
            b = none[none["category"] == category].groupby("task_key", observed=True)["gated"].mean()
            keys = a.index.intersection(b.index)
            if len(keys) >= 10:
                rows.append({"model": model, "category": category, "n": len(keys), "effect": float((a[keys] - b[keys]).mean())})
    return pd.DataFrame(rows)


def mediator_associations(frame: pd.DataFrame) -> pd.DataFrame:
    harness = answer_eligible(frame[frame["arm"].isin(["Normalized", "Wide"])]).copy()
    mediators = list(MEDIATOR_LABELS)
    rows = []
    for mediator in mediators:
        subset = harness.dropna(subset=["judge", mediator]).copy()
        # Patsy otherwise preserves unused pandas categorical levels after the
        # mediator-specific missingness filter, which can make the design singular.
        for field in ("model", "arm", "category"):
            subset[field + "_s"] = subset[field].astype("string")
        if mediator != "snippet_sufficiency":
            sd = subset[mediator].std()
            subset["mediator_x"] = (subset[mediator] - subset[mediator].mean()) / sd
            unit = "per 1 SD"
        else:
            subset["mediator_x"] = subset[mediator]
            unit = "present vs absent"
        fit = smf.glm(
            "judge ~ mediator_x + C(model_s) + C(arm_s) + C(category_s)",
            data=subset,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": subset["task_key"]})
        coefficient = fit.params["mediator_x"]
        lo, hi = fit.conf_int().loc["mediator_x"]
        rows.append({
            "mediator": MEDIATOR_LABELS[mediator],
            "field": mediator,
            "unit": unit,
            "n_rows": len(subset),
            "odds_ratio": math.exp(coefficient),
            "ci_low": math.exp(lo),
            "ci_high": math.exp(hi),
            "p_value": fit.pvalues["mediator_x"],
        })
    result = pd.DataFrame(rows).sort_values("p_value")
    adjusted = []
    running = 0.0
    total = len(result)
    for rank, p in enumerate(result["p_value"]):
        running = max(running, min(1.0, (total - rank) * p))
        adjusted.append(running)
    result["holm_p"] = adjusted
    return result


def mediator_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    harness = answer_eligible(frame[frame["arm"].isin(["Normalized", "Wide"])])
    for model in MODEL_ORDER:
        for field, label in MEDIATOR_LABELS.items():
            a = harness[(harness["model"] == model) & (harness["arm"] == "Wide")].dropna(subset=[field]).groupby("task_key", observed=True)[field].mean()
            b = harness[(harness["model"] == model) & (harness["arm"] == "Normalized")].dropna(subset=[field]).groupby("task_key", observed=True)[field].mean()
            keys = a.index.intersection(b.index)
            rows.append({"model": model, "mediator": label, "n": len(keys), "delta": float((a[keys] - b[keys]).mean())})
    return pd.DataFrame(rows)


def question_type_rates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = answer_eligible(frame[frame["arm"] == "Normalized"])
    rows = []
    for model in MODEL_ORDER:
        for question_type in ("Quantitative / composed", "Other factual"):
            subset = normalized[(normalized["model"] == model) & (normalized["question_type"] == question_type)]
            rate, lo, hi, n = cluster_mean_ci(subset, "judge")
            rows.append({"model": model, "question_type": question_type, "rate": rate, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


def scorer_rates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = answer_eligible(frame[frame["arm"] == "Normalized"])
    rows = []
    for model in MODEL_ORDER:
        subset = normalized[normalized["model"] == model]
        for field, label in (("qa_match", "Deterministic match"), ("judge", "Semantic judge")):
            rate, lo, hi, n = cluster_mean_ci(subset, field)
            rows.append({"model": model, "scorer": label, "rate": rate, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


def plot_effects(effects: pd.DataFrame) -> None:
    data = effects[effects["family"] == "Retrieval value"].copy().sort_values("effect")
    fig, ax = make_figure(width=12, height=6.8, left=.21, right=.94, top=.78)
    y = np.arange(len(data))
    xerr = np.vstack([data["effect"] - data["ci_low"], data["ci_high"] - data["effect"]])
    ax.errorbar(data["effect"], y, xerr=xerr, fmt="o", ms=12, color=BRAND["indigo"],
                ecolor=BRAND["purple"], elinewidth=3, capsize=7, capthick=2.5, zorder=4)
    ax.axvline(0, color=NEUTRAL["subtle"], ls=(0, (4, 4)), lw=1.5)
    ax.set_yticks(y, data["model"])
    ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.set_xlabel("Paired change in gated answer match")
    for yi, (_, row) in enumerate(data.iterrows()):
        ax.text(row["ci_high"] + .012, yi, f"+{row['effect']:.1%}", va="center", fontweight="bold")
    ax.set_xlim(-.03, max(data["ci_high"]) + .08)
    header(ax, "Retrieval helps every model—unequally",
           "Normalized harness minus the same model with no search; 95% task-bootstrap intervals",
           "Primary dependent variable: gated_answer_match")
    clean_axes(ax, despine_left=True, grid_axis="x")
    savefig(fig, PLOTS / "01_retrieval_effects_gated.png")
    plt.close(fig)


def plot_quality_heatmap(summary: pd.DataFrame) -> None:
    table = summary.pivot(index="model", columns="arm", values="gated").reindex(index=MODEL_ORDER, columns=ARM_ORDER)
    fig, ax = make_figure(width=11, height=6.4, left=.18, right=.93, top=.76)
    rate_heatmap(ax, table, annot=True, vmin=0, vmax=.5, cbar_label="gated answer match")
    header(ax, "The treatment surface reshapes model rankings",
           "Descriptive task-row means; native search is only available for frontier models",
           "Cells are rates, not paired effects")
    savefig(fig, PLOTS / "02_condition_quality_heatmap.png")
    plt.close(fig)


def plot_highlight_association(frame: pd.DataFrame) -> None:
    normalized = answer_eligible(frame[(frame["arm"] == "Normalized") & frame["snippet_sufficiency"].notna()])
    rows = []
    for model in MODEL_ORDER:
        for visible, label in ((0.0, "Gold not detected"), (1.0, "Literal gold visible")):
            subset = normalized[(normalized["model"] == model) & (normalized["snippet_sufficiency"] == visible)]
            rate, lo, hi, n = cluster_mean_ci(subset, "gated")
            rows.append({"model": model, "visibility": label, "rate": rate, "lo": lo, "hi": hi, "n": n})
    data = pd.DataFrame(rows)
    fig, ax = make_figure(width=12, height=7.6, left=.10, right=.96, top=.68)
    x = np.arange(len(MODEL_ORDER)); width=.34
    colors = [BRAND["pink"], BRAND["indigo"]]
    for i, label in enumerate(("Gold not detected", "Literal gold visible")):
        sub = data[data["visibility"] == label].set_index("model").loc[MODEL_ORDER]
        pos = x + (i-.5)*width
        err = np.vstack([sub["rate"]-sub["lo"], sub["hi"]-sub["rate"]])
        ax.bar(pos, sub["rate"], width, color=colors[i], edgecolor="white", linewidth=1.4,
               yerr=err, capsize=5, error_kw={"ecolor": BRAND["purple"], "elinewidth":2, "capthick":2}, label=label, zorder=3)
        for px, rate, hi in zip(pos, sub["rate"], sub["hi"]):
            ax.text(px, hi+.025, f"{rate:.0%}", ha="center", fontweight="bold", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none"))
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.set_ylabel("Gated answer match")
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1,1.005))
    header(ax, "Answer-bearing surface text is strongly associated with success",
           "Normalized harness only; literal alias detection is a conservative mediator, not a highlights treatment effect",
           "95% task-bootstrap intervals")
    clean_axes(ax, grid_axis="y")
    savefig(fig, PLOTS / "03_highlight_sufficiency_association.png")
    plt.close(fig)


def plot_mediator_deltas(deltas: pd.DataFrame) -> None:
    table = deltas.pivot(index="mediator", columns="model", values="delta").reindex(index=list(MEDIATOR_LABELS.values()), columns=MODEL_ORDER)
    limit = max(abs(table.min().min()), abs(table.max().max()))
    fig, ax = make_figure(width=12, height=7.4, left=.23, right=.94, top=.77)
    matrix_heatmap(ax, table, annot=True, fmt=lambda v:f"{v:+.1%}", cmap=diverging_cmap(),
                   vmin=-limit, vmax=limit, cbar_label="wide − normalized")
    header(ax, "Wide retrieval trades signal density for slightly broader coverage",
           "Paired mediator changes; all scores are on a 0–1 scale",
           "Exploratory: these are post-treatment variables")
    savefig(fig, PLOTS / "04_wide_mediator_deltas.png")
    plt.close(fig)


def plot_question_types(rates: pd.DataFrame) -> None:
    fig, ax = make_figure(width=12, height=7.6, left=.10, right=.96, top=.68)
    x=np.arange(len(MODEL_ORDER));width=.34
    labels=["Quantitative / composed","Other factual"];colors=[BRAND["pink"],BRAND["indigo"]]
    for i,label in enumerate(labels):
        sub=rates[rates["question_type"]==label].set_index("model").loc[MODEL_ORDER]
        pos=x+(i-.5)*width;err=np.vstack([sub["rate"]-sub["ci_low"],sub["ci_high"]-sub["rate"]])
        ax.bar(pos,sub["rate"],width,color=colors[i],edgecolor="white",linewidth=1.4,yerr=err,capsize=5,
               error_kw={"ecolor":BRAND["purple"],"elinewidth":2,"capthick":2},label=label,zorder=3)
        for px,rate,hi in zip(pos,sub["rate"],sub["ci_high"]):
            ax.text(px,hi+.018,f"{rate:.0%}",ha="center",fontweight="bold",fontsize=12,
                    bbox=dict(boxstyle="round,pad=.12",fc="white",ec="none"))
    ax.set_xticks(x,MODEL_ORDER);ax.set_ylim(0,.99);ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.set_ylabel("Semantic-judge accuracy")
    ax.legend(frameon=False,ncol=2,loc="lower right",bbox_to_anchor=(1,1.005))
    header(ax,"Compositional questions remain the hard part",
           "Normalized harness only; heuristic question-type split with 95% task-bootstrap intervals",
           "Dependent variable: simpleqa_grade")
    clean_axes(ax,grid_axis="y")
    savefig(fig,PLOTS/"05_question_type_judge_accuracy.png");plt.close(fig)


def plot_scorer_gap(rates: pd.DataFrame) -> None:
    fig,ax=make_figure(width=12,height=7.6,left=.10,right=.96,top=.68)
    x=np.arange(len(MODEL_ORDER));width=.34
    labels=["Deterministic match","Semantic judge"];colors=[BRAND["pink"],BRAND["indigo"]]
    for i,label in enumerate(labels):
        sub=rates[rates["scorer"]==label].set_index("model").loc[MODEL_ORDER]
        pos=x+(i-.5)*width;err=np.vstack([sub["rate"]-sub["ci_low"],sub["ci_high"]-sub["rate"]])
        ax.bar(pos,sub["rate"],width,color=colors[i],edgecolor="white",linewidth=1.4,yerr=err,capsize=5,
               error_kw={"ecolor":BRAND["purple"],"elinewidth":2,"capthick":2},label=label,zorder=3)
        for px,rate,hi in zip(pos,sub["rate"],sub["ci_high"]):
            ax.text(px,hi+.018,f"{rate:.0%}",ha="center",fontweight="bold",fontsize=12,
                    bbox=dict(boxstyle="round,pad=.12",fc="white",ec="none"))
    ax.set_xticks(x,MODEL_ORDER);ax.set_ylim(0,1.0);ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.set_ylabel("Answer accuracy")
    ax.legend(frameon=False,ncol=2,loc="lower right",bbox_to_anchor=(1,1.005))
    header(ax,"Measurement choice changes the apparent absolute performance",
           "Normalized harness only; raw deterministic match versus the single semantic judge",
           "95% task-bootstrap intervals; neither scorer should be silently substituted for the other")
    clean_axes(ax,grid_axis="y")
    savefig(fig,PLOTS/"06_scorer_disagreement.png");plt.close(fig)


def write_outputs(frame: pd.DataFrame, summary: pd.DataFrame, effects: pd.DataFrame,
                  interactions: pd.DataFrame,
                  secondary: pd.DataFrame,
                  categories: pd.DataFrame, associations: pd.DataFrame,
                  deltas: pd.DataFrame, question_rates: pd.DataFrame,
                  scorer: pd.DataFrame) -> None:
    task_columns = [
        "condition","model","arm","task_key","dataset_row_id","category","question_type","question","expected",
        "event_date","evaluated_at","event_age_days","gated","judge","qa_match",
        "snippet_sufficiency","evidence_precision","token_discounted_gain","temporal_grounding",
        "domain_entropy","compression_redundancy","used_searches","latency_s","answer_words",
        "zero_search","search_failed","search_degraded","refused","truncated",
    ]
    frame[task_columns].to_csv(ANALYSIS/"variable_row_level.csv",index=False)
    summary.to_csv(ANALYSIS/"variable_condition_summary.csv",index=False)
    effects.to_csv(ANALYSIS/"variable_main_effects.csv",index=False)
    interactions.to_csv(ANALYSIS/"variable_retrieval_interactions.csv",index=False)
    secondary.to_csv(ANALYSIS/"variable_secondary_outcome_effects.csv",index=False)
    categories.to_csv(ANALYSIS/"variable_category_effects.csv",index=False)
    associations.to_csv(ANALYSIS/"variable_mediator_associations.csv",index=False)
    deltas.to_csv(ANALYSIS/"variable_mediator_deltas.csv",index=False)
    question_rates.to_csv(ANALYSIS/"variable_question_type_rates.csv",index=False)
    scorer.to_csv(ANALYSIS/"variable_scorer_rates.csv",index=False)
    payload={
        "n_rows":len(frame),"n_tasks":int(frame["task_key"].nunique()),
        "main_effects":effects.to_dict(orient="records"),
        "retrieval_interactions":interactions.to_dict(orient="records"),
        "secondary_outcome_effects":secondary.to_dict(orient="records"),
        "mediator_associations":associations.to_dict(orient="records"),
    }
    (ANALYSIS/"variable_analysis_summary.json").write_text(json.dumps(payload,indent=2,default=str)+"\n",encoding="utf-8")


def main() -> int:
    use_braintrust_theme()
    PLOTS.mkdir(parents=True,exist_ok=True)
    frame=load_rows()
    summary=condition_summary(frame)
    effects=main_effects(frame)
    interactions=retrieval_interactions(frame)
    secondary=secondary_outcome_effects(frame)
    categories=category_effects(frame)
    associations=mediator_associations(frame)
    deltas=mediator_deltas(frame)
    q_rates=question_type_rates(frame)
    scorer=scorer_rates(frame)
    write_outputs(frame,summary,effects,interactions,secondary,categories,associations,deltas,q_rates,scorer)
    plot_effects(effects)
    plot_quality_heatmap(summary)
    plot_highlight_association(frame)
    plot_mediator_deltas(deltas)
    plot_question_types(q_rates)
    plot_scorer_gap(scorer)
    print(summary.to_string(index=False))
    print("\nMAIN EFFECTS\n",effects.to_string(index=False))
    print("\nRETRIEVAL INTERACTIONS\n",interactions.to_string(index=False))
    print("\nMEDIATOR ASSOCIATIONS\n",associations.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
