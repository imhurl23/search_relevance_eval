#!/usr/bin/env python3
"""Dependency-free paired analysis for exported Braintrust experiment rows.

The input is JSONL. Records may be canonical flat rows or Braintrust exports
with nested ``metadata``, ``metrics``, ``scores``, ``input``, and ``output``.
Only rows containing the requested score are analyzed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else None


def _question_key(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") or {}
    if metadata.get("task_key"):
        return str(metadata["task_key"])
    for key in ("task_key", "dataset_record_id", "dataset_row_id"):
        if record.get(key):
            return str(record[key])
    input_value = record.get("input")
    question = input_value.get("question") if isinstance(input_value, dict) else input_value
    if question:
        return hashlib.sha256(str(question).encode()).hexdigest()[:16]
    return None


def normalize_record(record: dict[str, Any], score_name: str) -> dict[str, Any] | None:
    metadata = record.get("metadata") or {}
    metrics = record.get("metrics") or {}
    scores = record.get("scores") or {}
    output = record.get("output") or {}

    score = _number(record.get("score"))
    if score is None:
        score = _number(scores.get(score_name))
    task_key = _question_key(record)
    condition = record.get("condition") or metadata.get("condition_id")
    if not condition:
        provider = metadata.get("provider")
        arm = metadata.get("arm")
        model = metadata.get("agent_model")
        if arm:
            condition = f"{model or 'model'}::{arm if arm == 'no_search' else f'{provider}-{arm}'}"
    if score is None or not task_key or not condition:
        return None

    category = (
        record.get("category")
        or metadata.get("benchmark_category")
        or metadata.get("category")
        or metadata.get("event_category")
        or metadata.get("data_source")
        or "uncategorized"
    )
    searches = _number(record.get("searches"))
    if searches is None:
        searches = _number(metrics.get("used_searches"))
    if searches is None and isinstance(output, dict):
        searches = _number(output.get("used_searches"))
    answer = output.get("final_answer", "") if isinstance(output, dict) else str(output)
    answer_words = _number(metrics.get("answer_words"))
    if answer_words is None:
        answer_words = float(len(answer.split()))

    # Deliberately do not treat search_cost_usd as total cost. Vals' analysis
    # shows model inference usually dominates; only consume an explicitly
    # aggregated total_cost_usd from a trace-level export.
    total_cost = _number(record.get("total_cost_usd"))
    if total_cost is None:
        total_cost = _number(metrics.get("total_cost_usd"))

    return {
        "task_key": task_key,
        "condition": str(condition),
        "study_id": metadata.get("study_id"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_version": metadata.get("dataset_version"),
        "category": str(category),
        "score": score,
        "searches": searches,
        "answer_words": answer_words,
        "total_cost_usd": total_cost,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _percentile(values: Iterable[float | None], probability: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    index = round((len(clean) - 1) * probability)
    return clean[index]


def aggregate_trials(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["task_key"], record["condition"])].append(record)
    aggregated = {}
    for key, trials in groups.items():
        aggregated[key] = {
            "task_key": key[0],
            "condition": key[1],
            "category": trials[0]["category"],
            "score": _mean(trial["score"] for trial in trials),
            "searches": _mean(trial["searches"] for trial in trials),
            "answer_words": _mean(trial["answer_words"] for trial in trials),
            "total_cost_usd": _mean(trial["total_cost_usd"] for trial in trials),
            "trials": len(trials),
        }
    return aggregated


def paired_effect(
    aggregated: dict[tuple[str, str], dict[str, Any]],
    baseline: str,
    condition: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    task_ids = sorted(
        task_key
        for task_key, candidate in aggregated
        if candidate == condition and (task_key, baseline) in aggregated
    )
    if not task_ids:
        return None
    differences = [
        aggregated[(task_key, condition)]["score"]
        - aggregated[(task_key, baseline)]["score"]
        for task_key in task_ids
    ]
    categories: dict[str, list[float]] = defaultdict(list)
    for task_key, difference in zip(task_ids, differences):
        categories[aggregated[(task_key, condition)]["category"]].append(difference)

    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot.append(statistics.fmean(sample))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    return {
        "condition": condition,
        "n_paired": len(task_ids),
        "mean_effect": statistics.fmean(differences),
        "ci_low": lower,
        "ci_high": upper,
        "category_balanced_effect": statistics.fmean(
            statistics.fmean(values) for values in categories.values()
        ),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def pareto_frontier(summaries: list[dict[str, Any]]) -> set[str]:
    candidates = [
        row for row in summaries
        if row["mean_cost_usd"] is not None and row["mean_score"] is not None
    ]
    frontier = set()
    for row in candidates:
        dominated = any(
            other["mean_cost_usd"] <= row["mean_cost_usd"]
            and other["mean_score"] >= row["mean_score"]
            and (
                other["mean_cost_usd"] < row["mean_cost_usd"]
                or other["mean_score"] > row["mean_score"]
            )
            for other in candidates
        )
        if not dominated:
            frontier.add(row["condition"])
    return frontier


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(
    records: list[dict[str, Any]],
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> str:
    aggregated = aggregate_trials(records)
    conditions = sorted({condition for _, condition in aggregated})
    summaries = []
    for condition in conditions:
        rows = [row for (_, candidate), row in aggregated.items() if candidate == condition]
        summaries.append({
            "condition": condition,
            "tasks": len(rows),
            "mean_score": _mean(row["score"] for row in rows),
            "mean_searches": _mean(row["searches"] for row in rows),
            "p95_searches": _percentile((row["searches"] for row in rows), 0.95),
            "mean_answer_words": _mean(row["answer_words"] for row in rows),
            "mean_cost_usd": _mean(row["total_cost_usd"] for row in rows),
        })
    frontier = pareto_frontier(summaries)

    lines = [
        "# Paired web-search experiment analysis",
        "",
        "## Condition summary",
        "",
        "| Condition | Tasks | Mean score | Mean searches | P95 searches | Mean answer words | Mean total cost | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['tasks']} | {_fmt(row['mean_score'])} | "
            f"{_fmt(row['mean_searches'], 2)} | {_fmt(row['p95_searches'], 2)} | "
            f"{_fmt(row['mean_answer_words'], 1)} | {_fmt(row['mean_cost_usd'], 4)} | "
            f"{'✓' if row['condition'] in frontier else ''} |"
        )

    lines.extend([
        "",
        f"## Paired effects versus `{baseline}`",
        "",
        "| Condition | Paired tasks | Mean effect | 95% task-bootstrap CI | Category-balanced effect | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        if condition == baseline:
            continue
        effect = paired_effect(aggregated, baseline, condition, bootstrap_samples, seed)
        if effect is None:
            lines.append(f"| {condition} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {condition} | {effect['n_paired']} | {_fmt(effect['mean_effect'])} | "
            f"[{_fmt(effect['ci_low'])}, {_fmt(effect['ci_high'])}] | "
            f"{_fmt(effect['category_balanced_effect'])} | "
            f"{effect['wins']}/{effect['ties']}/{effect['losses']} |"
        )

    if not frontier:
        lines.extend([
            "",
            "> Cost frontier omitted: supply trace-level `total_cost_usd` that "
            "includes both model inference and search fees. Search cost alone is "
            "not an acceptable substitute.",
        ])
    lines.extend([
        "",
        "The confidence interval resamples paired tasks, not individual trials. "
        "For publication, fit a mixed-effects model with condition as a fixed "
        "effect and task as a random intercept, matching the Vals methodology.",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Braintrust or canonical JSONL export")
    parser.add_argument("--score", default="qa_answer_match")
    parser.add_argument("--baseline", required=True, help="Exact condition_id")
    parser.add_argument("--study-id", help="Analyze only this experiment matrix")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    normalized = []
    with args.input.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            parsed = normalize_record(record, args.score)
            if parsed is not None:
                normalized.append(parsed)
    if args.study_id:
        normalized = [row for row in normalized if row["study_id"] == args.study_id]
    if not normalized:
        raise SystemExit(f"No rows contained score {args.score!r} and condition metadata")
    studies = {row["study_id"] for row in normalized if row["study_id"]}
    if len(studies) > 1:
        raise SystemExit(
            "Input contains multiple study_id values; pass --study-id to avoid "
            "combining experiment matrices"
        )
    dataset_versions = {
        (row["dataset_name"], row["dataset_version"])
        for row in normalized
        if row["dataset_name"] or row["dataset_version"]
    }
    if len(dataset_versions) > 1:
        raise SystemExit(
            "Input contains multiple dataset/version pairs; analyze each pinned "
            "snapshot separately"
        )
    print(render_report(normalized, args.baseline, args.bootstrap_samples, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
