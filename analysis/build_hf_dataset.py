#!/usr/bin/env python3
"""Build the Hugging Face release for the LiveNewsBench search-arm matrix.

Reads the five completed exports in ``analysis/`` and writes a single
analysis-ready Parquet table plus a dataset card. The join key is
``(condition, root_span_id)``, matching ``analyze_variables.load_rows`` so the
published table and the paper analysis cannot drift.

Three content decisions are enforced here rather than left to the operator:

* Model answer text (``output.final_answer``) is dropped. Derived length
  columns are kept. Provider terms permit publishing outputs but restrict using
  them to train competing models, and the release does not need the text.
* ``metadata.stripped_markdown`` is dropped. It is Wikipedia Portal:Current
  events text under CC BY-SA 4.0, and including it would mix a share-alike
  license into an otherwise MIT-derived release.
* Braintrust span and transaction identifiers are dropped. ``dataset_row_id``
  is retained because the importer derives it from the upstream release, split,
  question, and answer, so it joins back to LiveNewsBench.

``metadata.articles`` never reaches these files: ``export_braintrust_results``
strips it during export. The audit at the end asserts that no You.com snippet
text, article payload, or model answer survives into the Parquet.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "hf_dataset"
STUDY = "livenewsbench-full-sonnet-v2"

MAIN = ROOT / f"{STUDY}.jsonl"
COSTS = ROOT / f"{STUDY}-with-costs.jsonl"
HIGHLIGHTS = ROOT / f"{STUDY}-highlight-mediators.jsonl"
GATING = ROOT / f"{STUDY}-gating-scores.jsonl"
RETRIEVAL = ROOT / f"{STUDY}-retrieval-mediators.jsonl"

MODEL_MAP = {
    "zai-org/GLM-5.2": "GLM-5.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "DeepSeek V4",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-sonnet-5": "Claude Sonnet 5",
}
# Read the arm from the condition label. `metadata.arm` mirrors the --arm flag
# default and stays "normalized" on the no-search rows, so it is not usable here.
ARM_MAP = {"none": "No search", "normalized": "Normalized", "wide": "Wide", "native": "Native"}

# metrics lifted from the with-costs export, keyed by (condition, root_span_id)
COST_METRICS = [
    "model_cost_usd", "search_cost_usd", "total_cost_usd", "search_share_of_cost",
    "agent_prompt_tokens", "agent_completion_tokens", "agent_total_tokens",
    "agent_cached_prompt_tokens", "agent_cache_hit_rate",
]
BEHAVIOUR_METRICS = [
    "answer_words", "answer_chars", "latency_s", "distinct_surfaced_domains",
    "native_search_actions", "native_open_page_actions", "native_find_actions",
    "native_tool_calls", "native_emitted_query_count", "refused_tool_calls",
    "n_search_errors", "n_operator_violations",
]

# exact column names that must never appear in the published table. Matched by
# equality, not substring: `snippet_sufficiency` is a 0/1 score, not a payload.
BANNED_COLUMNS = {
    "final_answer", "stripped_markdown", "articles", "snippet", "snippets",
    "root_span_id", "object_id", "_xact_id", "raw_content", "page_content",
    "answer", "output", "results", "documents",
}
URL_RE = re.compile(r"https?://")


def score_map(path: Path) -> dict[tuple[str, str], dict]:
    mapped: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            mapped[(record["condition"], record["root_span_id"])] = record.get("scores") or {}
    return mapped


def metric_map(path: Path) -> dict[tuple[str, str], dict]:
    mapped: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            mapped[(record["condition"], record["root_span_id"])] = record.get("metrics") or {}
    return mapped


def question_type(question: str) -> str:
    lowered = question.lower()
    quantitative = ("how many", "difference", "total", "combined", "sum", "percent", "ratio")
    return "Quantitative / composed" if any(t in lowered for t in quantitative) else "Other factual"


def row_id(task_key: str, condition: str, dataset_row_id: str) -> str:
    payload = f"{task_key}|{condition}|{dataset_row_id}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build() -> pd.DataFrame:
    highlight, gating, retrieval = score_map(HIGHLIGHTS), score_map(GATING), score_map(RETRIEVAL)
    costs = metric_map(COSTS)
    rows = []

    with MAIN.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            condition = record["condition"]
            key = (condition, record["root_span_id"])
            meta = record.get("metadata") or {}
            output = record.get("output") or {}
            base_metrics = record.get("metrics") or {}
            scores = record.get("scores") or {}
            cost_metrics = costs.get(key, {})

            mediators: dict = {}
            mediators.update(highlight.get(key, {}))
            mediators.update(gating.get(key, {}))
            mediators.update(retrieval.get(key, {}))

            condition_model, condition_arm = condition.split("::", 1)
            model_id = meta.get("agent_model") or condition_model
            search_mode = meta.get("search_mode") or ""
            arm = ARM_MAP[condition_arm]

            question = (record.get("input") or {}).get("question", "")
            event_date = pd.to_datetime(meta.get("date"), errors="coerce", utc=True)
            evaluated_at = pd.to_datetime(meta.get("as_of"), errors="coerce", utc=True)
            age = (
                (evaluated_at.normalize() - event_date.normalize()).days
                if pd.notna(event_date) and pd.notna(evaluated_at) else None
            )
            category = meta.get("benchmark_category") or meta.get("category") or "Uncategorized"
            if category.lower() == "law and crime":
                category = "Law and crime"

            dataset_row_id = str((record.get("origin") or {}).get("id"))
            task_key = str(meta.get("task_key"))
            ydc = meta.get("ydc_setup") or {}
            start, end = base_metrics.get("start"), base_metrics.get("end")

            row = {
                # identity and join keys
                "row_id": row_id(task_key, condition, dataset_row_id),
                "dataset_row_id": dataset_row_id,
                "task_key": task_key,
                # experimental design
                "condition": condition,
                "model": MODEL_MAP.get(model_id, model_id),
                "model_id": model_id,
                "model_vendor": meta.get("model_vendor"),
                "model_class": meta.get("model_class"),
                "arm": arm,
                "search_mode": search_mode,
                "search_provider": meta.get("search_provider"),
                "search_budget": meta.get("search_budget"),
                "result_count_requested": ydc.get("count"),
                "freshness_treatment": meta.get("freshness_treatment"),
                "trial_index": meta.get("trial_index"),
                # task properties, fixed before assignment
                "question": question,
                "expected": record.get("expected"),
                "question_type": question_type(question),
                "category": category,
                "event_date": event_date.date().isoformat() if pd.notna(event_date) else None,
                "evaluated_at": evaluated_at.isoformat() if pd.notna(evaluated_at) else None,
                "event_age_days": age,
                "temporal_stability": meta.get("temporal_stability"),
                "human_review_status": meta.get("human_review_status"),
                "livenewsbench_release": meta.get("livenewsbench_release"),
                "livenewsbench_split": meta.get("livenewsbench_split"),
                # registered and audit outcomes
                "gated_answer_match": scores.get("gated_answer_match"),
                "simpleqa_grade": scores.get("simpleqa_grade"),
                "qa_answer_match": mediators.get("qa_answer_match"),
                "dealbreaker_gate": mediators.get("dealbreaker_gate"),
                # retrieved-evidence mediators, normalized-harness arms only
                "snippet_sufficiency": mediators.get("snippet_sufficiency"),
                "evidence_precision": mediators.get("evidence_precision"),
                "token_discounted_gain": mediators.get("token_discounted_gain"),
                "temporal_grounding": mediators.get("temporal_grounding"),
                "domain_entropy": mediators.get("domain_entropy"),
                "compression_redundancy": mediators.get("compression_redundancy"),
                # behaviour
                "used_searches": output.get("used_searches"),
                "used_clicks": output.get("used_clicks"),
                # integrity flags
                "zero_search_row": bool(meta.get("zero_search_row")),
                "search_fully_failed": bool(meta.get("search_fully_failed")),
                "search_degraded": bool(meta.get("search_degraded")),
                "model_refused": bool(meta.get("model_refused")),
                "answer_truncated": bool(meta.get("answer_truncated")),
                "model_cost_confirmed": bool(meta.get("model_cost_confirmed")),
                # provenance
                "canary": meta.get("canary"),
                "study_id": meta.get("study_id"),
                "dataset_version": meta.get("dataset_version"),
                "source_repository": meta.get("source_repository"),
                "source_commit": meta.get("source_commit"),
            }
            for field in BEHAVIOUR_METRICS + COST_METRICS:
                row[field] = cost_metrics.get(field)
            if row["latency_s"] is None and isinstance(start, (int, float)) and isinstance(end, (int, float)):
                row["latency_s"] = end - start
            rows.append(row)

    return pd.DataFrame(rows)


def audit(frame: pd.DataFrame) -> None:
    """Fail the build rather than publish a table that leaks payload text."""
    problems = []

    for column in frame.columns:
        if column in BANNED_COLUMNS:
            problems.append(f"banned column present: {column}")

    # question and expected are the licensed LiveNewsBench text; every other
    # string column must be short, enumerated metadata with no embedded URLs.
    allowed_text = {"question", "expected", "canary", "source_repository"}
    for column in frame.columns:
        if frame[column].dtype != object or column in allowed_text:
            continue
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        if values.str.contains(URL_RE).any():
            problems.append(f"URL found in {column}")
        longest = values.str.len().max()
        if longest > 120:
            problems.append(f"{column} holds free text up to {longest} chars")

    expected_rows, expected_conditions = 18606, 14
    if len(frame) != expected_rows:
        problems.append(f"expected {expected_rows} rows, got {len(frame)}")
    if frame["condition"].nunique() != expected_conditions:
        problems.append(f"expected {expected_conditions} conditions, got {frame['condition'].nunique()}")

    # the matrix is balanced: 1,329 dataset rows in each of the 14 cells, and the
    # Native arm exists only for the two vendors that ship a native toolchain.
    cells = frame.groupby(["model", "arm"], observed=True).size()
    off_size = cells[cells != 1329]
    if not off_size.empty:
        problems.append(f"unbalanced model-arm cells: {off_size.to_dict()}")
    if sorted(frame["arm"].unique()) != ["Native", "No search", "Normalized", "Wide"]:
        problems.append(f"unexpected arm set: {sorted(frame['arm'].unique())}")
    native_models = sorted(frame.loc[frame["arm"] == "Native", "model"].unique())
    if native_models != ["Claude Sonnet 5", "GPT-5.6 Terra"]:
        problems.append(f"unexpected Native models: {native_models}")
    if frame["row_id"].duplicated().any():
        problems.append("row_id is not unique")
    if frame["canary"].nunique() != 1:
        problems.append("canary string is missing or inconsistent")
    for required in ("gated_answer_match", "simpleqa_grade", "total_cost_usd"):
        if frame[required].isna().any():
            problems.append(f"{required} has nulls")

    if problems:
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        raise SystemExit("audit failed; nothing written")
    print("  audit passed: no answer text, no snippet payload, no URLs outside provenance")


def main() -> int:
    frame = build()
    audit(frame)

    (OUT / "data").mkdir(parents=True, exist_ok=True)
    target = OUT / "data" / "results-00000-of-00001.parquet"
    frame.to_parquet(target, index=False, compression="zstd")

    print(f"rows={len(frame)}  columns={len(frame.columns)}  tasks={frame['task_key'].nunique()}")
    print(f"conditions={frame['condition'].nunique()}  bytes={target.stat().st_size:,}")
    print(f"wrote {target}")

    summary = (
        frame.groupby(["model", "arm"], observed=True)
        .agg(n=("row_id", "size"),
             judge=("simpleqa_grade", "mean"),
             gated=("gated_answer_match", "mean"),
             cost=("total_cost_usd", "mean"))
        .round(4)
    )
    print("\n" + summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
