#!/usr/bin/env python3
"""Export the completed experiments in a matrix checkpoint to JSONL.

Braintrust stores Eval scores on scorer child spans rather than root task spans.
This exporter joins requested scorer values and task-span operational metrics
back onto their roots and writes one record per fully scored root. Eval root
spans generally contain only start/end timing; row-level cost, token, tool, and
integrity metrics are logged on the child span named ``task``. Credentials are
loaded through ``run_eval`` so values in ``.env`` always override ambient
credentials.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_eval import (
    _btql_filter_eq,
    _fetch_experiment_events,
    _open_existing_experiment,
    condition_experiment_name,
    condition_label,
    load_runtime_env,
)


def _command_value(command: list[str], flag: str, default: str | None = None) -> str | None:
    return command[command.index(flag) + 1] if flag in command else default


def _experiment_spec(study_id: str, label: str, state: dict) -> dict:
    command = state["attempts"][-1]["command"]
    dataset_name = _command_value(command, "--dataset-name")
    vendor = _command_value(command, "--model-vendor")
    model = _command_value(command, "--agent-model")
    search_mode = _command_value(command, "--search-mode")
    arm = _command_value(command, "--arm", "normalized")
    if not all((dataset_name, vendor, model, search_mode, arm)):
        raise ValueError(f"incomplete command metadata for {label}")
    experiment_attempt = int(state.get("row_resume_experiment_attempt") or 1)
    condition = condition_label(search_mode, arm, vendor)
    return {
        "label": label,
        "name": condition_experiment_name(
            study_id, dataset_name, model, condition, experiment_attempt
        ),
    }


def _compact_record(
    root: dict,
    label: str,
    scores: dict,
    task_metrics: dict | None = None,
) -> dict:
    metadata = dict(root.get("metadata") or {})
    # Upstream article bodies and search trajectories dominate export size but
    # are not inputs to the paired analyzer. They can be re-queried for a
    # targeted qualitative audit without duplicating them across 14 arms.
    metadata.pop("articles", None)
    output = root.get("output") or {}
    compact_output = (
        {
            key: output.get(key)
            for key in ("final_answer", "used_searches", "used_clicks")
            if key in output
        }
        if isinstance(output, dict)
        else output
    )
    metrics = dict(root.get("metrics") or {})
    metrics.update(task_metrics or {})
    return {
        "id": root.get("id"),
        "root_span_id": root.get("root_span_id"),
        "origin": root.get("origin"),
        "input": root.get("input"),
        "expected": root.get("expected"),
        "output": compact_output,
        "metadata": metadata,
        "metrics": metrics,
        "condition": label,
        "scores": scores,
    }


def _fetch_condition(project: str, api_key: str, spec: dict, scores: list[str]) -> tuple[str, list[dict]]:
    experiment = _open_existing_experiment(project, spec["name"], api_key)
    if experiment is None:
        raise ValueError(f"experiment not found: {spec['name']}")
    roots = _fetch_experiment_events(experiment, _btql_filter_eq(["is_root"], True))
    scores_by_root: dict[str, dict[str, float | None]] = {}
    for score_name in scores:
        scorer_events = _fetch_experiment_events(
            experiment,
            _btql_filter_eq(["span_attributes", "name"], score_name),
        )
        for event in scorer_events:
            root_id = event.get("root_span_id")
            if root_id:
                scores_by_root.setdefault(root_id, {}).update(event.get("scores") or {})

    task_events = _fetch_experiment_events(
        experiment,
        _btql_filter_eq(["span_attributes", "name"], "task"),
    )
    task_metrics_by_root: dict[str, dict] = {}
    for event in task_events:
        root_id = event.get("root_span_id")
        if not root_id:
            continue
        metrics = event.get("metrics") or {}
        if root_id in task_metrics_by_root:
            raise ValueError(
                f"multiple task spans found for root {root_id} in {spec['name']}"
            )
        task_metrics_by_root[root_id] = metrics

    records = []
    for root in roots:
        origin = root.get("origin") or {}
        if origin.get("object_type") != "dataset" or not origin.get("id"):
            continue
        root_scores = scores_by_root.get(root.get("root_span_id"), {})
        if not all(score_name in root_scores for score_name in scores):
            continue
        records.append(_compact_record(
            root,
            spec["label"],
            root_scores,
            task_metrics_by_root.get(root.get("root_span_id")),
        ))
    return spec["label"], records


def clean_export(source: Path, destination: Path) -> int:
    """Remove non-dataset roots from an already fetched export atomically."""
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    kept = 0
    with source.open(encoding="utf-8") as rows, temporary.open(
        "w", encoding="utf-8"
    ) as cleaned:
        for line in rows:
            record = json.loads(line)
            origin = record.get("origin") or {}
            if origin.get("object_type") != "dataset" or not origin.get("id"):
                continue
            compact = _compact_record(
                record, record["condition"], record.get("scores") or {}
            )
            cleaned.write(json.dumps(compact, separators=(",", ":")) + "\n")
            kept += 1
    os.replace(temporary, destination)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project", default="search_evals")
    parser.add_argument("--score", action="append", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        help="Export only this exact checkpoint condition label (repeatable).",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    api_key, _ = load_runtime_env(args.env_file)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("status") != "completed":
        raise SystemExit(f"checkpoint is not completed: {checkpoint.get('status')}")
    study_id = checkpoint["fingerprint"]["study_id"]
    specs = [
        _experiment_spec(study_id, label, state)
        for label, state in checkpoint["conditions"].items()
        if not args.condition or label in args.condition
    ]
    missing_conditions = set(args.condition or ()) - {spec["label"] for spec in specs}
    if missing_conditions:
        raise SystemExit(
            "conditions not found in checkpoint: " + ", ".join(sorted(missing_conditions))
        )

    exported: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _fetch_condition, args.project, api_key, spec, args.score
            ): spec["label"]
            for spec in specs
        }
        for future in as_completed(futures):
            label, records = future.result()
            exported[label] = records
            print(f"Fetched {label}: {len(records)} scored roots", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for label in sorted(exported):
            for record in exported[label]:
                destination.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Wrote {sum(map(len, exported.values()))} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
