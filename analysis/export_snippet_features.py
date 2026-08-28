#!/usr/bin/env python3
"""Export per-result features from the You.com search trajectories.

``export_braintrust_results._compact_record`` keeps only ``final_answer``,
``used_searches``, and ``used_clicks`` from ``output``, so the retrieval
trajectory is absent from every file in ``analysis/``. This script re-reads the
harness-arm experiments and emits one row per returned search result.

Rights position, and why the text is off by default
---------------------------------------------------
The snippet bodies are verbatim text from Reuters, AP, BBC, NYT, the Guardian
and similar publishers, returned through the You.com API. Two separate permissions
are needed to republish them: You.com's commercial agreement must allow
redistribution of returned Output, and the publishers' copyright in the quoted
text must be cleared or the quotation must qualify as fair use. You.com's public
terms prohibit programmatically extracting Output and prohibit using Output to
train models, which is what an open dataset invites.

So the default export carries URL, domain, rank, publication date, lengths, a
content hash, and the gold-alias flag. That is enough to re-derive
``snippet_sufficiency``, ``evidence_precision``, and ``domain_entropy``, and to
prove two people rebuilt the same corpus. ``compression_redundancy`` gzips the
concatenated snippets and cannot be re-derived without the text; the published
per-row score covers that case.

Pass ``--include-text`` only once both permissions are confirmed in writing. The
flag records that choice in the output so a reviewer can see which corpus a file
came from.

Usage
-----
    python analysis/export_snippet_features.py \
        --checkpoint .eval-checkpoints/<study>.json \
        --output analysis/snippet-features.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from export_braintrust_results import _experiment_spec
from run_eval import (
    _btql_filter_eq,
    _fetch_experiment_events,
    _open_existing_experiment,
    load_runtime_env,
)
from scorers import _answer_aliases, _snippet_contains_gold, _surface

# Only the shared-harness arms return a snippet layer. Native arms declare
# no_snippet (Anthropic) or urls_only (OpenAI); no-search arms return nothing.
HARNESS_ARMS = ("normalized", "wide")


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _row_id(task_key: str, condition: str, dataset_row_id: str) -> str:
    """Must match build_hf_dataset.row_id so the two tables join."""
    return hashlib.sha256(f"{task_key}|{condition}|{dataset_row_id}".encode()).hexdigest()[:16]


def _domain(url: str) -> str:
    host = urlsplit(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def result_features(
    root: dict,
    condition: str,
    include_text: bool,
) -> list[dict]:
    output = root.get("output") or {}
    if not isinstance(output, dict):
        return []
    surface = _surface(output)
    if surface != "full":
        return []

    metadata = root.get("metadata") or {}
    expected = root.get("expected")
    aliases = _answer_aliases(expected, metadata)  # curated aliases when present
    task_key = str(metadata.get("task_key"))
    dataset_row_id = str((root.get("origin") or {}).get("id"))
    row_id = _row_id(task_key, condition, dataset_row_id)

    rows: list[dict] = []
    search_index = 0
    for step in output.get("trajectory") or []:
        if step.get("type") != "search":
            continue
        results = step.get("results") or []
        for result in results:
            title = result.get("title") or ""
            snippet = result.get("snippet") or ""
            url = result.get("url") or ""
            gold = result.get("oracle_snippet_gold")
            if gold is None:
                gold = _snippet_contains_gold(result, aliases)
            record = {
                # join keys back to the results table
                "row_id": row_id,
                "task_key": task_key,
                "dataset_row_id": dataset_row_id,
                "condition": condition,
                # position in the trajectory
                "search_index": search_index,
                "rank": result.get("rank"),
                "results_in_search": len(results),
                "query_words": len((step.get("query") or "").split()),
                "query_sha256": _sha(step.get("query") or ""),
                "search_tokens": step.get("tokens"),
                # the returned result, described rather than reproduced
                "url": url,
                "domain": _domain(url),
                "published_date": result.get("published_date"),
                "title_chars": len(title),
                "snippet_chars": len(snippet),
                "snippet_words": len(snippet.split()),
                "snippet_sha256": _sha(snippet),
                "contains_gold": bool(gold),
                "gold_from_oracle_label": result.get("oracle_snippet_gold") is not None,
                "text_included": include_text,
            }
            if include_text:
                record["title"] = title
                record["snippet"] = snippet
            rows.append(record)
        search_index += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project", default="search_evals")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include title and snippet bodies. Requires written confirmation "
             "that the You.com agreement permits redistribution and that "
             "publisher copyright in the quoted text is cleared.",
    )
    args = parser.parse_args()

    api_key, _ = load_runtime_env(args.env_file)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("status") != "completed":
        raise SystemExit(f"checkpoint is not completed: {checkpoint.get('status')}")
    study_id = checkpoint["fingerprint"]["study_id"]

    specs = [
        _experiment_spec(study_id, label, state)
        for label, state in checkpoint["conditions"].items()
        if label.rsplit("::", 1)[-1] in HARNESS_ARMS
    ]
    if not specs:
        raise SystemExit("no harness-arm conditions found in the checkpoint")

    if args.include_text:
        print("WARNING: including snippet bodies. Confirm the You.com agreement "
              "permits redistribution and that publisher copyright is cleared.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    domains: Counter[str] = Counter()
    written = 0
    with args.output.open("w", encoding="utf-8") as destination:
        for spec in specs:
            experiment = _open_existing_experiment(args.project, spec["name"], api_key)
            if experiment is None:
                raise SystemExit(f"experiment not found: {spec['name']}")
            roots = _fetch_experiment_events(experiment, _btql_filter_eq(["is_root"], True))
            rows = 0
            for root in roots:
                origin = root.get("origin") or {}
                if origin.get("object_type") != "dataset" or not origin.get("id"):
                    continue
                for record in result_features(root, spec["label"], args.include_text):
                    destination.write(json.dumps(record, separators=(",", ":")) + "\n")
                    domains[record["domain"]] += 1
                    rows += 1
            written += rows
            print(f"{spec['label']}: {rows} results", flush=True)

    print(f"\nWrote {written} result rows to {args.output}")
    print(f"Distinct domains: {len(domains)}")
    for domain, count in domains.most_common(12):
        print(f"  {count:>7}  {domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
