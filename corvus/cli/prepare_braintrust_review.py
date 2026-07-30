#!/usr/bin/env python3
"""Prepare metadata-only Braintrust rows from the hash-only EDGAR review queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


FORBIDDEN_KEYS = {
    "local_path",
    "text",
    "excerpt",
    "evidence_context",
    "local_only_evidence_contexts",
}


def row_id(record: dict) -> str:
    value = f"{record['cik']}:{record['accession_number']}:corvus-review-v1"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assert_safe(value, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.casefold()
            if lowered in FORBIDDEN_KEYS or "excerpt" in lowered or "local_path" in lowered:
                raise ValueError(f"{path}.{key}: forbidden upload field")
            assert_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_safe(child, f"{path}[{index}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--priority", default="high", choices=("high", "medium", "low"))
    args = parser.parse_args()

    rows = []
    with args.review_queue.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["review_priority"] != args.priority:
                continue
            row = {
                "id": row_id(record),
                "input": {
                    "task": "Review this SEC Item 5.02 filing for a CEO or chair transition.",
                    "entity_name": record["entity_name"],
                    "cik": record["cik"],
                    "accession_number": record["accession_number"],
                    "filing_date": record["filing_date"],
                    "official_sec_url": record["filing_url"],
                    "detected_roles": record["strong_transition_role_signals"],
                    "instructions": [
                        "Open only the official SEC URL.",
                        "Mark not_relevant unless the filing establishes a CEO or chair transition.",
                        "Use the event's effective date, not filing or acceptance time.",
                        "Enter only derived facts; do not paste filing prose.",
                    ],
                },
                "expected": {
                    "decision": None,
                    "attribute": None,
                    "old_value": None,
                    "new_value": None,
                    "effective_ts": None,
                    "reviewer": None,
                    "reviewed_at": None,
                },
                "metadata": {
                    "dataset": "Corvus-QA",
                    "workflow": "edgar_human_review",
                    "review_priority": record["review_priority"],
                    "document_sha256": record["document_sha256"],
                    "compliance_source_ids": ["sec_edgar"],
                    "contains_source_text": False,
                    "source_type": "sec_8k_item_5_02",
                },
                "tags": ["corvus", "human-review", "edgar", record["review_priority"]],
            }
            assert_safe(row)
            rows.append(row)

    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(rows):,} metadata-only Braintrust review rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
