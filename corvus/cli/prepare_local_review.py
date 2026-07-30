#!/usr/bin/env python3
"""Prepare local-only evidence contexts for human Item 5.02 adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from corvus.review import (
    ROLE_PATTERNS,
    STRONG_TRANSITION_PATTERN,
    visible_text,
)


def evidence_contexts(text: str, limit: int = 3) -> list[dict[str, str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches = []
    for index, sentence in enumerate(sentences):
        if not STRONG_TRANSITION_PATTERN.search(sentence):
            continue
        if not any(
            ROLE_PATTERNS[role].search(sentence) for role in ("ceo", "chair")
        ):
            continue
        start = max(0, index - 1)
        end = min(len(sentences), index + 2)
        context = " ".join(sentences[start:end]).strip()[:2500]
        matches.append(
            {
                "text": context,
                "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            }
        )
        if len(matches) == limit:
            break
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-ledger", required=True, type=Path)
    parser.add_argument("--review-queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--priority", default="high", choices=("high", "medium", "low"))
    args = parser.parse_args()

    paths = {}
    with args.download_ledger.open() as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                paths[(record["cik"], record["accession_number"])] = Path(
                    record["local_path"]
                )

    packet = []
    with args.review_queue.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["review_priority"] != args.priority:
                continue
            key = (record["cik"], record["accession_number"])
            path = paths.get(key)
            if path is None:
                raise ValueError(f"missing downloaded filing for {key}")
            contexts = evidence_contexts(visible_text(path.read_bytes()))
            packet.append(
                {
                    "cik": record["cik"],
                    "entity_name": record["entity_name"],
                    "accession_number": record["accession_number"],
                    "filing_date": record["filing_date"],
                    "filing_url": record["filing_url"],
                    "document_sha256": record["document_sha256"],
                    "local_only_evidence_contexts": contexts,
                    "adjudication": {
                        "decision": None,
                        "attribute": None,
                        "old_value": None,
                        "new_value": None,
                        "effective_ts": None,
                        "reviewer": None,
                        "reviewed_at": None,
                        "selected_evidence_sha256": None,
                    },
                    "distribution_warning": (
                        "Local review artifact: do not publish evidence text."
                    ),
                }
            )

    with args.output.open("w") as output:
        for record in packet:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with_context = sum(bool(record["local_only_evidence_contexts"]) for record in packet)
    print(
        f"Wrote {len(packet):,} local review packets; "
        f"{with_context:,} contain strong-transition evidence contexts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
