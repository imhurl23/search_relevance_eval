#!/usr/bin/env python3
"""Create a hash-only local review queue from downloaded Item 5.02 filings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


ROLE_PATTERNS = {
    "ceo": re.compile(r"\b(?:chief executive officer|CEO)\b", re.IGNORECASE),
    "chair": re.compile(
        r"\b(?:chair(?:person|man|woman)?|executive chair)\b", re.IGNORECASE
    ),
    "cfo": re.compile(r"\b(?:chief financial officer|CFO)\b", re.IGNORECASE),
}
TRANSITION_PATTERN = re.compile(
    r"\b(?:appointed|elected|named|resigned|departed|terminated|retired|"
    r"ceased to serve|will serve|effective)\b",
    re.IGNORECASE,
)
STRONG_TRANSITION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:appointed|elected|selected|promoted)\s+(?:to\s+serve\s+)?"
    r"(?:as|to\s+the\s+position\s+of)|"
    r"named\s+(?:as\s+|the\s+)?|"
    r"resigned\s+(?:from|as)|"
    r"ceased\s+(?:to\s+serve|serving)|"
    r"terminated\s+(?:from|as)|"
    r"retired\s+(?:from|as)"
    r")\b",
    re.IGNORECASE,
)


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def visible_text(body: bytes) -> str:
    decoded = body.decode("utf-8", errors="replace")
    parser = TextExtractor()
    parser.feed(decoded)
    return re.sub(r"\s+", " ", " ".join(parser.parts))


def nearby_transition_roles(
    text: str, transition_pattern=TRANSITION_PATTERN, window: int = 500
) -> list[str]:
    roles = []
    for role, pattern in ROLE_PATTERNS.items():
        for match in pattern.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if transition_pattern.search(text[start:end]):
                roles.append(role)
                break
    return sorted(roles)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    queue = []
    with args.ledger.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            path = Path(record["local_path"])
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"{path}: hash differs from download ledger")
            text = visible_text(body)
            roles = sorted(
                role for role, pattern in ROLE_PATTERNS.items() if pattern.search(text)
            )
            transition_signal = bool(TRANSITION_PATTERN.search(text))
            nearby_roles = nearby_transition_roles(text)
            strong_roles = nearby_transition_roles(text, STRONG_TRANSITION_PATTERN)
            priority = (
                "high"
                if any(role in strong_roles for role in ("ceo", "chair"))
                else "medium"
                if nearby_roles or (transition_signal and roles)
                else "low"
            )
            queue.append(
                {
                    "cik": record["cik"],
                    "entity_name": record["entity_name"],
                    "accession_number": record["accession_number"],
                    "filing_date": record["filing_date"],
                    "filing_url": record["filing_url"],
                    "document_sha256": digest,
                    "role_signals": roles,
                    "nearby_transition_role_signals": nearby_roles,
                    "strong_transition_role_signals": strong_roles,
                    "transition_signal": transition_signal,
                    "review_priority": priority,
                    "review_status": "pending_human_review",
                }
            )

    queue.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}[item["review_priority"]],
            item["filing_date"],
            item["cik"],
        )
    )
    with args.output.open("w") as output:
        for record in queue:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    counts = {
        priority: sum(item["review_priority"] == priority for item in queue)
        for priority in ("high", "medium", "low")
    }
    print(f"Wrote {len(queue):,} review records: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
