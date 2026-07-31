#!/usr/bin/env python3
"""Build schema-v2 claim preparation and fact verification datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from corvus.review_io import (
    authority_family_for_url,
    read_jsonl,
    require_unique_nonempty_rows,
    stable_id,
    write_jsonl,
)
from corvus.review_schema import (
    CLAIM_PREPARATION_TASK,
    FACT_VERIFICATION_TASK,
    SCHEMA_VERSION,
    ClaimPreparationExpected,
    ClaimPreparationInput,
    FactVerificationExpected,
    FactVerificationInput,
    braintrust_schemas,
    validate_review_row,
)


def _claim_preparation_row(
    *,
    legacy_row_id: str,
    source_kind: str,
    source_record_id: str,
    subject_hint: str | None,
    source_urls: list[str],
    candidate_context: dict[str, Any],
    instructions: list[str],
    metadata: dict[str, Any],
    tags: list[str],
) -> dict[str, Any]:
    row = {
        "id": stable_id("claim-preparation", legacy_row_id, SCHEMA_VERSION),
        "input": ClaimPreparationInput(
            task=CLAIM_PREPARATION_TASK,
            source_kind=source_kind,
            source_record_id=source_record_id,
            subject_hint=subject_hint,
            source_urls=source_urls,
            candidate_context=candidate_context,
            instructions=instructions,
        ).model_dump(mode="json"),
        "expected": ClaimPreparationExpected().model_dump(mode="json"),
        "metadata": {
            **metadata,
            "workflow": "claim_preparation",
            "review_stage": "claim_preparation",
            "schema_version": SCHEMA_VERSION,
            "verification_eligible": False,
            "contains_source_text": False,
        },
        "tags": sorted(set(tags) | {"claim-preparation", "schema-v2"}),
    }
    validate_review_row(row, stage="claim_preparation")
    return row


def edgar_preparation_row(record: dict[str, Any]) -> dict[str, Any]:
    legacy_row_id = stable_id(
        record["cik"],
        record["accession_number"],
        "corvus-review-v1",
    )
    return _claim_preparation_row(
        legacy_row_id=legacy_row_id,
        source_kind="sec_8k_item_5_02",
        source_record_id=record["accession_number"],
        subject_hint=record["entity_name"],
        source_urls=[record["filing_url"]],
        candidate_context={
            "cik": record["cik"],
            "entity_name": record["entity_name"],
            "filing_date": record["filing_date"],
            "detected_roles": record["strong_transition_role_signals"],
        },
        instructions=[
            "Open only the official SEC filing URL.",
            "Draft one atomic CEO or chair transition claim, or mark no_atomic_fact.",
            "State the person, role, company, and effective time explicitly.",
            "Use the event's effective time, not the filing or acceptance time.",
            "Record only derived facts; do not paste filing prose.",
        ],
        metadata={
            "dataset": "Corvus-QA",
            "original_workflow": "edgar_human_review",
            "review_priority": record["review_priority"],
            "document_sha256": record["document_sha256"],
            "compliance_source_ids": ["sec_edgar"],
            "source_type": "sec_8k_item_5_02",
        },
        tags=["corvus", "human-review", "edgar", record["review_priority"]],
    )


def news_preparation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for source_url in sorted(set(record["external_source_urls"])):
            host = (urlparse(source_url).hostname or "").casefold().removeprefix("www.")
            legacy_row_id = stable_id(
                "wikipedia-current-events",
                record["revision_id"],
                source_url,
                "corvus-review-v1",
            )
            rows.append(
                _claim_preparation_row(
                    legacy_row_id=legacy_row_id,
                    source_kind="wikipedia_current_events_citation",
                    source_record_id=stable_id(record["revision_id"], source_url),
                    subject_hint=None,
                    source_urls=[source_url, record["permanent_url"]],
                    candidate_context={
                        "event_date": record["event_date"],
                        "cited_domain": host,
                        "wikipedia_revision_id": record["revision_id"],
                        "wikipedia_revision_ts": record["revision_ts"],
                    },
                    instructions=[
                        "Open the cited publisher URL and assess it manually.",
                        "Draft one explicit, atomic, externally verifiable claim.",
                        "Include a canonical subject, predicate, object value, and time basis.",
                        "Mark no_atomic_fact if the link cannot support a suitable claim.",
                        "Record only derived facts; do not paste publisher prose.",
                    ],
                    metadata={
                        "dataset": "Corvus-QA",
                        "original_workflow": "news_human_review",
                        "wikipedia_revision_id": record["revision_id"],
                        "wikipedia_revision_ts": record["revision_ts"],
                        "authority_family": authority_family_for_url(source_url),
                        "compliance_source_ids": ["wikipedia_current_events"],
                        "license": record["license"],
                        "attribution": (
                            "Current-events citation metadata sourced from Wikipedia "
                            "under CC BY-SA 4.0."
                        ),
                        "source_type": "wikipedia_current_events_citation",
                    },
                    tags=["corvus", "human-review", "news"],
                )
            )
    return rows


def sports_verification_row(record: dict[str, Any]) -> dict[str, Any]:
    provider = record["compliance_source_id"]
    score = (
        f"{record['home_team']} {record['home_score']}–"
        f"{record['away_score']} {record['away_team']}"
    )
    claim_id = stable_id(
        provider,
        record["source_event_id"],
        "final_score",
        score,
    )
    row = {
        "id": stable_id("fact-verification", claim_id, SCHEMA_VERSION),
        "input": FactVerificationInput(
            task=FACT_VERIFICATION_TASK,
            claim={
                "claim_id": claim_id,
                "statement": (
                    f"The final score of {record['home_team']} vs "
                    f"{record['away_team']} was {record['home_score']}–"
                    f"{record['away_score']}."
                ),
                "subject_id": f"{provider}:{record['source_event_id']}",
                "subject_name": (
                    f"{record['home_team']} vs {record['away_team']}"
                ),
                "subject_type": "sports_event",
                "predicate": "final_score",
                "object_value": score,
                "previous_value": None,
                "asserted_effective_ts": None,
                "time_basis": (
                    "Scheduled event start is context only; reviewer must confirm "
                    "completion/effective time."
                ),
            },
            evidence=[
                {
                    "url": record["source_url"],
                    "source_role": "candidate",
                    "compliance_source_id": provider,
                    "authority_family": record["authority_family"],
                    "license": record["license"],
                    "attribution": record["attribution"],
                }
            ],
            context={
                "sport": record["sport"],
                "competition": record["competition"],
                "season": record["season"],
                "scheduled_start_ts": record["event_start_ts"],
            },
            instructions=[
                "Verify the exact teams, event identity, completion status, and score.",
                "Use an authoritative source independent of the candidate provider.",
                "Choose verified only when every material part of the claim is supported.",
                "Choose insufficient_evidence when the claim cannot be established.",
                "Choose contradicted when reliable evidence conflicts with the claim.",
                "Do not paste source prose, artwork, video, or other media.",
            ],
        ).model_dump(mode="json"),
        "expected": FactVerificationExpected().model_dump(mode="json"),
        "metadata": {
            "dataset": "Corvus-QA",
            "original_workflow": "sports_result_human_review",
            "workflow": "fact_verification",
            "review_stage": "fact_verification",
            "schema_version": SCHEMA_VERSION,
            "verification_eligible": True,
            "independent_confirmation_required": True,
            "observed_ts": record["observed_ts"],
            "resolver_id": record["resolver_id"],
            "authority_family": record["authority_family"],
            "compliance_source_ids": [provider],
            "license": record["license"],
            "attribution": record["attribution"],
            "contains_source_text": False,
            "source_type": record["source_type"],
        },
        "tags": sorted(
            {
                "corvus",
                "fact-verification",
                "human-review",
                provider,
                "schema-v2",
                "sports",
            }
        ),
    }
    validate_review_row(row, stage="fact_verification")
    return row


def _write_schema(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgar-review-queue", required=True, type=Path)
    parser.add_argument("--wikipedia-candidates", required=True, type=Path)
    parser.add_argument("--openligadb-results", required=True, type=Path)
    parser.add_argument("--thesportsdb-results", required=True, type=Path)
    parser.add_argument("--priority", default="high", choices=("high", "medium", "low"))
    parser.add_argument("--preparation-output", required=True, type=Path)
    parser.add_argument("--verification-output", required=True, type=Path)
    parser.add_argument("--preparation-schema-output", required=True, type=Path)
    parser.add_argument("--verification-schema-output", required=True, type=Path)
    args = parser.parse_args()

    preparation = [
        edgar_preparation_row(record)
        for record in read_jsonl(args.edgar_review_queue)
        if record["review_priority"] == args.priority
    ]
    preparation.extend(news_preparation_rows(read_jsonl(args.wikipedia_candidates)))
    verification = [
        sports_verification_row(record)
        for path in (args.openligadb_results, args.thesportsdb_results)
        for record in read_jsonl(path)
    ]
    require_unique_nonempty_rows(preparation, label="claim-preparation")
    require_unique_nonempty_rows(verification, label="fact-verification")

    write_jsonl(args.preparation_output, preparation)
    write_jsonl(args.verification_output, verification)
    _write_schema(
        args.preparation_schema_output,
        braintrust_schemas(ClaimPreparationInput, ClaimPreparationExpected),
    )
    _write_schema(
        args.verification_schema_output,
        braintrust_schemas(FactVerificationInput, FactVerificationExpected),
    )
    print(
        f"Wrote {len(preparation):,} claim-preparation rows and "
        f"{len(verification):,} fact-verification rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
