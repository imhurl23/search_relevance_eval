#!/usr/bin/env python3
"""Build a fact-verification queue from Section 16-corroborated transitions.

These candidates skip claim preparation. A Form 3 states the office and the
event date as structured fields, so the claim is already atomic and the
reviewer's job is confirmation rather than extraction: does the issuer's own
Item 5.02 filing describe this person taking this office on this date?

The pairing window is deliberately generous, so a Form 3 can sit beside an
unrelated Item 5.02 — catching that is the first thing the instructions ask for.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from corvus.review_io import read_jsonl, require_unique_nonempty_rows, stable_id, write_jsonl
from corvus.review_schema import (
    FACT_VERIFICATION_TASK,
    SCHEMA_VERSION,
    FactVerificationExpected,
    FactVerificationInput,
    braintrust_schemas,
    validate_review_row,
)
from corvus.section16 import Section16Filing


PREDICATE_LABEL = {"ceo_of": "Chief Executive Officer", "chairperson_of": "Chairperson"}

# Blank-check issuers file Item 5.02 at formation, so the "transition" is an
# initial appointment at an entity with no search footprint and no plausible
# user question. Flagged as a review hint only — the reviewer decides, because
# this is a name heuristic and names are not a reliable classifier.
BLANK_CHECK_HINT = re.compile(
    r"\b(?:acquisition|blank[ -]check)\b|\bcapital\s+corp\b|"
    r"\bcorp\.?\s+(?:i{1,3}|iv|v)\b|\bcorp\s+(?:i{1,3}|iv|v)\b",
    re.IGNORECASE,
)


def section16_verification_row(
    filing: Section16Filing,
    *,
    entity_name: str,
    item_502_urls: list[str],
    item_502_accessions: list[str],
    review_priority: str | None,
) -> dict[str, Any]:
    attribute = filing.role_attribute
    if attribute is None:
        raise ValueError(
            f"{filing.accession_number}: no Corvus attribute for "
            f"officerTitle {filing.officer_title!r}"
        )
    office = PREDICATE_LABEL[attribute]
    person = filing.natural_owner_name
    effective = filing.effective_ts
    claim_id = stable_id(
        f"CIK{filing.issuer_cik}", attribute, f"CIK{filing.owner_cik}",
        filing.period_of_report.isoformat(),
    )
    possible_blank_check = bool(BLANK_CHECK_HINT.search(entity_name))

    evidence: list[dict[str, Any]] = [
        {
            "url": url,
            "source_role": "candidate",
            "compliance_source_id": "sec_edgar",
            "authority_family": "sec",
            "attester_id": f"CIK{filing.issuer_cik}",
            "attester_role": "issuer",
        }
        for url in item_502_urls
    ]
    evidence.append(
        {
            "url": filing.filing_url,
            "source_role": "independent_verification",
            "compliance_source_id": "sec_edgar",
            "authority_family": "sec",
            "attester_id": f"CIK{filing.owner_cik}",
            "attester_role": "reporting_owner",
        }
    )

    row = {
        "id": stable_id("fact-verification", claim_id, SCHEMA_VERSION),
        "input": FactVerificationInput(
            task=FACT_VERIFICATION_TASK,
            claim={
                "claim_id": claim_id,
                "statement": (
                    f"{person} became {office} of {entity_name}, "
                    f"effective {filing.period_of_report.isoformat()}."
                ),
                "subject_id": f"CIK{filing.issuer_cik}",
                "subject_name": entity_name,
                "subject_type": "company",
                "predicate": attribute,
                "object_value": person,
                # Section 16 does not attest a predecessor; the reviewer takes
                # it from the 8-K.
                "previous_value": None,
                "asserted_effective_ts": effective,
                "time_basis": (
                    "Section 16 'Date of Event Requiring Statement' "
                    f"({filing.period_of_report.isoformat()}), day precision. "
                    "Not the filing date and not the 8-K acceptance time."
                ),
            },
            evidence=evidence,
            context={
                "issuer_cik": filing.issuer_cik,
                "reporting_owner_cik": filing.owner_cik,
                "section16_form_type": filing.form_type,
                "section16_officer_title": filing.officer_title,
                "section16_is_officer": filing.is_officer,
                "section16_is_director": filing.is_director,
                "edgar_owner_name_as_filed": filing.owner_name,
                "item_502_accessions": item_502_accessions,
                "possible_blank_check_issuer": possible_blank_check,
            },
            instructions=[
                "Confirm the Item 5.02 filing describes this person taking this "
                "office. The Form 3 was paired by date, so it may belong to a "
                "different appointment at the same issuer.",
                "Confirm the effective date. Section 16 gives day precision; if "
                "the 8-K states a different date, choose contradicted and put "
                "the 8-K's date in correction.",
                "Record the predecessor in correction if the 8-K names one. "
                "Section 16 does not attest the previous officer.",
                "Reject titles that are interim, acting, co-held, or scoped to a "
                "subsidiary or division; the question admits one answer.",
                "Confirm the name spelling for the published answer. EDGAR files "
                "names surname-first, so the reordering may be wrong for a "
                "compound surname.",
                "If possible_blank_check_issuer is set, judge whether a user "
                "would ever ask this question; mark insufficient_evidence for a "
                "shell with no operating history.",
                "Record only derived facts; do not paste filing prose.",
            ],
        ).model_dump(mode="json"),
        "expected": FactVerificationExpected().model_dump(mode="json"),
        "metadata": {
            "dataset": "Corvus-QA",
            "original_workflow": "section16_corroborated_transition",
            "workflow": "fact_verification",
            "review_stage": "fact_verification",
            "schema_version": SCHEMA_VERSION,
            "verification_eligible": True,
            "independent_confirmation_required": False,
            "corroboration": "dual_attester_single_publisher",
            "attester_ids": sorted(
                {f"CIK{filing.issuer_cik}", f"CIK{filing.owner_cik}"}
            ),
            "attester_roles": ["issuer", "reporting_owner"],
            "authority_families": ["sec"],
            "resolver_ids": ["edgar-8k", "edgar-section16"],
            "attribute": attribute,
            "effective_ts": effective.isoformat(),
            "effective_ts_precision": "day",
            "old_value_attested": False,
            "review_priority": review_priority,
            "possible_blank_check_issuer": possible_blank_check,
            "section16_document_sha256": filing.document_sha256,
            "compliance_source_ids": ["sec_edgar"],
            "contains_source_text": False,
            "source_type": f"sec_form_{filing.form_type.replace('/', '_').lower()}",
        },
        "tags": sorted(
            {
                "corvus",
                "fact-verification",
                "human-review",
                "edgar",
                "section16",
                "schema-v2",
                attribute.replace("_", "-"),
            }
            | ({"possible-blank-check"} if possible_blank_check else set())
        ),
    }
    validate_review_row(row, stage="fact_verification")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section16-filings", required=True, type=Path)
    parser.add_argument("--section16-refs", required=True, type=Path)
    parser.add_argument("--item-502-candidates", required=True, type=Path)
    parser.add_argument(
        "--review-queue",
        type=Path,
        help="Optional triage queue, used only to carry review_priority through.",
    )
    parser.add_argument(
        "--exclude-blank-check",
        action="store_true",
        help="Drop issuers matching the blank-check name heuristic instead of tagging them.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schema-output", required=True, type=Path)
    args = parser.parse_args()

    filings = [
        Section16Filing.model_validate(record)
        for record in read_jsonl(args.section16_filings)
    ]
    refs = {record["accession_number"]: record for record in read_jsonl(args.section16_refs)}
    candidates = {record["accession_number"]: record for record in read_jsonl(args.item_502_candidates)}
    priority = {}
    if args.review_queue:
        rank = {"high": 2, "medium": 1, "low": 0}
        for record in read_jsonl(args.review_queue):
            current = priority.get(record["cik"])
            if current is None or rank[record["review_priority"]] > rank[current]:
                priority[record["cik"]] = record["review_priority"]

    rows, skipped = [], []
    for filing in filings:
        if filing.role_attribute is None:
            continue
        ref = refs.get(filing.accession_number)
        if ref is None:
            skipped.append({"accession_number": filing.accession_number,
                            "reason": "no_pairing_reference"})
            continue
        accessions = ref.get("paired_item_502_accessions") or []
        paired = [candidates[a] for a in accessions if a in candidates]
        if not paired:
            skipped.append({"accession_number": filing.accession_number,
                            "reason": "paired_item_502_candidate_missing"})
            continue
        entity_name = paired[0]["entity_name"]
        if args.exclude_blank_check and BLANK_CHECK_HINT.search(entity_name):
            skipped.append({"accession_number": filing.accession_number,
                            "reason": "blank_check_heuristic",
                            "entity_name": entity_name})
            continue
        rows.append(
            section16_verification_row(
                filing,
                entity_name=entity_name,
                item_502_urls=sorted({record["filing_url"] for record in paired}),
                item_502_accessions=sorted(accessions),
                review_priority=priority.get(filing.issuer_cik),
            )
        )

    require_unique_nonempty_rows(rows, label="section16-fact-verification")
    write_jsonl(args.output, rows)
    args.schema_output.parent.mkdir(parents=True, exist_ok=True)
    args.schema_output.write_text(
        json.dumps(
            braintrust_schemas(FactVerificationInput, FactVerificationExpected),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    flagged = sum(1 for row in rows if row["metadata"]["possible_blank_check_issuer"])
    print(f"Wrote {len(rows):,} Section 16 fact-verification rows to {args.output}")
    print(f"  flagged possible blank-check issuer: {flagged:,}")
    for reason in sorted({item["reason"] for item in skipped}):
        count = sum(1 for item in skipped if item["reason"] == reason)
        print(f"  skipped ({reason}): {count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
