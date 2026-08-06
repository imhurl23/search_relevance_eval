#!/usr/bin/env python3
"""Stage 1b: emit dual-attested FactEvents from already-collected EDGAR artifacts.

This is the curation step that Corvus-QA previously required a human to perform,
made deterministic. It consumes only artifacts already on disk from
``collect_sources`` and makes NO network requests, so it can be re-run freely and
its output is a pure function of the frozen inputs.

For each parsed Section 16 filing whose ``officerTitle`` maps to a benchmark
attribute, it reads the paired Item 5.02 filing body from local disk and asks
whether the issuer's own text asserts the same person, office, and effective date
(see corvus/issuer_corroboration.py for why that is verification rather than
extraction). When it does, TWO observations are emitted for the transition:

  * the issuer, attester_role="issuer", accountable via the issuer CIK
  * the reporting owner, attester_role="reporting_owner", via the owner CIK

which is what satisfies ``build_dataset --min-attesters 2``. When it does not,
the candidate is written to the rejection ledger with a reason code, because a
yield number is only interpretable alongside what it excluded.

Two design points worth stating plainly, since both are visible in the output:

``new_value`` is taken from the STRUCTURED Section 16 field on both observations,
not from the issuer's prose. The issuer's own spelling goes into ``aliases``. The
two attesters therefore agree that a particular PERSON holds the office rather
than agreeing on a byte string — which is the only defensible reading, because
"Stephen R. Curley" and "Curley Stephen Russell" are the same attestation and
``_agreement_key`` would otherwise split them into two single-attested groups.
The published answer stays the structured field, exactly as
``person_agreement_key`` documents.

``old_value`` is None on both observations, so every row's ``answer_class`` is
``post_cutoff_novel``. That field here means "no predecessor was attested", NOT
"the office did not previously exist" — a CEO succession has a predecessor and
this pipeline does not attest it, because neither a Form 3 (an initial statement)
nor a name match in the successor's announcement establishes who held the office
before. Do not stratify on answer_class for this freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, time, timezone
from pathlib import Path

from pydantic import ValidationError

from corvus.compliance import SOURCE_POLICY_PATH, require_approved_sources, sha256_file
from corvus.issuer_corroboration import (NotCorroborated, confirm_appointment,
                                         filing_text)
from corvus.models import FactEvent
from corvus.section16 import OwnershipFilingRef, Section16Adapter, Section16Filing
from corvus.sources import EdgarAdapter, EdgarFilingCandidate, OfficerTransition


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
    return rows


def load_typed(path: Path, model, label: str) -> list:
    values = []
    for number, row in enumerate(load_jsonl(path), start=1):
        try:
            values.append(model.model_validate(row))
        except ValidationError as exc:
            raise ValueError(f"Invalid {label} at {path}:{number}: {exc}") from exc
    return values


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for record in records:
            if isinstance(record, dict):
                sink.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                sink.write(record.model_dump_json() + "\n")


def midnight_utc(value) -> datetime:
    """Day-precision timestamp. Both attesters must land on the same instant."""
    return datetime.combine(value, time(0, 0), tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit dual-attested Corvus-QA FactEvents by confirming each Section 16 "
            "appointment against the issuer's own Item 5.02 filing. Reads local "
            "artifacts only; makes no network requests."
        ),
        epilog=(
            "Inspect the rejection ledger before trusting the yield, then pass the "
            "output to corvus.cli.build_dataset."
        ),
    )
    parser.add_argument("--candidates", required=True, type=Path,
                        help="edgar_item_502_candidates.jsonl")
    parser.add_argument("--form3-filings", required=True, type=Path,
                        help="Parsed Section16Filing JSONL.")
    parser.add_argument("--form3-refs", required=True, type=Path,
                        help="OwnershipFilingRef JSONL carrying the Item 5.02 pairing.")
    parser.add_argument("--filings-ledger", required=True, type=Path,
                        help="filing_download_ledger.jsonl, for local document paths.")
    parser.add_argument("--output", required=True, type=Path,
                        help="FactEvent JSONL, two observations per confirmed row.")
    parser.add_argument("--rejections-file", type=Path,
                        help="Defaults to <output>.rejections.jsonl.")
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    args = parser.parse_args()

    candidates = load_typed(args.candidates, EdgarFilingCandidate,
                            "EdgarFilingCandidate")
    filings = load_typed(args.form3_filings, Section16Filing, "Section16Filing")
    refs = load_typed(args.form3_refs, OwnershipFilingRef, "OwnershipFilingRef")
    ledger = load_jsonl(args.filings_ledger)

    require_approved_sources({"sec_edgar"}, path=args.source_policy)

    by_accession = {(c.cik, c.accession_number): c for c in candidates}
    local_path = {
        (row["cik"], row["accession_number"]): Path(row["local_path"])
        for row in ledger
        if row.get("local_path")
    }
    pairing = {ref.accession_number: ref for ref in refs}

    events: list[FactEvent] = []
    rejected: list[dict] = []
    reasons: Counter = Counter()
    attributes: Counter = Counter()
    # One document is read many times when several officers file against it, so
    # the normalized text is cached rather than re-derived.
    text_cache: dict[Path, str] = {}
    confirmed_groups = 0

    for filing in sorted(filings, key=lambda f: f.accession_number):
        attribute = filing.role_attribute
        if attribute is None:
            reasons["title_not_a_benchmark_office"] += 1
            continue
        ref = pairing.get(filing.accession_number)
        if ref is None:
            reasons["no_pairing_record"] += 1
            continue

        confirmation = None
        candidate = None
        attempts: list[dict] = []
        for accession in sorted(ref.paired_item_502_accessions):
            option = by_accession.get((filing.issuer_cik, accession))
            if option is None:
                attempts.append({"accession": accession,
                                 "reason": "candidate_not_in_window"})
                continue
            path = local_path.get((filing.issuer_cik, accession))
            if path is None or not path.is_file():
                attempts.append({"accession": accession,
                                 "reason": "filing_body_not_on_disk"})
                continue
            if path not in text_cache:
                text_cache[path] = filing_text(path.read_bytes())
            try:
                confirmation = confirm_appointment(
                    text_cache[path],
                    edgar_owner_name=filing.owner_name,
                    role_attribute=attribute,
                    effective_date=filing.period_of_report,
                    # Needed to tell "CEO of the Company" from "CEO of
                    # Inflection Point" in a filing that mentions both.
                    issuer_name=option.entity_name,
                )
            except NotCorroborated as error:
                attempts.append({"accession": accession, "reason": error.reason,
                                 "detail": error.detail})
                continue
            candidate = option
            break

        if confirmation is None or candidate is None:
            # Report the most specific failure rather than a generic one: an
            # issuer that named a different office is a different finding from an
            # issuer whose body was never downloaded.
            reason = attempts[0]["reason"] if attempts else "no_paired_candidate"
            reasons[reason] += 1
            rejected.append({
                "issuer_cik": filing.issuer_cik,
                "issuer_name": filing.issuer_name,
                "owner_cik": filing.owner_cik,
                "officer_title": filing.officer_title,
                "role_attribute": attribute,
                "section16_accession": filing.accession_number,
                "period_of_report": filing.period_of_report.isoformat(),
                "reason": reason,
                "attempts": attempts,
            })
            continue

        effective_ts = midnight_utc(filing.period_of_report)
        # The answer is the structured Section 16 name on BOTH observations, so
        # the two group together. The issuer's spelling rides along as an alias.
        answer = filing.natural_owner_name
        aliases = [confirmation.issuer_spelling] if confirmation.issuer_spelling else []
        transition = OfficerTransition(
            role_attribute=attribute,
            old_value=None,
            new_value=answer,
            effective_ts=effective_ts,
            aliases=aliases,
            evidence_excerpt_sha256=confirmation.evidence_excerpt_sha256,
        )
        issuer_event = EdgarAdapter.emit_fact(candidate, transition)
        owner_event = Section16Adapter.emit_fact(
            filing,
            entity_name=candidate.entity_name,
            # When the filing was made, not when the event happened. The default
            # would reuse the event date and understate observation lag.
            observed_ts=midnight_utc(ref.filing_date),
        )
        # Record what the issuer side was actually checked against, so a reader
        # can re-audit the confirmation without the document body.
        issuer_event.provenance.update({
            "issuer_corroboration": {
                "matched_date_form": confirmation.matched_date_form,
                "evidence_start": confirmation.evidence_start,
                "evidence_end": confirmation.evidence_end,
                "issuer_spelling": confirmation.issuer_spelling,
                "answer_source": "section16_structured_field",
                "local_document_sha256": next(
                    (row["sha256"] for row in ledger
                     if row.get("cik") == filing.issuer_cik
                     and row.get("accession_number") == candidate.accession_number),
                    None),
            },
            "section16_accession": filing.accession_number,
        })
        events.extend([issuer_event, owner_event])
        confirmed_groups += 1
        attributes[attribute] += 1
        reasons["confirmed"] += 1

    rejections_path = args.rejections_file or args.output.with_suffix(
        args.output.suffix + ".rejections.jsonl")
    write_jsonl(args.output, events)
    write_jsonl(rejections_path, rejected)

    manifest = {
        "stage": "Corvus-QA issuer corroboration",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in [
                ("candidates", args.candidates),
                ("form3_filings", args.form3_filings),
                ("form3_refs", args.form3_refs),
                ("filings_ledger", args.filings_ledger),
            ]
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output),
                   "observation_count": len(events),
                   "confirmed_transition_count": confirmed_groups},
        "rejections": {"path": str(rejections_path),
                       "sha256": sha256_file(rejections_path),
                       "count": len(rejected)},
        "outcomes": dict(sorted(reasons.items())),
        "confirmed_by_attribute": dict(sorted(attributes.items())),
        "network_requests": 0,
        "notes": [
            "Filing bodies were read from local disk and are not reproduced in "
            "the output; only a SHA-256 of each evidence window is retained.",
            "answer_class is post_cutoff_novel on every row because no "
            "predecessor is attested. It does not mean the office is new.",
            "new_value comes from the Section 16 structured field on both "
            "observations; the issuer's spelling is recorded as an alias.",
        ],
        "source_policy": str(args.source_policy),
        "source_policy_sha256": sha256_file(args.source_policy),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Confirmed {confirmed_groups} transitions "
          f"({len(events)} observations) from {len(filings)} Section 16 filings.")
    print(f"Outcomes: {dict(sorted(reasons.items()))}")
    print(f"By attribute: {dict(sorted(attributes.items()))}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
