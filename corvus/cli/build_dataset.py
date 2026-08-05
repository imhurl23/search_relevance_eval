#!/usr/bin/env python3
"""Stage 2: freeze final Corvus-QA rows from curated FactEvent JSONL.

This command is deliberately downstream of source collection and curation. It
does not accept source-specific candidates. Curators must first produce one
normalized ``FactEvent`` observation per attester/resolver.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from corvus.models import (
    CoverageAssessment,
    DatasetSplit,
    FactEvent,
    TrapObservation,
    build_rows,
    build_trap_rows,
    canonical_value,
)
from corvus.compliance import SOURCE_POLICY_PATH, require_approved_sources, sha256_file


def load_events(path: Path) -> list[FactEvent]:
    events = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                events.append(FactEvent.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"Invalid FactEvent at {path}:{line_number}: {exc}") from exc
    return events


def load_models(path: Path, model_type, label: str):
    values = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                values.append(model_type.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"Invalid {label} at {path}:{line_number}: {exc}") from exc
    return values


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def coverage_key(item: CoverageAssessment) -> tuple[str, str, str]:
    return (
        item.entity_id.casefold(),
        item.attribute.casefold(),
        canonical_value(item.new_value),
    )


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for record in records:
            if hasattr(record, "model_dump"):
                record = record.model_dump(mode="json")
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 of Corvus-QA curation: apply deterministic eligibility "
            "rules and freeze final benchmark rows from normalized FactEvents."
        ),
        epilog=(
            "Inputs must be FactEvent JSONL, not collected candidates. Inspect "
            "the output, rejection ledger, and manifest before "
            "running corvus.cli.import_dataset."
        ),
    )
    parser.add_argument(
        "events_file",
        type=Path,
        help="Curated FactEvent JSONL (one observation per source).",
    )
    parser.add_argument("output_file", type=Path, help="Frozen CorvusRow JSONL artifact.")
    parser.add_argument("--split", required=True, choices=[item.value for item in DatasetSplit])
    parser.add_argument(
        "--freeze-id",
        required=True,
        help="Pre-registered source-window/freeze identifier, included in every row ID.",
    )
    parser.add_argument("--min-resolvers", type=int, default=2)
    parser.add_argument(
        "--min-attesters",
        type=int,
        default=2,
        help=(
            "Independently accountable parties that must agree. This is the "
            "corroboration gate; default 2."
        ),
    )
    parser.add_argument(
        "--min-authorities",
        type=int,
        default=1,
        help=(
            "Distinct publishers required. Defaults to 1 because publisher "
            "diversity is provenance, not corroboration; raise it for a study "
            "that needs distribution-channel independence too."
        ),
    )
    parser.add_argument(
        "--as-of",
        required=True,
        type=parse_datetime,
        help="Frozen evaluation timestamp used to assign recency rungs.",
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        help="Optional JSONL of coverage assessments with confirmed storage rights.",
    )
    parser.add_argument(
        "--traps-file",
        type=Path,
        help="Optional JSONL of independently confirmed future-event observations.",
    )
    parser.add_argument(
        "--run-end",
        type=parse_datetime,
        help="Required with --traps-file; traps must resolve more than 7 days later.",
    )
    parser.add_argument(
        "--rejections-file",
        type=Path,
        help="Defaults to <output>.rejections.jsonl.",
    )
    parser.add_argument(
        "--source-policy", type=Path, default=SOURCE_POLICY_PATH
    )
    args = parser.parse_args()

    events = load_events(args.events_file)
    coverage_items = (
        load_models(args.coverage_file, CoverageAssessment, "CoverageAssessment")
        if args.coverage_file
        else []
    )
    coverage = {coverage_key(item): item for item in coverage_items}
    if len(coverage) != len(coverage_items):
        raise ValueError("coverage file contains duplicate assessment keys")
    source_ids = {event.compliance_source_id for event in events}
    source_ids.update(item.compliance_source_id for item in coverage_items)

    rows, rejected = build_rows(
        events,
        split=DatasetSplit(args.split),
        freeze_id=args.freeze_id,
        min_resolvers=args.min_resolvers,
        min_attesters=args.min_attesters,
        min_authorities=args.min_authorities,
        as_of_ts=args.as_of,
        coverage=coverage,
    )
    trap_observation_count = 0
    if args.traps_file:
        if args.run_end is None:
            parser.error("--run-end is required with --traps-file")
        traps = load_models(args.traps_file, TrapObservation, "TrapObservation")
        source_ids.update(item.compliance_source_id for item in traps)
        trap_observation_count = len(traps)
        trap_rows, trap_rejected = build_trap_rows(
            traps,
            split=DatasetSplit(args.split),
            freeze_id=args.freeze_id,
            run_end=args.run_end,
            min_resolvers=args.min_resolvers,
            min_attesters=args.min_attesters,
            min_authorities=args.min_authorities,
        )
        rows.extend(trap_rows)
        rejected.extend(trap_rejected)
    elif args.run_end is not None:
        parser.error("--run-end is only valid with --traps-file")
    require_approved_sources(source_ids, path=args.source_policy)
    rejections_path = args.rejections_file or args.output_file.with_suffix(
        args.output_file.suffix + ".rejections.jsonl"
    )
    write_jsonl(args.output_file, rows)
    write_jsonl(rejections_path, rejected)

    manifest = {
        "dataset": "Corvus-QA",
        "freeze_id": args.freeze_id,
        "split": args.split,
        "events_file": str(args.events_file),
        "events_sha256": sha256_file(args.events_file),
        "output_file": str(args.output_file),
        "output_sha256": sha256_file(args.output_file),
        "event_count": len(events),
        "trap_observation_count": trap_observation_count,
        "coverage_assessment_count": len(coverage_items),
        "row_count": len(rows),
        "rejection_count": len(rejected),
        "min_resolvers": args.min_resolvers,
        "min_attesters": args.min_attesters,
        "min_authorities": args.min_authorities,
        "as_of": args.as_of.isoformat(),
        "run_end": args.run_end.isoformat() if args.run_end else None,
        "source_policy": str(args.source_policy),
        "source_policy_sha256": sha256_file(args.source_policy),
        "compliance_source_ids": sorted(source_ids),
    }
    manifest_path = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Built {len(rows)} {args.split} rows; rejected {len(rejected)} groups. "
        f"Manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
