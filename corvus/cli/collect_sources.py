#!/usr/bin/env python3
"""Collect compliance-approved Corvus source observations and candidates."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import date, datetime
from pathlib import Path

from corvus.compliance import REPOSITORY_ROOT
from corvus.sources import (
    SEC_BULK_SUBMISSIONS,
    EdgarFilingCandidate,
    claim_entity_ids,
    make_news_sports_source_adapters,
    make_official_source_adapters,
)


def acquire_collector_lock():
    """Prevent two collectors in this workspace from spending the same budget."""
    lock_path = REPOSITORY_ROOT / ".corvus" / "collector.lock"
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(
            "another Corvus collector is running in this workspace"
        ) from exc
    return lock


def aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def write_jsonl(path: Path, records) -> None:
    with path.open("w") as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")


def load_candidates(path: Path) -> list[EdgarFilingCandidate]:
    candidates = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                candidates.append(EdgarFilingCandidate.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate: {exc}") from exc
    return candidates


def write_dict_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    edgar_parser = subparsers.add_parser("edgar-candidates")
    edgar_parser.add_argument("--cik", action="append", required=True)
    edgar_parser.add_argument("--since", type=date.fromisoformat, required=True)
    edgar_parser.add_argument("--until", type=date.fromisoformat, required=True)
    edgar_parser.add_argument("--output", type=Path, required=True)

    bulk_parser = subparsers.add_parser("edgar-bulk-candidates")
    bulk_parser.add_argument("--since", type=date.fromisoformat, required=True)
    bulk_parser.add_argument("--until", type=date.fromisoformat, required=True)
    bulk_parser.add_argument("--archive", type=Path, required=True)
    bulk_parser.add_argument("--output", type=Path, required=True)
    bulk_parser.add_argument(
        "--max-download-gb",
        type=float,
        default=5.0,
        help="Hard compressed-download ceiling; default 5 GiB.",
    )

    fetch_parser = subparsers.add_parser("edgar-fetch-filings")
    fetch_parser.add_argument("--candidates", type=Path, required=True)
    fetch_parser.add_argument("--output-dir", type=Path, required=True)
    fetch_parser.add_argument("--ledger", type=Path, required=True)

    wikidata_parser = subparsers.add_parser("wikidata-latest")
    wikidata_parser.add_argument("--qid", required=True)
    wikidata_parser.add_argument("--property", required=True, dest="property_id")
    wikidata_parser.add_argument("--attribute", required=True)
    wikidata_parser.add_argument("--effective-ts", type=aware_datetime, required=True)
    wikidata_parser.add_argument("--canonical-entity-id")
    wikidata_parser.add_argument("--entity-type", default="wikidata_item")
    wikidata_parser.add_argument("--output", type=Path, required=True)

    crosswalk_parser = subparsers.add_parser("wikidata-cik-leadership")
    crosswalk_parser.add_argument("--review-queue", type=Path, required=True)
    crosswalk_parser.add_argument(
        "--priority", choices=("high", "medium", "low"), default="high"
    )
    crosswalk_parser.add_argument("--output", type=Path, required=True)

    news_parser = subparsers.add_parser("wikipedia-current-events")
    news_parser.add_argument("--date", type=date.fromisoformat, required=True)
    news_parser.add_argument("--output", type=Path, required=True)

    openliga_parser = subparsers.add_parser("openligadb-results")
    openliga_parser.add_argument("--league", required=True)
    openliga_parser.add_argument("--season", required=True)
    openliga_parser.add_argument("--group-order", type=int)
    openliga_parser.add_argument("--output", type=Path, required=True)

    sportsdb_parser = subparsers.add_parser("thesportsdb-results")
    sportsdb_parser.add_argument("--date", type=date.fromisoformat, required=True)
    sportsdb_parser.add_argument("--sport")
    sportsdb_parser.add_argument("--league-id")
    sportsdb_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    collector_lock = acquire_collector_lock()
    if args.command in {
        "wikipedia-current-events",
        "openligadb-results",
        "thesportsdb-results",
    }:
        source_id = {
            "wikipedia-current-events": "wikipedia_current_events",
            "openligadb-results": "openligadb",
            "thesportsdb-results": "thesportsdb",
        }[args.command]
        wikipedia, openligadb, thesportsdb = make_news_sports_source_adapters(
            args.env_file,
            enabled_sources={source_id},
        )
        if args.command == "wikipedia-current-events":
            candidate = wikipedia.page_candidate(args.date)
            write_jsonl(args.output, [candidate])
            print(
                f"Wrote metadata-only Wikipedia Current Events candidate to "
                f"{args.output}; page text was not copied"
            )
        elif args.command == "openligadb-results":
            candidates = openligadb.completed_results(
                args.league,
                args.season,
                group_order=args.group_order,
            )
            write_jsonl(args.output, candidates)
            print(
                f"Wrote {len(candidates):,} completed OpenLigaDB results to "
                f"{args.output}"
            )
        else:
            candidates = thesportsdb.completed_results(
                args.date,
                sport=args.sport,
                league_id=args.league_id,
            )
            write_jsonl(args.output, candidates)
            print(
                f"Wrote {len(candidates):,} completed TheSportsDB results to "
                f"{args.output}"
            )
        collector_lock.close()
        return 0

    edgar, wikidata = make_official_source_adapters(args.env_file)

    if args.command in {"edgar-candidates", "edgar-bulk-candidates"}:
        if args.until < args.since:
            parser.error("--until cannot precede --since")
    if args.command == "edgar-candidates":
        candidates = []
        for cik in args.cik:
            candidates.extend(
                edgar.list_item_502(cik, since=args.since, until=args.until)
            )
        write_jsonl(args.output, candidates)
        print(
            f"Wrote {len(candidates)} Item 5.02 candidates to {args.output}. "
            "Review the filing and create OfficerTransition records before emission."
        )
        collector_lock.close()
        return 0

    if args.command == "edgar-bulk-candidates":
        if args.max_download_gb <= 0:
            parser.error("--max-download-gb must be positive")
        size, digest = edgar.client.download(
            SEC_BULK_SUBMISSIONS,
            args.archive,
            max_bytes=int(args.max_download_gb * 1024 ** 3),
        )
        candidates, filer_count = edgar.collect_bulk_item_502(
            args.archive, since=args.since, until=args.until
        )
        write_jsonl(args.output, candidates)
        print(
            f"Downloaded official SEC submissions archive: {size:,} bytes, "
            f"sha256={digest}"
        )
        print(
            f"Scanned {filer_count:,} filer records; wrote {len(candidates):,} "
            f"Item 5.02 candidates to {args.output}"
        )
        collector_lock.close()
        return 0

    if args.command == "edgar-fetch-filings":
        candidates = load_candidates(args.candidates)
        if not candidates:
            raise ValueError("candidate file is empty")
        ledger = edgar.fetch_candidate_filings(
            candidates, output_dir=args.output_dir
        )
        write_dict_jsonl(args.ledger, ledger)
        print(
            f"Wrote {len(ledger):,} local filing-document records to {args.ledger}"
        )
        collector_lock.close()
        return 0

    if args.command == "wikidata-cik-leadership":
        ciks = set()
        with args.review_queue.open() as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("review_priority") == args.priority:
                    ciks.add(record["cik"])
        rows = wikidata.fetch_current_ceos_by_cik(ciks)
        write_dict_jsonl(args.output, rows)
        print(
            f"Queried {len(ciks):,} unique CIKs in batches; wrote "
            f"{len(rows):,} current Wikidata CEO/reference rows to {args.output}"
        )
        collector_lock.close()
        return 0

    before, after, observed_ts = wikidata.fetch_latest_revision_pair(args.qid)
    label_ids = (
        {args.qid}
        | claim_entity_ids(before, args.property_id)
        | claim_entity_ids(after, args.property_id)
    )
    labels = wikidata.fetch_english_labels(label_ids)
    events = wikidata.events_from_revision_pair(
        qid=args.qid,
        property_id=args.property_id,
        before=before,
        after=after,
        observed_ts=observed_ts,
        effective_ts=args.effective_ts,
        labels=labels,
        attribute=args.attribute,
        canonical_entity_id=args.canonical_entity_id,
        entity_type=args.entity_type,
    )
    write_jsonl(args.output, events)
    print(f"Wrote {len(events)} referenced Wikidata change observations to {args.output}")
    collector_lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
