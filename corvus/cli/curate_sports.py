#!/usr/bin/env python3
"""Turn reviewed sports result mappings into normalized Corvus FactEvents."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

from corvus.compliance import SOURCE_POLICY_PATH, require_approved_sources, sha256_file
from corvus.sports_curation import (
    load_candidates,
    load_decisions,
    reconcile_sports,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply explicit, reviewed match-identity and completion-time decisions "
            "to collected sports results."
        )
    )
    parser.add_argument(
        "--candidates", action="append", required=True, type=Path,
        help="Repeat for each provider's SportsResultCandidate JSONL.",
    )
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-resolvers", type=int, default=2)
    parser.add_argument("--min-authorities", type=int, default=2)
    parser.add_argument("--max-start-delta-minutes", type=int, default=360)
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    decisions = load_decisions(args.decisions)
    facts, skipped = reconcile_sports(
        candidates,
        decisions,
        min_resolvers=args.min_resolvers,
        min_authorities=args.min_authorities,
        max_start_delta=timedelta(minutes=args.max_start_delta_minutes),
    )
    if not facts:
        raise ValueError("no approved sports decisions produced facts")
    source_ids = {fact.compliance_source_id for fact in facts}
    require_approved_sources(source_ids, path=args.source_policy)
    write_jsonl(args.output, facts)
    manifest = {
        "stage": "Corvus-QA sports reconciliation",
        "candidate_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.candidates
        ],
        "decisions_file": str(args.decisions),
        "decisions_sha256": sha256_file(args.decisions),
        "output_file": str(args.output),
        "output_sha256": sha256_file(args.output),
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "approved_decision_count": len(decisions) - len(skipped),
        "skipped_decision_ids": skipped,
        "fact_event_count": len(facts),
        "compliance_source_ids": sorted(source_ids),
        "source_policy": str(args.source_policy),
        "source_policy_sha256": sha256_file(args.source_policy),
        "min_resolvers": args.min_resolvers,
        "min_authorities": args.min_authorities,
        "max_start_delta_minutes": args.max_start_delta_minutes,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(facts)} sports observations from "
        f"{len(decisions) - len(skipped)} approved matches to {args.output}; "
        f"skipped {len(skipped)} unapproved decisions"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
