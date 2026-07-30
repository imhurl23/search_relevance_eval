#!/usr/bin/env python3
"""Upload the approved metadata-only Corvus review queue to Braintrust."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import braintrust

from corvus.compliance import SOURCE_POLICY_PATH, require_import_approval, sha256_file
from import_livenewsbench import load_env


DATASET_NAME = "Corvus-QA-EDGAR-Review-2026-07"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--compliance-approval", required=True, type=Path)
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    approval = require_import_approval(
        args.compliance_approval,
        artifact_path=args.source_file,
        source_policy_path=args.source_policy,
    )
    rows = [
        json.loads(line)
        for line in args.source_file.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("refusing to upload an empty review queue")
    if any(row["metadata"].get("contains_source_text") is not False for row in rows):
        raise ValueError("every review row must explicitly exclude source text")

    env = load_env(args.env_file)
    api_key = env.get("BRAINTRUST_API_KEY")
    project_id = env.get("BRAINTRUST_PROJECT_ID")
    if not api_key or not project_id:
        raise ValueError("Braintrust credentials must be present in .env")
    os.environ["BRAINTRUST_API_KEY"] = api_key

    dataset = braintrust.init_dataset(
        project_id=project_id,
        name=DATASET_NAME,
        description=(
            "Metadata-only SEC Item 5.02 review queue. Filing bodies and excerpts "
            "are not stored in Braintrust."
        ),
        api_key=api_key,
        metadata={
            "artifact_sha256": sha256_file(args.source_file),
            "source_policy_sha256": sha256_file(args.source_policy),
            "approval_id": approval["approval_id"],
            "approval_scope": approval["scope"],
            "contains_source_text": False,
            "row_count": len(rows),
        },
    )
    for row in rows:
        dataset.insert(
            id=row["id"],
            input=row["input"],
            expected=row["expected"],
            metadata=row["metadata"],
            tags=row["tags"],
        )
    dataset.flush()
    print(
        f"Uploaded {len(rows):,} rows to {dataset.project.name}/{dataset.name} "
        f"(dataset_id={dataset.id})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
