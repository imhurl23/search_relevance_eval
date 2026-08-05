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
from corvus.review_io import read_jsonl
from corvus.review_schema import SCHEMA_VERSION, validate_review_row
from import_livenewsbench import load_env


DEFAULT_DATASET_NAME = "Corvus-QA-EDGAR-Review-2026-07"
DEFAULT_DESCRIPTION = (
    "Metadata-only SEC Item 5.02 review queue. Filing bodies and excerpts "
    "are not stored in Braintrust."
)

# The expected-side property that identifies which review stage a schema file
# describes. Importing fact-verification rows under the claim-preparation schema
# would put reviewer labels somewhere no scorer reads them.
STAGE_EXPECTED_MARKER = {
    "claim_preparation": "preparation_status",
    "fact_verification": "fact_verification",
}


def validate_rows(rows: list[dict], *, source: Path) -> str:
    """Enforce schema v2 on every row and return the single stage they share.

    Pre-v2 artifacts carry no ``review_stage`` and were previously accepted,
    which is how non-adherent rows reached Braintrust. They now fail here with
    an actionable message rather than becoming unlabelable dataset rows.
    """
    stages: set[str] = set()
    for index, row in enumerate(rows, start=1):
        stage = (row.get("metadata") or {}).get("review_stage")
        if not stage:
            raise ValueError(
                f"{source}:{index}: row has no metadata.review_stage. This is a "
                f"pre-{SCHEMA_VERSION} artifact; regenerate it with "
                "corvus.cli.prepare_claim_review or "
                "corvus.cli.prepare_section16_review before importing."
            )
        try:
            validate_review_row(row, stage=stage)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source}:{index}: {exc}") from exc
        stages.add(stage)
    if len(stages) > 1:
        raise ValueError(
            f"{source} mixes review stages {sorted(stages)}; import each stage "
            "into its own dataset so a scorer cannot span both."
        )
    return stages.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--compliance-approval", required=True, type=Path)
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument(
        "--schema-file",
        type=Path,
        help="Optional Braintrust metadata.__schemas JSON object.",
    )
    args = parser.parse_args()

    approval = require_import_approval(
        args.compliance_approval,
        artifact_path=args.source_file,
        source_policy_path=args.source_policy,
        schema_path=args.schema_file,
    )
    rows = read_jsonl(args.source_file)
    if not rows:
        raise ValueError("refusing to upload an empty review queue")
    # Enforces schema v2, the no-source-text rule, and metadata-only fields.
    stage = validate_rows(rows, source=args.source_file)
    schemas = None
    if args.schema_file:
        schemas = json.loads(args.schema_file.read_text())
        if not isinstance(schemas, dict) or not schemas:
            raise ValueError("--schema-file must contain a non-empty JSON object")
        for field in ("input", "expected"):
            if not isinstance(schemas.get(field), dict):
                raise ValueError(f"--schema-file is missing {field!r}")
            if schemas[field].get("enforce") is not True:
                raise ValueError(f"--schema-file {field!r} must set enforce=true")
        marker = STAGE_EXPECTED_MARKER[stage]
        if marker not in (schemas["expected"].get("properties") or {}):
            raise ValueError(
                f"--schema-file describes a different review stage: rows are "
                f"{stage!r} but the expected schema has no {marker!r} property"
            )

    env = load_env(args.env_file)
    api_key = env.get("BRAINTRUST_API_KEY")
    project_id = env.get("BRAINTRUST_PROJECT_ID")
    if not api_key or not project_id:
        raise ValueError("Braintrust credentials must be present in .env")
    os.environ["BRAINTRUST_API_KEY"] = api_key

    dataset = braintrust.init_dataset(
        project_id=project_id,
        name=args.dataset_name,
        description=args.description,
        api_key=api_key,
        metadata={
            "artifact_sha256": sha256_file(args.source_file),
            "source_policy_sha256": sha256_file(args.source_policy),
            "approval_id": approval["approval_id"],
            "approval_scope": approval["scope"],
            "contains_source_text": False,
            "row_count": len(rows),
            "review_stage": stage,
            "schema_version": SCHEMA_VERSION,
            **(
                {"schema_sha256": sha256_file(args.schema_file)}
                if args.schema_file
                else {}
            ),
            **({"__schemas": schemas} if schemas else {}),
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
