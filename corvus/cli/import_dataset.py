#!/usr/bin/env python3
"""Import one frozen Corvus-QA split into Braintrust and snapshot it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import braintrust
from pydantic import ValidationError

from corvus.compliance import (
    SOURCE_POLICY_PATH,
    require_approved_sources,
    require_import_approval,
)
from corvus.models import CorvusRow, DatasetSplit
from import_livenewsbench import load_env


def iter_rows(path: Path) -> Iterator[CorvusRow]:
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield CorvusRow.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"Invalid Corvus row at {path}:{line_number}: {exc}") from exc


def clear_dataset(dataset: Any) -> int:
    ids = [row["id"] for row in dataset.fetch()]
    for row_id in ids:
        dataset.delete(row_id)
    if ids:
        dataset.flush()
    return len(ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=[item.value for item in DatasetSplit])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    parser.add_argument(
        "--compliance-approval",
        required=True,
        type=Path,
        help="Authorized, artifact-bound approval for private Braintrust import.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != args.split:
        raise ValueError("manifest split does not match --split")
    if manifest.get("output_sha256") != sha256_file(args.source_file):
        raise ValueError("source SHA-256 does not match the frozen manifest")
    rows = list(iter_rows(args.source_file))
    if not rows:
        raise ValueError("refusing to clear a dataset for an empty source file")
    bad_splits = {
        row.metadata.get("corvus_split")
        for row in rows
        if row.metadata.get("corvus_split") != args.split
    }
    if bad_splits:
        raise ValueError(f"source contains rows outside {args.split!r}: {sorted(bad_splits)}")
    if manifest.get("row_count") != len(rows):
        raise ValueError("manifest row_count does not match source file")
    source_ids = {
        source_id
        for row in rows
        for source_id in row.metadata.get("compliance_source_ids", [])
    }
    require_approved_sources(source_ids, path=args.source_policy)
    if manifest.get("source_policy_sha256") != sha256_file(args.source_policy):
        raise ValueError("manifest does not match the current source compliance policy")
    approval = require_import_approval(
        args.compliance_approval,
        artifact_path=args.source_file,
        source_policy_path=args.source_policy,
    )
    manifest["compliance_approval"] = {
        "approval_id": approval["approval_id"],
        "approved_by": approval["approved_by"],
        "approved_at": approval["approved_at"],
        "scope": approval["scope"],
    }

    env = load_env(args.env_file)
    api_key = env.get("BRAINTRUST_API_KEY")
    project_id = env.get("BRAINTRUST_PROJECT_ID")
    if not api_key or not project_id:
        raise ValueError(
            "BRAINTRUST_API_KEY and BRAINTRUST_PROJECT_ID are required in the env file"
        )
    # Repository policy: the .env value always overrides ambient credentials.
    os.environ["BRAINTRUST_API_KEY"] = api_key

    dataset_name = f"Corvus-QA-{args.split}"
    freeze_id = str(manifest["freeze_id"])
    dataset = braintrust.init_dataset(
        project_id=project_id,
        name=dataset_name,
        description=(
            f"Corvus-QA {args.split} split: independently resolved, "
            "post-cutoff fact transitions."
        ),
        api_key=api_key,
        metadata=manifest,
    )
    deleted = clear_dataset(dataset)
    for row in rows:
        dataset.insert(
            id=row.id,
            input=row.input,
            expected=row.expected,
            metadata=row.metadata,
            tags=row.tags,
        )
    dataset.flush()

    safe_freeze_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", freeze_id)
    snapshot_name = f"{args.split}-{safe_freeze_id}"
    subprocess.run(
        [
            "bt",
            "datasets",
            "--env-file",
            str(args.env_file.resolve()),
            "-p",
            dataset.project.name,
            "snapshots",
            "create",
            dataset_name,
            snapshot_name,
            "--description",
            f"Corvus-QA {args.split}: {len(rows)} rows, freeze {freeze_id}",
            "--json",
            "--no-input",
        ],
        check=True,
    )
    print(
        f"Uploaded {len(rows)} rows to {dataset_name}; replaced {deleted} head rows; "
        f"snapshot={snapshot_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
