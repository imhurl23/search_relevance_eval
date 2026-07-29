#!/usr/bin/env python3
"""Import a pinned RetrievalQA release into a versioned Braintrust dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import braintrust

from import_livenewsbench import load_env


DATASET_NAME = "RetrievalQA"
SOURCE_REPOSITORY = "https://huggingface.co/datasets/zihanz/RetrievalQA"


def iter_rows(source_path: Path) -> Iterator[dict[str, Any]]:
    with source_path.open() as source_file:
        for line_number, line in enumerate(source_file, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source_path}:{line_number}") from exc


def clear_dataset(dataset: Any) -> int:
    ids = [row["id"] for row in dataset.fetch()]
    for row_id in ids:
        dataset.delete(row_id)
    if ids:
        dataset.flush()
    print(f"Dataset head cleared ({len(ids):,} rows)", flush=True)
    return len(ids)


def create_snapshot(
    project_name: str,
    snapshot_name: str,
    description: str,
    env_path: Path,
) -> None:
    subprocess.run(
        [
            "bt",
            "datasets",
            "--env-file",
            str(env_path),
            "-p",
            project_name,
            "snapshots",
            "create",
            DATASET_NAME,
            snapshot_name,
            "--description",
            description,
            "--json",
            "--no-input",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()

    env = load_env(args.env_file)
    api_key = env.get("BRAINTRUST_API_KEY")
    project_id = env.get("BRAINTRUST_PROJECT_ID")
    if not api_key or not project_id:
        raise ValueError(
            "BRAINTRUST_API_KEY and BRAINTRUST_PROJECT_ID are required in the env file"
        )

    # Intentionally override any ambient credential and also pass the key explicitly.
    os.environ["BRAINTRUST_API_KEY"] = api_key
    snapshot_name = f"hf_{args.source_revision[:12]}"
    dataset = braintrust.init_dataset(
        project_id=project_id,
        name=DATASET_NAME,
        description=(
            "RetrievalQA short-form open-domain QA, pinned to an exact Hugging "
            "Face dataset revision."
        ),
        api_key=api_key,
        metadata={
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": args.source_revision,
            "source_sha256": args.source_sha256,
        },
    )
    print(
        f"Target: project={dataset.project.name!r}, "
        f"dataset={dataset.name!r} ({dataset.id})",
        flush=True,
    )

    clear_dataset(dataset)
    count = 0
    for source_row in iter_rows(args.source_file):
        question_id = source_row.pop("question_id")
        question = source_row.pop("question")
        expected = source_row.pop("ground_truth")
        data_source = source_row.get("data_source")
        dataset.insert(
            id=question_id,
            input={"question": question},
            expected=expected,
            metadata={
                **source_row,
                "question_id": question_id,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": args.source_revision,
                "source_sha256": args.source_sha256,
            },
            tags=[data_source] if data_source else None,
        )
        count += 1

    dataset.flush()
    print(f"Uploaded {count:,} rows", flush=True)
    create_snapshot(
        dataset.project.name,
        snapshot_name,
        (
            f"RetrievalQA: {count} rows from Hugging Face revision "
            f"{args.source_revision}"
        ),
        args.env_file.resolve(),
    )
    print(dataset.summarize(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
