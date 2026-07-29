#!/usr/bin/env python3
"""Import LiveNewsBench releases into a versioned Braintrust dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import braintrust


DATASET_NAME = "LiveNewsBench"
RELEASES = ("sep_2025_release_1", "jan_2026_release_2")
SOURCE_REPOSITORY = "https://github.com/YunfanZhang42/LiveNewsBench"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def iter_source_rows(release_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    for source_path in sorted(release_dir.glob("*.jsonl")):
        split = source_path.stem
        with source_path.open() as source_file:
            for line_number, line in enumerate(source_file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {source_path}:{line_number}") from exc
                yield split, row


def stable_id(release: str, split: str, row: dict[str, Any]) -> str:
    identity = json.dumps(
        [release, split, row.get("question"), row.get("answer")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def insert_release(
    dataset: Any,
    datasets_root: Path,
    release: str,
    source_commit: str,
) -> int:
    count = 0
    for split, source_row in iter_source_rows(datasets_root / release):
        question = source_row.pop("question")
        answer = source_row.pop("answer")
        row_id = stable_id(release, split, {"question": question, "answer": answer})
        metadata = {
            **source_row,
            "livenewsbench_release": release,
            "livenewsbench_split": split,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": source_commit,
        }
        dataset.insert(
            id=row_id,
            input={"question": question},
            expected=answer,
            metadata=metadata,
            tags=[release, split],
        )
        count += 1
        if count % 500 == 0:
            print(f"{release}: queued {count:,} rows", flush=True)
    dataset.flush()
    print(f"{release}: uploaded {count:,} rows", flush=True)
    return count


def clear_dataset(dataset: Any) -> int:
    ids = [row["id"] for row in dataset.fetch()]
    for offset, row_id in enumerate(ids, start=1):
        dataset.delete(row_id)
        if offset % 500 == 0:
            print(f"Deleted {offset:,} prior head rows", flush=True)
    if ids:
        dataset.flush()
    print(f"Dataset head cleared ({len(ids):,} rows)", flush=True)
    return len(ids)


def create_snapshot(
    project_name: str,
    release: str,
    description: str,
    env_path: Path,
) -> None:
    command = [
        "bt",
        "datasets",
        "--env-file",
        str(env_path),
        "-p",
        project_name,
        "snapshots",
        "create",
        DATASET_NAME,
        release,
        "--description",
        description,
        "--json",
        "--no-input",
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_root", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--source-commit", required=True)
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
    dataset = braintrust.init_dataset(
        project_id=project_id,
        name=DATASET_NAME,
        description=(
            "LiveNewsBench quarterly releases, imported from the upstream JSONL "
            "files with named Braintrust snapshots."
        ),
        api_key=api_key,
        metadata={
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": args.source_commit,
        },
    )
    project_name = dataset.project.name
    print(
        f"Target: project={project_name!r}, dataset={dataset.name!r} ({dataset.id})",
        flush=True,
    )

    for release in RELEASES:
        clear_dataset(dataset)
        count = insert_release(dataset, args.datasets_root, release, args.source_commit)
        create_snapshot(
            project_name,
            release,
            (
                f"LiveNewsBench {release}: {count} rows from upstream commit "
                f"{args.source_commit}"
            ),
            args.env_file.resolve(),
        )

    summary = dataset.summarize()
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
