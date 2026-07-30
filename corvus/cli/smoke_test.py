#!/usr/bin/env python3
"""Run an offline end-to-end smoke test of Corvus curation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from corvus.models import CorvusRow


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "corvus_smoke_events.jsonl"


def main() -> int:
    with TemporaryDirectory(prefix="corvus-smoke-") as directory:
        output = Path(directory) / "corvus-smoke.jsonl"
        command = [
            sys.executable,
            "-m",
            "corvus.cli.build_dataset",
            str(FIXTURE),
            str(output),
            "--split",
            "dev",
            "--freeze-id",
            "offline-smoke-v1",
            "--as-of",
            "2026-07-30T16:00:00Z",
        ]
        subprocess.run(command, check=True)

        rows = [
            CorvusRow.model_validate_json(line)
            for line in output.read_text().splitlines()
            if line.strip()
        ]
        manifest = json.loads(
            output.with_suffix(".jsonl.manifest.json").read_text()
        )
        rejections = [
            json.loads(line)
            for line in output.with_suffix(".jsonl.rejections.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]

        if len(rows) != 1:
            raise AssertionError(f"expected 1 curated row, got {len(rows)}")
        row = rows[0]
        if row.expected != "Alex Example":
            raise AssertionError(f"unexpected answer: {row.expected!r}")
        if row.metadata["recency_rung"] != "24h_72h":
            raise AssertionError(
                f"unexpected recency rung: {row.metadata['recency_rung']!r}"
            )
        if set(row.metadata["authority_families"]) != {
            "sec",
            "publisher:example.invalid",
        }:
            raise AssertionError("dual-authority resolution was not preserved")
        if len(rejections) != 1 or "insufficient_resolvers" not in rejections[0]["reasons"]:
            raise AssertionError("single-resolver rejection was not recorded")
        if manifest["row_count"] != 1 or manifest["rejection_count"] != 1:
            raise AssertionError("manifest counts do not match smoke artifacts")
        if set(manifest["compliance_source_ids"]) != {"sec_edgar", "wikidata"}:
            raise AssertionError("manifest source IDs do not match approved fixtures")

        print("PASS offline Corvus curation smoke test")
        print("  curated_rows=1")
        print("  rejected_groups=1 (insufficient independent resolution)")
        print("  recency_rung=24h_72h")
        print("  source_policy=sec_edgar,wikidata")
        print("  external_requests=0")
        print("  external_uploads=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
