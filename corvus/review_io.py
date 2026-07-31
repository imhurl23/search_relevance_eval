"""Small deterministic I/O helpers shared by Corvus review tools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def stable_id(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def authority_family_for_url(url: str) -> str:
    host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
    if not host:
        raise ValueError(f"URL has no hostname: {url!r}")
    return f"publisher:{host}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in materialized:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(materialized)


def require_unique_nonempty_rows(
    rows: list[dict[str, Any]], *, label: str
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty {label} dataset")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {label} dataset")
