#!/usr/bin/env python3
"""Live check of the You.com request shape against the real API.

tests/test_provider_adapters.py mocks the response, so it pins our handling but
cannot confirm the shape. That gap is load-bearing: if `contents.highlights`
disappeared, the adapter would fall back to `description` and every mocked test
would still pass while the decision surface silently degraded.

Costs one call per query. Run after any change to the request shape, and record
the date in that file's spec block.

    .venv/bin/python smoke_youdotcom.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from import_livenewsbench import load_env

# One news-intent and one evergreen query: the `news` section only appears when
# You.com reads news intent, so one query cannot exercise both paths.
QUERIES = [
    "latest developments in the Middle East ceasefire talks",
    "who won the most recent Champions League final",
]


def main() -> int:
    env = load_env(Path(__file__).resolve().parent / ".env")
    os.environ.setdefault("YDC_API_KEY", env.get("YDC_API_KEY", ""))
    if not os.environ.get("YDC_API_KEY"):
        print("YDC_API_KEY is not set; cannot run a live check.", file=sys.stderr)
        return 2

    import run_eval

    failures = []
    for query in QUERIES:
        results, raw = run_eval.youdotcom_search(query, "normalized", [])
        sections = raw.get("results") or {}
        web = sections.get("web") or []
        news = sections.get("news") or []
        sources = [r["source"] for r in results]
        highlighted = sum(1 for r in web if (r.get("contents") or {}).get("highlights"))

        print(f"\n{query}")
        print(f"  web={len(web)} news={len(news)} -> surfaced {len(results)}")
        print(f"  web with contents.highlights: {highlighted}/{len(web)}")
        print(f"  source order: {sources}")

        if not web:
            failures.append(f"{query!r}: no web results")
            continue
        # The claim the mocked tests cannot reach.
        if not highlighted:
            failures.append(
                f"{query!r}: extraction_mode=highlights produced no "
                f"contents.highlights — silently falling back to description")
        # News is additive, so every returned result should surface.
        if len(results) != len(web) + len(news):
            failures.append(
                f"{query!r}: surfaced {len(results)} of {len(web) + len(news)} "
                f"returned — the merge is dropping results")
        # Interleaved, not concatenated: news must not all sit at the bottom.
        if news and "news" in sources and sources.index("news") > 1:
            failures.append(
                f"{query!r}: first news result at rank "
                f"{sources.index('news') + 1}; sections look concatenated")

    for failure in failures:
        print(f"\nFAIL: {failure}", file=sys.stderr)
    print(f"\n{len(QUERIES)} queries, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
