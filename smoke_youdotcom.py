#!/usr/bin/env python3
"""Live smoke test of the You.com adapter against the real API.

The unit tests in tests/test_provider_adapters.py mock the response, so they
pin OUR handling but cannot confirm the response SHAPE. That gap is not
theoretical: if `contents.highlights` were absent, the adapter would silently
fall back to `description` and every mocked test would still pass while the
decision surface quietly degraded. This script is the check that closes it.

Costs one You.com call per query below. Run it after any change to the request
shape, and record the date in the spec block of tests/test_provider_adapters.py.

    .venv/bin/python smoke_youdotcom.py
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

from import_livenewsbench import load_env

REPOSITORY_ROOT = Path(__file__).resolve().parent

# One news-intent query and one evergreen query: You.com returns the `news`
# section only when it reads news intent, so a single query cannot exercise
# both the highlights path and the description fallback.
QUERIES = [
    "latest developments in the Middle East ceasefire talks",
    "who won the most recent Champions League final",
]


def main() -> int:
    env = load_env(REPOSITORY_ROOT / ".env")
    for key in ("YDC_API_KEY",):
        if key not in os.environ and key in env:
            os.environ[key] = env[key]
    if not os.environ.get("YDC_API_KEY"):
        print("YDC_API_KEY is not set; cannot run a live check.", file=sys.stderr)
        return 2

    import run_eval

    failures: list[str] = []
    notes: list[str] = []
    web_chars: list[int] = []
    news_chars: list[int] = []

    for query in QUERIES:
        results, raw = run_eval.youdotcom_search(query, "normalized", [])
        sections = raw.get("results") or {}
        web = sections.get("web") or []
        news = sections.get("news") or []
        setup_count = run_eval.ydc_setup("normalized")["count"]

        print(f"\n{query}")
        print(f"  returned  web={len(web)} news={len(news)} "
              f"-> surfaced {len(results)} (count={setup_count})")

        if not web:
            failures.append(f"{query!r}: no web results at all")
            continue

        # The claim the mocked tests cannot reach.
        with_highlights = [r for r in web
                           if (r.get("contents") or {}).get("highlights")]
        if not with_highlights:
            failures.append(
                f"{query!r}: extraction_mode=highlights returned no "
                f"contents.highlights on any web result — the adapter is "
                f"silently falling back to description")
        print(f"  web contents.highlights: {len(with_highlights)}/{len(web)}")

        if news:
            news_hl = [r for r in news if (r.get("contents") or {}).get("highlights")]
            print(f"  news contents.highlights: {len(news_hl)}/{len(news)}")
            if not news_hl:
                notes.append(
                    f"{query!r}: news results carry no contents.highlights; "
                    f"their snippet is the short `description` field")

        # News is additive, so the surface is web + news, both sections whole.
        if len(results) != len(web) + len(news):
            failures.append(
                f"{query!r}: surfaced {len(results)} of {len(web) + len(news)} "
                f"returned results; the merge is dropping results")

        if [r["rank"] for r in results] != list(range(1, len(results) + 1)):
            failures.append(f"{query!r}: ranks are not contiguous from 1")

        # Interleaving, not concatenation: news must not all sit at the bottom.
        sources = [r["source"] for r in results]
        if "news" in sources and "web" in sources:
            first_news = sources.index("news")
            if first_news >= len([s for s in sources if s == "web"]):
                failures.append(
                    f"{query!r}: first news result at rank {first_news + 1}; "
                    f"sections look concatenated rather than interleaved")
        print(f"  source order: {sources}")

        empty = [r["rank"] for r in results if not r["snippet"].strip()]
        if empty:
            notes.append(f"{query!r}: no text at rank(s) {empty}; surfaced for "
                         f"the title, URL, and publication date they carry")

        for r in results:
            (web_chars if r["source"] == "web" else news_chars).append(
                len(r["snippet"]))
        if results:
            print(f"  snippet chars: min={min(len(r['snippet']) for r in results)} "
                  f"max={max(len(r['snippet']) for r in results)}")

    if web_chars and news_chars:
        w, n = int(statistics.median(web_chars)), int(statistics.median(news_chars))
        print(f"\nmedian snippet chars: web={w} news={n}"
              + (f"  ({w // max(n, 1)}x)" if n else ""))
        if n and w // max(n, 1) >= 5:
            notes.append(
                f"web results carry ~{w // max(n, 1)}x the text of news results "
                f"({w} vs {n} chars). Expected: news snippets come from the "
                f"short `description` field. News earns its place on freshness "
                f"and publication dates, not text volume, and is additive, so "
                f"this costs no web coverage.")

    for note in notes:
        print(f"\nNOTE: {note}")
    for failure in failures:
        print(f"\nFAIL: {failure}", file=sys.stderr)

    print(f"\n{len(QUERIES)} queries, {len(failures)} failures, {len(notes)} notes")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
