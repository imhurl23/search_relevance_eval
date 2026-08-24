"""Braintrust harness for a You.com-vs-native-search freshness study.

Two treatment axes, selected on the command line:

    --model-vendor   baseten (OSS) | openai (frontier) | anthropic (frontier)
    --agent-model    which model within that vendor; see agents.MATRIX_MODELS
    --search-mode    none (parametric) | harness (You.com) | native (vendor's own)
    --arm            the You.com setup, for --search-mode harness

You.com is the only search API, which makes the SETUP the harness treatment
rather than the provider. One consequence to carry into any writeup: nothing in
this design separates "You.com beats native search" from "independent APIs beat
native search."

Within a condition the prompt, tool contract, snippet budget, and search budget
are frozen. Sampling is NOT frozen and cannot be: both frontier vendors reject
sampling parameters outright and no vendor here supports `seed`, so only the OSS
arm pins temperature. Each run records what it actually sent (`sampling_params`,
`sampling_pinned`) rather than implying parity.

Braintrust-native throughout:

  * reads a versioned dataset from the project named by BRAINTRUST_PROJECT_ID;
    real comparisons require a pinned version
  * agent LLM calls auto-traced via wrap_openai / wrap_anthropic
  * every approved search is a `tool` span with native metrics; raw provider
    payloads are never retained
  * per-row search_cost_usd, model_cost_usd, total_cost_usd, and latency_s,
    because search fees are the small half of the bill
  * trial_count for retrieval nondeterminism (the web is not frozen by anything
    we control)

Dataset contract (set by the importers, consumed by scorers.py):
    input    = {"question": str}
    expected = answer string or list of acceptable answer strings
    metadata = upstream row fields (link, articles, event_date, ...) plus
               importer provenance; Corvus-QA adds recency_rung / coverage_tier
Leakage excludes and temporal grounding both read from metadata, not expected.

Usage:
    # .env supplies BRAINTRUST_API_KEY + BRAINTRUST_PROJECT_ID and always wins.
    # All credentials come from .env; ambient credentials are ignored.

    # dataset push lives in the importers, not here:
    python import_livenewsbench.py <datasets_root> --source-commit <sha>

    # one command per matrix cell; interleave them in time, don't run days apart
    python run_eval.py run --model-vendor openai --search-mode harness \
      --arm normalized \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 1
    python run_eval.py run --model-vendor openai --search-mode native \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 1
    python run_eval.py run --model-vendor baseten \
      --agent-model deepseek-ai/DeepSeek-V4-Flash-0731 --search-mode none \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 1

Native search is only attributable WITHIN a vendor: run each frontier vendor's
none/harness/native arms together and compare its native arm to its own harness
arm, never to the other vendor's. Full matrix and the contrasts it supports:
README "The test matrix" and docs/study-design.md.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import httpx
from braintrust import Eval, current_span, init_dataset, traced, wrap_openai
from openai import OpenAI

import agents
from agents import (SEARCH_MODE_HARNESS, SEARCH_MODE_NATIVE, SEARCH_MODE_NONE,
                    SEARCH_MODES, SURFACE_FULL, SURFACE_NONE, VENDORS,
                    make_harness_session, native_search_rate_usd, vendor_of)
from import_livenewsbench import DATASET_NAME, load_env
from import_retrievalqa import (
    DATASET_NAME as RETRIEVALQA_DATASET_NAME,
    retrievalqa_answer_as_of,
)
from corvus.sources import SharedHostLimiter, retry_after_seconds
from scorers import (DEFAULT_JUDGE_MODEL, DETERMINISTIC_SCORERS,
                     iter_source_urls, make_jury_grader,
                     make_simpleqa_grader)

# ---------------------------------------------------------------------------
# Credentials: values from .env always override ambient credentials.
# ---------------------------------------------------------------------------


RUNTIME_ENV_NAMES = (
    "BRAINTRUST_API_KEY",
    "BRAINTRUST_PROJECT_ID",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BASETEN_API_KEY",
    "YDC_API_KEY",
    "JUDGE_API_KEY",
    # Serving path. Setting BRAINTRUST_GATEWAY_URL routes every vendor through
    # the Braintrust gateway, which then supplies the vendor credentials from the
    # org rather than from the three keys above. See agents.gateway_config.
    agents.GATEWAY_URL_ENV,
    agents.GATEWAY_KEY_ENV,
    agents.GATEWAY_PROJECT_ENV,
    agents.GATEWAY_ORG_ENV,
)


def load_runtime_env(env_path: Path) -> tuple[str, str]:
    """Load all supported credentials only from .env, overriding the ambient."""
    env = load_env(env_path)
    for name in RUNTIME_ENV_NAMES:
        if env.get(name):
            os.environ[name] = env[name]
        else:
            os.environ.pop(name, None)
    api_key = env.get("BRAINTRUST_API_KEY")
    project_id = env.get("BRAINTRUST_PROJECT_ID")
    if not api_key or not project_id:
        raise SystemExit(
            f"{env_path}: BRAINTRUST_API_KEY and BRAINTRUST_PROJECT_ID are required"
        )
    return api_key, project_id


# ---------------------------------------------------------------------------
# Frozen-agent constants. Changing ANY of these is a new experiment condition.
# ---------------------------------------------------------------------------

# Default agent model per vendor lives in agents.VENDORS. This constant is the
# fallback for --model-vendor openai, kept so existing invocations that pass
# --agent-model explicitly behave unchanged.
DEFAULT_MODEL_VENDOR = "openai"
AGENT_MODEL = VENDORS[DEFAULT_MODEL_VENDOR].default_model
MAX_SEARCHES, MAX_CLICKS = 5, 0
# Highlights are passed through whole. Recorded as a sentinel so harness rows
# stay distinct from native rows, where the field does not apply; rows written
# before the switch to highlights carry the integer 400 under `snippet_chars`.
SNIPPET_TRUNCATION = "none"
# You.com applies `count` independently to its web and news sections. Five per
# section therefore gives the registered harness/native contrast a target
# surface of up to 10 results per search: 5 web + 5 news when both sections are
# available. Native providers do not expose section-specific result counts, so
# their observed surface size is recorded rather than post-hoc truncated.
N_RESULTS = 5                          # per section; `wide` overrides it
NATIVE_RESULT_COUNT_TARGET = 2 * N_RESULTS

# Pin the overlapping location control so requests do not inherit the machine,
# gateway, or provider account's geography. You.com also exposes language and
# SafeSearch controls without native equivalents; pin and declare their values
# rather than relying on defaults that can change.
SEARCH_USER_LOCATION = agents.SEARCH_USER_LOCATION
YDC_LANGUAGE = "en"
YDC_SAFESEARCH = "moderate"

# Alternate web and news by within-section rank. The sections carry no
# cross-comparable score, so alternating is the neutral merge; concatenating
# would pin every news result below every web result and bury the freshest
# coverage on a benchmark that is mostly news.
YDC_MERGE_POLICY = "interleave"

# ---------------------------------------------------------------------------
# You.com setups — the harness treatment axis.
#
# You.com is the only search API here. That makes the SETUP the treatment: each
# entry below is one declared retrieval configuration, and `--arm` selects it.
# Everything not named in a setup is held constant (snippet budget, exclusion
# list, search budget), so a difference between two setups is attributable to the
# parameters that differ.
#
# Because there is one API, no result here can distinguish "You.com beats native"
# from "independent APIs beat native" — see docs/study-design.md.
#
# Documented parameters (you.com/docs/api-reference/search, checked 2026-07-30):
#   count      results per call, does NOT affect price
#   freshness  day | week | month | year | YYYY-MM-DDtoYYYY-MM-DD
# `freshness` is never derived from a row's event_date — that would leak the
# label into retrieval.
# ---------------------------------------------------------------------------

YDC_SETUPS: dict[str, dict] = {
    # Baseline: no freshness filter, so the index decides recency on its own.
    "normalized": {"count": N_RESULTS, "freshness": None},
    # The declared one-day freshness treatment.
    "native_fresh": {"count": N_RESULTS, "freshness": "day"},
    # Does the WIDTH of the freshness window matter, or just its presence? A
    # one-day filter can starve a query whose coverage lags by 48 hours.
    "fresh_week": {"count": N_RESULTS, "freshness": "week"},
    # A bigger decision surface at identical cost: You.com bills per call
    # independent of `count`, so if 20-per-section beats 5-per-section that is a free win. Tests
    # whether the 5-per-section baseline was leaving recall on the table.
    "wide": {"count": 20, "freshness": None},
}
DEFAULT_ARM = "normalized"


def ydc_setup(arm: str, answer_as_of: str | None = None) -> dict:
    try:
        setup = dict(YDC_SETUPS[arm])
    except KeyError:
        raise SystemExit(
            f"unknown --arm {arm!r}; expected one of {sorted(YDC_SETUPS)}") from None
    if answer_as_of and setup["freshness"] in ("day", "week"):
        end = date.fromisoformat(answer_as_of)
        start = end if setup["freshness"] == "day" else end - timedelta(days=6)
        setup["freshness"] = f"{start.isoformat()}to{end.isoformat()}"
        setup["freshness_reference"] = "answer_as_of"
    return setup


def experiment_ydc_setup(arm: str, dataset_name: str) -> dict:
    """Declare row-relative historical filtering without pretending it is `day`."""
    setup = ydc_setup(arm)
    if (dataset_name == RETRIEVALQA_DATASET_NAME
            and setup["freshness"] in ("day", "week")):
        setup["freshness"] = f"answer_as_of_{setup['freshness']}"
        setup["freshness_reference"] = "row_answer_as_of"
    return setup


def qualify_retrievalqa_question(
    question: str,
    metadata: dict,
    dataset_name: str,
) -> tuple[str, str | None, str | None]:
    """Anchor frozen dynamic QA labels to the date on which they were true."""
    if dataset_name != RETRIEVALQA_DATASET_NAME:
        return question, None, None
    answer_as_of, basis = retrievalqa_answer_as_of(metadata)
    if not answer_as_of:
        return question, None, None
    qualified = (
        f"{question}\n\nReference date: {answer_as_of}. Answer as of that date, "
        "not as of today. Interpret words such as latest, current, this week, "
        "and most recent relative to the reference date. When searching, "
        f"include the reference date ({answer_as_of}) in the query."
    )
    return qualified, answer_as_of, basis

FROZEN_SYSTEM_PROMPT = """\
You are a web research agent answering a time-sensitive factual question. You have one tool:
search_web(query). You may use at most 5 searches and may not fetch result pages.
Search results show rank, title, url, snippet, and
published date when available. Do not use search operators (such as site:, intitle:, inurl:, intext:, filetype:, before:, after:, OR, or quoted exact-match). Write the query as a plain natural-language phrase. When you know the answer, stop calling tools
and reply with ONLY the final answer, as concisely as possible. If you cannot
determine the answer within budget, reply exactly: I could not find this."""

# Control arm: no tools at all, so the agent answers from parametric memory.
# Without this baseline a provider's score is uninterpretable — you cannot tell
# retrieval quality from what the model already knew, and LiveNewsBench rows
# older than the agent's training cutoff are answerable with no search at all.
# Subtract this arm from every provider arm to get retrieval's marginal value.
# Required for EVERY model, not just the OSS one: a frontier native-search score
# with no parametric floor under it cannot be separated from recall.
NO_SEARCH_ARM = "no_search"
NO_SEARCH_SYSTEM_PROMPT = """\
You are answering a time-sensitive factual question from memory. You have no tools and no web
access. When you know the answer, reply with ONLY the final answer, as concisely
as possible. If you do not know, reply exactly: I could not find this."""

# Native (server-side) search arm. The tool contract genuinely differs — the
# model provider owns the search, so there is no per-call schema to describe and
# no way to state a result format. Budget and answer format are held identical to
# FROZEN_SYSTEM_PROMPT; the tool sentence is the only intentional difference, and
# it is a declared confound rather than an oversight (prompt_version records it).
NATIVE_SEARCH_SYSTEM_PROMPT = """\
You are a web research agent answering a time-sensitive factual question. You have built-in web
search. You may use at most 5 searches. Do not use search operators (such as site:, intitle:, inurl:, intext:, filetype:, before:, after:, OR, or quoted exact-match). Write the query as a plain natural-language phrase. When you know the answer, stop searching
and reply with ONLY the final answer, as concisely as possible. If you cannot
determine the answer within budget, reply exactly: I could not find this."""

PROMPT_VERSIONS = {
    SEARCH_MODE_HARNESS: "frozen-v3-no-operators",
    SEARCH_MODE_NONE: "no-search-v2-parametric",
    SEARCH_MODE_NATIVE: "native-search-v2-no-operators",
}

# The harness tool schema lives in agents.SEARCH_TOOL_* — one definition
# translated per wire protocol, so the OpenAI and Anthropic harness arms cannot
# drift into offering subtly different tools.

# You.com search pricing (https://you.com/docs/api-reference/search, checked
# 2026-07-30): $5.00 per 1,000 calls, independent of `count`.
#
# That independence is a load-bearing fact, not trivia: it means the `wide` setup
# requests 20 results per section for the same per-call price, so a recall gain
# there is free. It also
# means search spend across the harness setups is identical, which isolates the
# cost comparison to the native arms and the model tokens.
#
# livecrawl would add $1/1k pages and is left off in every setup: it populates
# `result.contents`, not the `snippets` field the decision surface is built from,
# so enabling it would add latency and cost without changing what is measured.
YDC_USD_PER_CALL = 0.005

SEARCH_PROVIDER = "youdotcom"
SEARCH_PROVIDER_KEY = "YDC_API_KEY"


# ---------------------------------------------------------------------------
# Search-operator detection.
#
# The harness arm's queries should be plain natural language. If one model
# emits `site:reuters.com Apple CEO` and another emits `Apple CEO`, a score
# difference is query-construction, not retrieval quality. The prompt forbids
# operators, but prompts are soft control — a model may still emit them.
#
# The harness does NOT alter the query: rewriting what the model decided to
# send is itself an intervention, and the logged query would no longer be what
# the model actually chose. Instead the harness detects operators, records them
# so the per-model violation rate is auditable in the spans, and passes the
# raw query through to You.com unchanged.
#
# Detect both `operator:term` forms and the additional syntax forbidden by the
# prompt: uppercase boolean operators, leading +/- terms, and quoted matches.
# A normal hyphenated word is not leading syntax and remains unflagged.
# ---------------------------------------------------------------------------

_SEARCH_OPERATOR_RE = re.compile(
    r"(?<!\w)(?:site|inurl|intitle|intext|inanchor|filetype|ext|"
    r"allintitle|allinurl|allintext|allinanchor|related|cache|define|"
    r"author|before|after|source|loc|location|language|num|filter|"
    r"safe|sort|date|info|link|blogurl)"
    r":(?:\"[^\"]*\"|\S+)",
    re.IGNORECASE,
)

_SEARCH_SYNTAX_RE = re.compile(
    r'(?<!\S)[+-](?=\S)\S+|\b(?:AND|OR|NOT)\b|"(?:[^"\\]|\\.)+"'
)


def _detect_search_operators(query: str) -> list[str]:
    """Return the list of search operators (site:, intitle:, etc.) in a query.

    Detection only — the query is never modified. The raw query is passed to
    the search layer unchanged; the detected operators are logged so the
    per-model violation rate is auditable.
    """
    if not isinstance(query, str):
        return []
    matches = [
        (m.start(), m.end(), m.group(0))
        for m in _SEARCH_OPERATOR_RE.finditer(query)
    ]
    for match in _SEARCH_SYNTAX_RE.finditer(query):
        if not any(start < match.end() and end > match.start()
                   for start, end, _ in matches):
            matches.append((match.start(), match.end(), match.group(0)))
    return [value for _, _, value in sorted(matches)]


def search_cost_usd(arm: str, n_results: int) -> float:
    """You.com bills per call, so the result count does not enter the price."""
    del arm, n_results  # documented to be price-irrelevant; kept for call-site clarity
    return YDC_USD_PER_CALL


ARCHIVE_EXCLUDES = ["web.archive.org", "archive.org", "archive.is",
                    "archive.ph", "archive.today"]


def _tok(text: str) -> int:
    """Cheap token estimate (chars/4). Swap in tiktoken for publication runs."""
    return max(1, len(text) // 4)


def _domain_of(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def source_domains_of(metadata: dict, expected) -> list[str]:
    """Domains that must never appear in the decision surface, read from the
    SAME fields scorers.leakage_guard checks: metadata['link'] plus every
    metadata['articles'][*].link/url. Falls back to a dict-shaped `expected`
    for datasets that predate the importer's schema.

    Ordered source-domains-first: Parallel caps exclude_domains at 10, so any
    truncation must drop archive mirrors, never a gold source.
    """
    seen, domains = set(), []
    for url in iter_source_urls(metadata, expected):
        host = _domain_of(url)
        if host and host not in seen:
            seen.add(host)
            domains.append(host)
    return domains


# ---------------------------------------------------------------------------
# Provider adapters. Each returns (normalized_results, raw_payload).
# Normalized schema: {rank, url, title, snippet, published_date} — the agent
# must not be able to tell providers apart from formatting.
#
# NB: no adapter may derive date parameters from the item's event_date (label
# leakage into retrieval), and queries must avoid temporal keywords — You.com
# takes the broader of query-implied vs parameter freshness, which would
# silently un-blind the arms.
# ---------------------------------------------------------------------------

_http = httpx.Client(
    timeout=httpx.Timeout(30.0, connect=10.0),
    follow_redirects=False,
    headers={"user-agent": "CorvusEval/0.1"},
)
_provider_limiters: dict[str, SharedHostLimiter] = {}
DEFAULT_YDC_REQUESTS_PER_SECOND = 1.0
MAX_YDC_REQUESTS_PER_SECOND = 10.0


def _provider_json(method: str, url: str, *, timing: dict | None = None,
                   requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND,
                   **kwargs):
    """Rate-limited provider request with bounded retry/backoff."""
    host = (urlparse(url).hostname or "").lower()
    approved_hosts = {"ydc-index.io"}
    if host not in approved_hosts:
        raise ValueError(f"unapproved provider API host: {host!r}")
    if not 0 < requests_per_second <= MAX_YDC_REQUESTS_PER_SECOND:
        raise ValueError(
            "requests_per_second must be greater than zero and no more than "
            f"{MAX_YDC_REQUESTS_PER_SECOND:g}"
        )
    interval = 1.0 / requests_per_second
    limiter = _provider_limiters.get(host)
    if limiter is None:
        limiter = SharedHostLimiter(f"provider-{host}", interval)
        _provider_limiters[host] = limiter
    elif not math.isclose(limiter.interval, interval):
        raise ValueError(
            f"conflicting request rates configured for {host}: "
            f"existing={1.0 / limiter.interval:g}/s, "
            f"requested={requests_per_second:g}/s"
        )
    timing = timing if timing is not None else {}
    timing.setdefault("rate_limit_wait_s", 0.0)
    timing.setdefault("provider_http_s", 0.0)
    timing.setdefault("retry_backoff_s", 0.0)
    timing.setdefault("provider_retry_count", 0)
    timing.setdefault("provider_429_count", 0)
    for attempt in range(5):
        wait_started = time.perf_counter()
        limiter.wait()
        timing["rate_limit_wait_s"] += time.perf_counter() - wait_started
        request_started = time.perf_counter()
        response = _http.request(method, url, **kwargs)
        timing["provider_http_s"] += time.perf_counter() - request_started
        if 300 <= response.status_code < 400:
            raise ValueError("provider API redirect refused")
        if (response.url.host or "").lower() not in approved_hosts:
            raise ValueError("provider API redirected to an unapproved host")
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError(f"{url}: expected a JSON object")
            return value
        if attempt == 4:
            response.raise_for_status()
        timing["provider_retry_count"] += 1
        if response.status_code == 429:
            timing["provider_429_count"] += 1
        delay = max(retry_after_seconds(response.headers.get("Retry-After")),
                    min(60.0, 5.0 * (2 ** attempt)))
        timing["retry_backoff_s"] += delay
        print(
            f"Provider {host} returned HTTP {response.status_code}; "
            f"retrying in {delay:g}s (attempt {attempt + 1}/5).",
            file=sys.stderr,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def _interleave(web: list, news: list) -> list:
    """Alternate two independently ranked sections, appending the longer tail.

    web[0], news[0], web[1], news[1], ... Neither section exposes a score the
    other can be compared against, so alternating by within-section rank is the
    neutral merge; whichever section is longer contributes the remainder.
    """
    merged = []
    for pair in zip_longest(web, news):
        merged.extend(hit for hit in pair if hit is not None)
    return merged


def youdotcom_search(query: str, arm: str, exclude_domains: list[str],
                     setup: dict | None = None, timing: dict | None = None,
                     requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND):
    # POST https://ydc-index.io/v1/search — POST is the documented path for
    # new features. extraction_mode: "highlights" returns query-aware passages
    # purpose-built for agent grounding, and is only available on POST. GET
    # still works but receives no new feature updates.
    #
    # exclude_domains is a JSON array on POST (up to 500), mutually exclusive
    # with include_domains (sending both returns 422).
    setup = setup or ydc_setup(arm)
    body: dict = {
        "query": query,
        "count": setup["count"],
        "country": SEARCH_USER_LOCATION["country"],
        "language": YDC_LANGUAGE,
        "safesearch": YDC_SAFESEARCH,
        # Highlights are token-efficient passages from each page most relevant to the
        # query. They are designed to replace snippets and
        # are free — only full_page extraction carries a per-page charge.
        "extraction": {"extraction_mode": "highlights"},
    }
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    if setup["freshness"] is not None:
        # day|week|month|year or YYYY-MM-DDtoYYYY-MM-DD. Never derived from a
        # row's event_date, which would leak the label into retrieval.
        body["freshness"] = setup["freshness"]
    raw = _provider_json(
        "POST",
        "https://ydc-index.io/v1/search",
        timing=timing,
        requests_per_second=requests_per_second,
        json=body,
        headers={
            "X-API-Key": os.environ["YDC_API_KEY"],
            # This is a client cache directive, not a guarantee that You.com
            # bypasses its index or extraction caches.
            "Cache-Control": "no-cache",
        },
    )
    # Both sections, with news ADDITIVE: `count` is applied per section, so a
    # news-intent query surfaces up to 2x it. Two of the three datasets are news
    # benchmarks and news is the only section reporting a true publication
    # timestamp, so it is on-target retrieval rather than overflow. Surface size
    # therefore varies per query, not per arm; n_web_results / n_news_results
    # record it. See docs/study-design.md.
    results_obj = raw.get("results") or {}
    web_hits = [("web", h) for h in results_obj.get("web") or []]
    news_hits = [("news", h) for h in results_obj.get("news") or []]
    results = []
    section_positions = {"web": 0, "news": 0}
    for i, (source, res) in enumerate(_interleave(web_hits, news_hits), start=1):
        section_positions[source] += 1
        # Highlights replace `snippets` entirely when extraction is requested.
        # News results carry no `contents` at all, so they land on description.
        contents = res.get("contents") or {}
        highlights = contents.get("highlights") or []
        snippets = res.get("snippets") or []
        if highlights:
            snippet = "\n".join(highlights)
        elif snippets:
            snippet = "\n".join(snippets)
        else:
            snippet = res.get("description", "") or ""
        results.append({
            "rank": i,
            # rank is the harness's display order after interleaving. Only
            # section_rank retains the provider's within-section ordering.
            "section_rank": section_positions[source],
            "url": res.get("url", ""),
            "title": res.get("title") or "",
            "snippet": snippet,
            "published_date": res.get("page_age"),
            "date_semantics": (
                "publication" if source == "news"
                else "provider_page_age_unverified"),
            # Preserve the section because You.com ranks web and news
            # independently and does not expose a cross-section score.
            "source": source,
        })
    return results, raw


def _provider_request_id(raw: dict) -> str | None:
    """You.com's safe correlation ID, without retaining the raw payload."""
    return (raw.get("metadata") or {}).get("search_uuid")


# ---------------------------------------------------------------------------
# Traced tools. type="tool" spans; native metrics without raw payload retention.
# notrace_io=True on both: @traced otherwise auto-logs the return value over
# the explicit output= below, and these return tuples, not the payload we want.
# ---------------------------------------------------------------------------

@traced(type="tool", name="search_web", notrace_io=True)
def run_search(arm: str, query: str, exclude_domains: list[str],
               setup: dict | None = None,
               requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND):
    setup = setup or ydc_setup(arm)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    results, raw = youdotcom_search(
        query, arm, exclude_domains, setup, timing, requests_per_second
    )
    latency = time.perf_counter() - t0
    rendered = "\n".join(
        f"[{r['rank']}] {r['title']}\n    {r['url']}\n    "
        f"published: {r['published_date'] or 'unknown'}\n    {r['snippet']}"
        for r in results) or "No results."
    # Split web vs news: the news section is where the freshest results live, so
    # a row carrying only web results may be missing the most time-sensitive
    # coverage the API returned.
    results_obj = raw.get("results") or {}
    n_web = len(results_obj.get("web") or [])
    n_news = len(results_obj.get("news") or [])
    current_span().log(
        input={"query": query, "provider": SEARCH_PROVIDER, "arm": arm,
               "exclude_domains": exclude_domains},
        output=results,
        metadata={"provider": SEARCH_PROVIDER, "arm": arm,
                  # The setup's resolved parameters, so a span is interpretable
                  # without cross-referencing YDC_SETUPS at the run's commit.
                  "ydc_count": setup["count"],
                  "ydc_freshness": setup["freshness"],
                  "ydc_extraction_mode": "highlights",
                  "ydc_merge_policy": YDC_MERGE_POLICY,
                  "search_user_location": SEARCH_USER_LOCATION,
                  "ydc_language": YDC_LANGUAGE,
                  "ydc_safesearch": YDC_SAFESEARCH,
                  "ydc_requests_per_second": requests_per_second,
                  "provider_request_id": _provider_request_id(raw),
                  "raw_payload_retained": False},
        metrics={"tokens": _tok(rendered),
                 "latency_s": latency,
                 "provider_http_latency_s": timing.get("provider_http_s", 0.0),
                 "rate_limit_wait_s": timing.get("rate_limit_wait_s", 0.0),
                 "retry_backoff_s": timing.get("retry_backoff_s", 0.0),
                 "provider_retry_count": timing.get("provider_retry_count", 0),
                 "provider_429_count": timing.get("provider_429_count", 0),
                 "search_cost_usd": search_cost_usd(arm, len(results)),
                 "n_results": len(results),
                 # Per section; both reach the agent, so n_results is their sum.
                 "n_web_results": n_web,
                 "n_news_results": n_news,
                 # Per section, NOT a total: compare against n_web_results and
                 # n_news_results separately, never against n_results. On `wide`
                 # a shortfall is the finding, not noise.
                 "n_results_requested_per_section": setup["count"]},
    )
    return results, rendered, _tok(rendered)


# ---------------------------------------------------------------------------
# Frozen agent loop
# ---------------------------------------------------------------------------

# Built lazily: constructing a vendor client at import time raises on a missing
# key, which would crash `--help` and preempt _preflight's clear message.
def get_agent_client(vendor: str):
    """Traced client for one vendor. wrap_anthropic keeps Claude's Messages-API
    calls in the same span tree as the OpenAI-compatible ones, so token and cost
    rollups are comparable across arms."""
    if vendor == "anthropic":
        from braintrust import wrap_anthropic

        return agents.get_client(vendor, wrap_anthropic)
    return agents.get_client(vendor, wrap_openai)


def _system_prompt_for(search_mode: str) -> str:
    if search_mode == SEARCH_MODE_NONE:
        return NO_SEARCH_SYSTEM_PROMPT
    if search_mode == SEARCH_MODE_NATIVE:
        return NATIVE_SEARCH_SYSTEM_PROMPT
    return FROZEN_SYSTEM_PROMPT


def condition_label(search_mode: str, arm: str, model_vendor: str) -> str:
    """One slug per matrix cell. Distinguishes the two collision-prone names:
    `arm=native_fresh` is a SEARCH VENDOR's freshness parameter, while
    `search_mode=native` is the MODEL vendor's own server-side search."""
    if search_mode == SEARCH_MODE_NONE:
        return "no_search"
    if search_mode == SEARCH_MODE_NATIVE:
        return f"native-{model_vendor}"
    return f"harness-{SEARCH_PROVIDER}-{arm}"


def make_task(
    arm: str,
    agent_model: str = AGENT_MODEL,
    study_id: str = "freshness-v1",
    dataset_name: str = DATASET_NAME,
    dataset_version: str | None = None,
    search_mode: str = SEARCH_MODE_HARNESS,
    model_vendor: str = DEFAULT_MODEL_VENDOR,
    ydc_requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND,
):
    spec = vendor_of(model_vendor)
    system_prompt = _system_prompt_for(search_mode)
    condition = condition_label(search_mode, arm, model_vendor)
    condition_id = f"{model_vendor}:{agent_model}::{condition}"

    def task(input: dict, hooks) -> dict:
        # Leakage excludes come from row METADATA (link + articles[*]), which is
        # where the importer puts them and where leakage_guard reads them.
        # `expected` is the answer string, so it has no .get().
        row_metadata = hooks.metadata or {}
        source_domains = source_domains_of(row_metadata, hooks.expected)
        excludes = source_domains + ARCHIVE_EXCLUDES
        original_question = input["question"]
        question, answer_as_of, answer_as_of_basis = qualify_retrievalqa_question(
            original_question, row_metadata, dataset_name)
        search_setup = ydc_setup(arm, answer_as_of)
        task_key = hashlib.sha256(original_question.encode("utf-8")).hexdigest()[:16]
        # Category is load-bearing for reporting, not decoration: the closest
        # published comparison saw this effect swing from ~+6 points to zero
        # across two domains, so a pooled number can hide a sign change. The
        # Corvus-QA fallbacks are here because those rows carry none of the
        # LiveNewsBench category fields and were all landing in "uncategorized",
        # which would have made the second domain unsliceable.
        benchmark_category = (
            row_metadata.get("category")
            or row_metadata.get("event_category")
            or row_metadata.get("data_source")
            or row_metadata.get("attribute")        # Corvus-QA: ceo_of, etc.
            or row_metadata.get("entity_type")
            or "uncategorized"
        )

        client = get_agent_client(model_vendor)
        # Wall-clock per row. Latency is not derivable from search count: a layer
        # issuing more but faster calls can finish ahead of one issuing fewer slow
        # ones, so speed has to be measured rather than inferred. Per-search
        # latency already lands on the tool spans; this is the total a user would
        # actually wait, model turns included.
        t0 = time.perf_counter()
        if search_mode == SEARCH_MODE_NATIVE:
            outcome = _run_native(client, spec, agent_model, system_prompt,
                                  question, excludes)
        else:
            outcome = _run_harness(client, spec, agent_model, system_prompt,
                                   question, excludes, arm, search_mode,
                                   search_setup, ydc_requests_per_second)
        wall_clock_s = time.perf_counter() - t0

        surfaced = agents.surfaced_domains(outcome["trajectory"])
        # The confound that quietly voids a search arm: the tool was available
        # and the model never used it, so the row is a no-search row wearing a
        # search arm's label. Weak OSS tool-calling and a native model that
        # decides not to search both land here. Filter on it in analysis — do
        # not average it away.
        search_available = search_mode != SEARCH_MODE_NONE
        zero_search_row = bool(search_available and outcome["used_searches"] == 0)
        # The sibling confound to zero_search_row: the model DID search, but the
        # search layer failed, so the row carries a search arm's label while
        # having been served less retrieval than the condition specifies. A
        # partly-degraded row still answers, and the judge still scores that
        # answer, so without this flag a provider outage reads as the model
        # getting worse. Filter on it in analysis the same way — do not average
        # it away.
        search_errors = outcome["search_errors"]
        search_degraded = bool(search_errors)
        # Stronger case: every search the model attempted failed, so the row
        # received no retrieval at all and its answer is parametric. That makes
        # it a no-search row wearing a search arm's label, exactly what
        # zero_search_row catches for the never-tried case.
        search_fully_failed = bool(
            search_available and outcome["used_searches"] > 0
            and not outcome["trajectory"])

        hooks.metadata.update({
            # --- matrix axes: the four fields any slice should key on ---
            "model_class": spec.model_class,
            "model_vendor": model_vendor,
            "search_mode": search_mode,
            "search_provider": (
                SEARCH_PROVIDER if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else "none"
            ),
            # Freshness treatment applies to harness arms only; None elsewhere
            # keeps `native_fresh` from being read as "native search".
            "freshness_treatment": (
                arm if search_mode == SEARCH_MODE_HARNESS else None),
            # --- kept for continuity with runs made before the axes existed ---
            "provider": (
                SEARCH_PROVIDER if search_mode == SEARCH_MODE_HARNESS else "none"
                if search_mode == SEARCH_MODE_NONE else f"{model_vendor}_native"),
            "arm": arm,
            # The resolved You.com setup, so a row is interpretable without
            # reading YDC_SETUPS at the run's commit.
            "ydc_setup": (
                search_setup if search_mode == SEARCH_MODE_HARNESS
                else None),
            "ydc_requests_per_second": (
                ydc_requests_per_second
                if search_mode == SEARCH_MODE_HARNESS else None),
            "answer_as_of": answer_as_of,
            "answer_as_of_basis": answer_as_of_basis,
            "temporal_qualification_applied": answer_as_of is not None,
            "effective_question": question,
            "agent_model": agent_model,
            "study_id": study_id,
            "condition_id": condition_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "trial_index": hooks.trial_index,
            "task_key": task_key,
            "benchmark_category": benchmark_category,
            # --- subgroup variables for stratified reporting ---
            # Promoted to first-class row metadata because they answer the
            # freshness question directly and only Corvus-QA supplies them:
            # recency_rung buckets how recently the fact changed, and
            # coverage_tier records whether a pinned reference search could find
            # the answer at all. The latter is a headroom control: a row no search
            # engine can answer measures the ceiling, not the provider.
            "recency_rung": row_metadata.get("recency_rung"),
            "coverage_tier": row_metadata.get("coverage_tier"),
            "answer_class": row_metadata.get("answer_class"),
            "dataset_family": row_metadata.get("dataset") or (
                "LiveNewsBench" if row_metadata.get("livenewsbench_release")
                else "unknown"),
            # --- observability declarations the scorers gate on ---
            "decision_surface": outcome["decision_surface"],
            "exclusion_requested": outcome["exclusion_requested"],
            "excluded_source_domains": source_domains,
            "search_budget": MAX_SEARCHES,
            # --- per-row integrity flags ---
            "zero_search_row": zero_search_row,
            "search_degraded": search_degraded,
            "search_fully_failed": search_fully_failed,
            "bad_tool_calls": outcome["bad_tool_calls"],
            "operator_violations": outcome["operator_violations"],
            "refused_searches": outcome["refused_searches"],
            "refused_clicks": outcome["refused_clicks"],
            "search_errors": search_errors,
            "model_refused": outcome["refused"],
            "answer_truncated": outcome["truncated"],
            "pause_turns": outcome["pause_turns"],
            "native_actions": outcome["native_actions"],
            "native_emitted_queries": outcome["emitted_queries"],
            "response_model": outcome["response_model"],
            "as_of": datetime.now(timezone.utc).isoformat(),
        })
        # Task-span metrics; Braintrust rolls these into the experiment summary.
        # search_cost_usd is search spend ONLY — model inference cost comes from
        # the wrapped LLM spans, so keep the two decomposable rather than summing
        # them here. Total cost per row is an analysis-side join of the two.
        final = outcome["final_answer"]
        # Search fees are the small half of the bill. Reporting them alone would
        # rank the arms on the wrong quantity, so the row carries both halves and
        # their sum. model_cost_usd is None (not 0.0) for an unpriced model, which
        # keeps that arm out of a cost frontier instead of placing it at the
        # origin — where it would read as free.
        inference_cost, cost_confirmed = agents.model_cost_usd(
            agent_model, outcome["prompt_tokens"], outcome["completion_tokens"],
            outcome["cached_prompt_tokens"])
        cost_metrics = {"search_cost_usd": outcome["search_cost"]}
        if inference_cost is not None:
            cost_metrics["model_cost_usd"] = inference_cost
            cost_metrics["total_cost_usd"] = inference_cost + outcome["search_cost"]
            # What fraction of the bill the search layer actually is. Logged per
            # row so the assumption that inference dominates stays checkable on
            # this data instead of being carried as a premise.
            total = inference_cost + outcome["search_cost"]
            cost_metrics["search_share_of_cost"] = (
                outcome["search_cost"] / total if total else 0.0)
        hooks.metadata["model_cost_confirmed"] = cost_confirmed
        current_span().log(metrics={
            **cost_metrics,
            "latency_s": wall_clock_s,
            "used_searches": outcome["used_searches"],
            "native_search_actions": outcome["native_search_actions"],
            "native_tool_calls": outcome["native_tool_calls"],
            "native_emitted_query_count": len(outcome["emitted_queries"]),
            "native_open_page_actions": sum(
                action.get("type") == "open_page"
                for action in outcome["native_actions"]),
            "native_find_actions": sum(
                action.get("type") == "find_in_page"
                for action in outcome["native_actions"]),
            "used_clicks": outcome["used_clicks"],
            "refused_tool_calls": (
                outcome["refused_searches"] + outcome["refused_clicks"]),
            "agent_prompt_tokens": outcome["prompt_tokens"],
            # Cached input bills at a fraction of the base rate, and OpenAI
            # caches automatically on every turn after the first. Logged so a
            # cost difference between arms can be traced to cache behavior
            # rather than read as a difference in work done.
            "agent_cached_prompt_tokens": outcome["cached_prompt_tokens"],
            "agent_cache_hit_rate": (
                outcome["cached_prompt_tokens"] / outcome["prompt_tokens"]
                if outcome["prompt_tokens"] else 0.0),
            "agent_completion_tokens": outcome["completion_tokens"],
            "agent_total_tokens": (
                outcome["prompt_tokens"] + outcome["completion_tokens"]),
            "answer_words": len(final.split()),
            "answer_chars": len(final),
            "distinct_surfaced_domains": len(surfaced),
            "zero_search_row": int(zero_search_row),
            "n_operator_violations": len(outcome["operator_violations"]),
            "n_search_errors": len(search_errors),
            "search_degraded": int(search_degraded),
            "search_fully_failed": int(search_fully_failed),
        })
        return {
            "final_answer": final,
            "trajectory": outcome["trajectory"],
            # Consumed by scorers.py to decide which metrics are computable on
            # this row. Without it a native arm's empty snippet layer reads as a
            # clean pass rather than an unobservable one.
            "decision_surface": outcome["decision_surface"],
            "used_searches": outcome["used_searches"],
            "used_clicks": outcome["used_clicks"],
            "refused_searches": outcome["refused_searches"],
            "refused_clicks": outcome["refused_clicks"],
            "citations": outcome["citations"],
            "agent_prompt_tokens": outcome["prompt_tokens"],
            "agent_completion_tokens": outcome["completion_tokens"],
        }

    return task


def _search_error_code(exc: httpx.HTTPError) -> str:
    """Normalize a search failure onto the native arms' `error_code` shape.

    An HTTP status is the useful discriminator (429 and 5xx mean the provider
    was overloaded and the row is retryable; 4xx means the request was wrong and
    every row will fail the same way), so it is preferred over the exception
    class. Transport failures have no status, so they fall back to the class
    name rather than a fabricated code.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return str(status) if status is not None else type(exc).__name__


def _blank_outcome(decision_surface: str) -> dict:
    return {
        "final_answer": "", "trajectory": [], "used_searches": 0,
        "used_clicks": 0, "refused_searches": 0, "refused_clicks": 0,
        "bad_tool_calls": 0, "search_cost": 0.0, "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0, "decision_surface": decision_surface,
        "exclusion_requested": False, "citations": [], "search_errors": [],
        "refused": False, "truncated": False, "pause_turns": 0,
        "operator_violations": [],
        "native_actions": [], "emitted_queries": [],
        "native_search_actions": 0, "native_tool_calls": 0,
        "response_model": None,
    }


def _run_harness(client, spec, agent_model, system_prompt, question, excludes,
                 arm, search_mode, search_setup: dict | None = None,
                 ydc_requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND) -> dict:
    """Tool-calling loop for the harness arm and the no-tool control arm.

    Identical driver for both: the control arm is this loop with the tool never
    offered, so the two differ only in tool availability.
    """
    out = _blank_outcome(
        SURFACE_FULL if search_mode == SEARCH_MODE_HARNESS else SURFACE_NONE)
    out["exclusion_requested"] = (
        search_mode == SEARCH_MODE_HARNESS and bool(excludes))
    tools_allowed = search_mode == SEARCH_MODE_HARNESS

    session = make_harness_session(client, spec, agent_model, system_prompt)
    session.add_user(question)
    final = None

    for _ in range(2 * (MAX_SEARCHES + MAX_CLICKS) + 2):
        # Once both budgets are spent there is nothing left to call, so drop
        # the tools and force a final answer instead of burning turns on
        # "budget exhausted" replies until the loop cap trips.
        out_of_budget = (out["used_searches"] >= MAX_SEARCHES
                         and out["used_clicks"] >= MAX_CLICKS)
        turn = session.step(tools_enabled=tools_allowed and not out_of_budget)
        if turn.response_model:
            out["response_model"] = turn.response_model
        out["prompt_tokens"] += turn.prompt_tokens
        out["cached_prompt_tokens"] += turn.cached_prompt_tokens
        out["completion_tokens"] += turn.completion_tokens
        if turn.refused:
            out["refused"] = True
            break
        if turn.truncated:
            out["truncated"] = True
        if not turn.tool_calls:
            final = turn.text
            break

        results_to_send = []
        for call in turn.tool_calls:
            if call["malformed"]:
                out["bad_tool_calls"] += 1
            if call["name"] == agents.SEARCH_TOOL_NAME:
                arguments = call.get("arguments")
                query = (arguments.get("query", "")
                         if isinstance(arguments, dict) else None)
                if call["malformed"] or not isinstance(query, str):
                    if not call["malformed"]:
                        out["bad_tool_calls"] += 1
                    content = (
                        "Invalid search query: query must be a string. "
                        "Call search_web again with a plain natural-language phrase."
                    )
                elif out["used_searches"] >= MAX_SEARCHES:
                    out["refused_searches"] += 1
                    content = "Search budget exhausted."
                else:
                    # Incremented BEFORE the call, so a failed search still
                    # spends budget. The native arms already work this way ("an
                    # error round still spent budget"), and not charging for a
                    # failure would hand a flaky provider unlimited retries
                    # inside the turn cap.
                    out["used_searches"] += 1
                    detected_ops = _detect_search_operators(query)
                    if detected_ops:
                        out["operator_violations"].append({
                            "query": query,
                            "operators": detected_ops,
                        })
                    try:
                        results, content, tok = run_search(
                            arm, query, excludes, search_setup,
                            ydc_requests_per_second)
                    except httpx.HTTPError as exc:
                        # The provider failed after _provider_json's retries.
                        # Recorded and survivable rather than fatal: killing the
                        # row would drop it from the dataset entirely, and rows
                        # do not drop at random — a provider fails hardest on the
                        # queries it handles worst, so the surviving rows would
                        # be a favorable subset of the ones actually asked.
                        #
                        # Only httpx.HTTPError is caught. The ValueErrors
                        # _provider_json raises are integrity guards (refused
                        # redirect, unapproved host, non-object JSON) — those
                        # mean the environment is wrong, not that one search
                        # failed, and they must stop the run rather than be
                        # logged as routine.
                        out["search_errors"].append(
                            {"query": query, "error_code": _search_error_code(exc)})
                        content = ("Search failed: the search API returned an "
                                   "error for this query and no results are "
                                   "available.")
                    else:
                        # Billed only on success. A request that errored out was
                        # not a served search — the same rule the vendors apply
                        # to their own server-side search.
                        #
                        # Accumulated per call rather than computed as
                        # searches x rate at the end, so a mid-row failure still
                        # bills what it actually spent. You.com prices per call
                        # regardless of `count`, so the two agree today; keeping
                        # the accumulation means a per-result price would not
                        # silently under-report.
                        out["search_cost"] += search_cost_usd(
                            arm, len(results))
                        out["trajectory"].append({
                            "type": "search", "query": query,
                            "tokens": tok, "results": results})
            else:
                out["bad_tool_calls"] += 1
                content = (f"Unknown tool: {call['name']}. "
                           f"Use {agents.SEARCH_TOOL_NAME}.")
            results_to_send.append((call["id"], content))
        session.add_tool_results(results_to_send)

    out["final_answer"] = final if final else "I could not find this."
    return out


def _run_native(client, spec, agent_model, system_prompt, question,
                excludes) -> dict:
    """Server-side search arm — the model provider runs the search."""
    if spec.name == "anthropic":
        run = agents.anthropic_native_search(
            client, agent_model, system_prompt, question, excludes,
            MAX_SEARCHES)
    elif spec.name == "openai":
        run = agents.openai_native_search(
            client, agent_model, system_prompt, question, excludes,
            # Same effort the harness arm sends, so this vendor's native-vs-
            # harness contrast varies only where the search happens.
            spec.reasoning_effort, max_searches=MAX_SEARCHES)
    else:
        raise SystemExit(
            f"--search-mode native is unavailable for --model-vendor {spec.name}: "
            "no server-side search exists for this vendor.")

    rate, _confirmed = native_search_rate_usd(spec.name, agent_model)
    out = _blank_outcome(run.surface)
    out.update({
        "final_answer": run.final_answer or "I could not find this.",
        "trajectory": run.trajectory,
        # For native arms this is the provider-enforced budget unit. OpenAI
        # counts every built-in tool action; Anthropic counts web searches.
        # Separate fields retain search-action and subquery counts.
        "used_searches": run.vendor_search_count or run.n_searches,
        # Native tools enforce their budget server-side, so a per-call refusal
        # count is not observable. budget_economy reads used_searches.
        "search_cost": rate * (
            run.billable_searches
            if run.billable_searches is not None else run.n_searches),
        "prompt_tokens": run.prompt_tokens,
        "cached_prompt_tokens": run.cached_prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "exclusion_requested": run.exclusion_requested,
        "citations": run.citations,
        "search_errors": run.search_errors,
        "refused": run.refused,
        "truncated": run.truncated,
        "pause_turns": run.pause_turns,
        "native_actions": run.native_actions,
        "emitted_queries": run.emitted_queries,
        "native_search_actions": run.n_searches,
        "native_tool_calls": run.vendor_search_count or run.n_searches,
        "response_model": run.response_model,
        "operator_violations": [
            {"query": query, "operators": operators}
            for query in run.emitted_queries
            if (operators := _detect_search_operators(query))
        ],
    })
    return out


# ---------------------------------------------------------------------------
# Eval run. Dataset push lives in import_livenewsbench.py — a second pusher
# here produced an incompatible schema (link/event_date under `expected`, no
# livenewsbench_release) that silently disabled leakage_guard and
# temporal_grounding. Do not reintroduce it.
# ---------------------------------------------------------------------------

def select_rows(
    dataset,
    split: str | None,
    limit: int | None,
    count_full: bool = False,
):
    """Deterministically subset a dataset, and fingerprint what was selected.

    Pilots and cost-bounded runs need a subset, but a subset is only usable if
    EVERY arm sees the same rows — all contrasts are paired by task_key, so two
    arms drawn from different subsets silently lose their pairing and the drop
    shows up as missing data rather than as an error.

    Determinism therefore comes from sorting by row id before slicing, not from
    dataset iteration order, which is not contractually stable. `subset_id` is a
    hash of the selected ids: two runs claiming to be comparable must show the
    same value, which makes a pairing mistake checkable instead of assumed.
    """
    if split is None and limit is None and not count_full:
        return dataset, {"subset_applied": False, "split": None,
                         "limit": None, "n_rows": None, "n_available": None,
                         "subset_id": None}

    full_dataset = split is None and limit is None
    rows = list(dataset)
    n_available = len(rows)
    if split is not None:
        rows = [r for r in rows
                if ((r.get("metadata") or {}).get("livenewsbench_split") == split
                    or (r.get("metadata") or {}).get("corvus_split") == split)]
        if not rows:
            raise SystemExit(
                f"--split {split!r} matched no rows out of {n_available}")
    rows.sort(key=lambda r: str(r.get("id", "")))
    if limit is not None:
        rows = rows[:limit]
    ids = [str(r.get("id", "")) for r in rows]
    # A Braintrust Dataset is re-iterable. Keep passing the original object on
    # full runs so Eval retains dataset-origin linkage; the materialized copy is
    # used only to count and fingerprint the pinned contents.
    selected_data = dataset if full_dataset else rows
    return selected_data, {
        "subset_applied": split is not None or limit is not None,
        "split": split,
        "limit": limit,
        "n_rows": len(rows),
        "n_available": n_available,
        "subset_id": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Gateway preflight
#
# The gateway can fail in ways a direct vendor call cannot, and all of them are
# cheaper to hit before a run than on row 1 of N:
#   * the token belongs to an org that has no provider configured for the model
#     (404, gateway-origin) — the common case when a key is copied from the wrong
#     Braintrust org
#   * the token is not authorized for the org named in x-bt-org-name (401)
#   * the org's stored vendor key is stale (403, vendor-origin)
# The x-bt-error-origin response header separates gateway-side from vendor-side
# failures, which is the difference between "fix your Braintrust settings" and
# "fix your vendor account", so the message repeats it verbatim.
# ---------------------------------------------------------------------------


def _gateway_probe(gw, spec, agent_model: str):
    """One minimal request over the wire this vendor's arm will use.

    Carries the same sampling and reasoning knobs the real arm sends, not a bare
    "hello". A proxy that rejects `temperature` on the OSS row, or `thinking` /
    `reasoning.effort` on a frontier row, would otherwise pass a stripped-down
    probe and then fail every row of the run — and on the OSS row a silently
    *dropped* temperature is worse than a rejected one, because it unpins the
    only sampling parameter this study manages to pin.

    Returns the httpx.Response. Uses httpx rather than the SDK clients on
    purpose: the SDKs raise typed exceptions that discard the response headers
    this needs, and a preflight should read the failure, not re-raise it.
    """
    headers = {"Authorization": f"Bearer {gw.api_key}",
               "Content-Type": "application/json", **gw.headers()}
    if spec.name == "anthropic":
        url = f"{gw.anthropic_base_url}/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        body = {"model": agent_model, "max_tokens": agents.ANTHROPIC_MAX_TOKENS,
                "messages": [{"role": "user", "content": "ping"}],
                "thinking": agents.ANTHROPIC_THINKING,
                "output_config": {"effort": agents.ANTHROPIC_EFFORT}}
    elif spec.harness_protocol == agents.PROTOCOL_RESPONSES:
        url = f"{gw.openai_base_url}/responses"
        body = {"model": agent_model, "input": "ping", "max_output_tokens": 16}
        if spec.reasoning_effort:
            body["reasoning"] = {"effort": spec.reasoning_effort}
    else:
        url = f"{gw.openai_base_url}/chat/completions"
        body = {"model": agent_model, "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                **spec.sampling}
    return httpx.post(url, headers=headers, json=body, timeout=120.0)


# Values x-bt-error-origin uses for a failure raised by the proxy itself rather
# than relayed from a model vendor. A missing header means the same thing.
_GATEWAY_ORIGINS = ("braintrust", "gateway", "proxy")


def _gateway_failure(resp, agent_model: str) -> str:
    origin = resp.headers.get("x-bt-error-origin") or "gateway"
    from_gateway = origin.lower() in _GATEWAY_ORIGINS
    detail = (resp.text or "").strip()[:400]
    hint = ""
    if resp.status_code == 404 and from_gateway:
        hint = ("\nThe org this token belongs to has no AI provider configured "
                f"for {agent_model!r}. Add the vendor key under Settings -> AI "
                "Providers in that org, or point "
                f"{agents.GATEWAY_ORG_ENV} at the org that has one.")
    elif resp.status_code == 401:
        hint = (f"\nThe token is not authorized for that org. Issue the key or "
                f"service token from the same org named in "
                f"{agents.GATEWAY_ORG_ENV}.")
    elif resp.status_code == 403 and not from_gateway:
        hint = (f"\nThe gateway reached {origin} and was rejected there — the "
                "vendor key stored in the Braintrust org is invalid or expired.")
    return (f"Gateway request failed: HTTP {resp.status_code} "
            f"(origin: {origin})\n{detail}{hint}")


def _gateway_preflight(gw, spec, agent_model: str, search_mode: str) -> None:
    """Confirm the gateway will actually serve this arm's model."""
    try:
        resp = _gateway_probe(gw, spec, agent_model)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"Could not reach the gateway at {gw.root}: {exc}") from None
    if resp.status_code >= 400:
        raise SystemExit(_gateway_failure(resp, agent_model))
    if search_mode == SEARCH_MODE_NATIVE:
        # Deliberately not tested here: a hosted-tool probe costs a real billed
        # search, which is too much to spend on every run's preflight. It is also
        # the one gateway failure the run cannot detect for itself — a dropped
        # tool block yields an empty trajectory, which decision-surface gating
        # reports as unobservable rather than as an error.
        print("WARNING: native search through the gateway is not verified by "
              "this preflight. Hosted-tool passthrough (OpenAI Responses "
              "web_search, Anthropic web_search_20250305) is a property of the "
              "proxy, and if the tool blocks are dropped this arm records an "
              "empty trajectory. Run `python run_eval.py gateway-check "
              f"--model-vendor {spec.name} --agent-model {agent_model}` once "
              "per gateway/model pair before trusting a native arm.")


def gateway_check(env_path: Path, model_vendor: str,
                  agent_model: str | None) -> None:
    """Verify a gateway/model pair end to end, including hosted search tools.

    Separate from `run` because it spends a real billed search on purpose: the
    only way to know whether the proxy preserves server-side search blocks is to
    ask for one and look at what comes back.
    """
    load_runtime_env(env_path)
    gw = agents.gateway_config()
    if gw is None:
        raise SystemExit(
            f"{agents.GATEWAY_URL_ENV} is not set in {env_path}; there is no "
            "gateway to check. Set it to https://gateway.braintrust.dev to route "
            "model calls through Braintrust.")
    spec = vendor_of(model_vendor)
    agent_model = agent_model or spec.default_model
    print(f"gateway {gw.root} | org {gw.org or '(token default)'} | "
          f"project {gw.project or '(none)'}")
    print(f"vendor {spec.name} | model {agent_model}")

    resp = _gateway_probe(gw, spec, agent_model)
    if resp.status_code >= 400:
        raise SystemExit(_gateway_failure(resp, agent_model))
    print(f"  chat/messages: OK (HTTP {resp.status_code})")

    if not spec.supports_native_search:
        print(f"  native search: N/A — {spec.notes}")
        return

    if spec.name == "anthropic":
        tools = [{"type": agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
                  "name": "web_search", "max_uses": 1}]
        # The block types the native normalizer in agents.py reads. If these do
        # not survive the proxy, the arm silently loses its trajectory.
        wanted = ("server_tool_use", "web_search_tool_result")
    else:
        tools = [{"type": agents.OPENAI_WEB_SEARCH_TOOL_TYPE,
                  "search_context_size": agents.OPENAI_SEARCH_CONTEXT_SIZE}]
        wanted = ("web_search_call", "url_citation")
    resp = _gateway_search_probe(gw, spec, agent_model, tools)
    if resp.status_code >= 400:
        raise SystemExit(
            "Hosted search tool rejected through the gateway.\n"
            + _gateway_failure(resp, agent_model)
            + "\nThis vendor's --search-mode native arm cannot run on this "
              "gateway. Run the native arms direct-to-vendor, and note that "
              "doing so puts them on a different serving path from the harness "
              "arms — which breaks the native-vs-harness contrast.")
    body = resp.text
    found = [name for name in wanted if f'"{name}"' in body]
    if not found:
        raise SystemExit(
            f"The gateway returned HTTP {resp.status_code} but the response "
            f"carries none of {wanted}. The proxy accepted the hosted search "
            "tool and dropped its result blocks, which would give this arm an "
            "empty trajectory and an unobservable decision surface. Do not run "
            "--search-mode native on this gateway.")
    print(f"  native search: OK — response carries {', '.join(found)}")
    print("\nBoth checks passed. Record serving_path=gateway in the study notes "
          "and keep every arm of a contrast on this same path.")


def _gateway_search_probe(gw, spec, agent_model: str, tools: list):
    """Hosted-search probe: a question that cannot be answered parametrically.

    A stale-index model could answer 'what year is it' from memory without ever
    calling the tool, and a no-tool-call response is indistinguishable from a
    proxy that dropped the tool.
    """
    headers = {"Authorization": f"Bearer {gw.api_key}",
               "Content-Type": "application/json", **gw.headers()}
    question = ("Search the web and tell me one news headline published today. "
                "You must use the web search tool.")
    if spec.name == "anthropic":
        url = f"{gw.anthropic_base_url}/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        body = {"model": agent_model, "max_tokens": 1024, "tools": tools,
                "messages": [{"role": "user", "content": question}]}
    else:
        url = f"{gw.openai_base_url}/responses"
        body = {"model": agent_model, "input": question, "tools": tools,
                "max_tool_calls": 1,
                "include": ["web_search_call.action.sources"]}
    return httpx.post(url, headers=headers, json=body, timeout=180.0)


def _preflight(arm: str, search_mode: str, model_vendor: str,
               agent_model: str) -> None:
    """Fail before spending money, not on row 1 of N."""
    spec = vendor_of(model_vendor)

    # Structurally impossible cells first: no point sending someone to provision
    # a credential for an arm that cannot exist.
    if search_mode == SEARCH_MODE_NATIVE and not spec.supports_native_search:
        raise SystemExit(
            f"--search-mode native is not available for --model-vendor "
            f"{model_vendor}. {spec.notes} Run this vendor with "
            "--search-mode harness or none.")

    gw = agents.gateway_config()
    # Under gateway routing the vendor key lives in the Braintrust org, so
    # demanding a local one would block a correctly configured run. The You.com
    # key is still local either way: the harness calls that API itself, and the
    # gateway proxies model vendors only.
    needed = [] if gw else [spec.api_key_env]
    if search_mode == SEARCH_MODE_HARNESS:
        needed.append(SEARCH_PROVIDER_KEY)
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing env var(s): {', '.join(missing)}")

    if gw is not None:
        _gateway_preflight(gw, spec, agent_model, search_mode)

    if search_mode == SEARCH_MODE_HARNESS:
        ydc_setup(arm)  # raises, naming the valid setups, if the arm is unknown

    if search_mode == SEARCH_MODE_NATIVE:
        rate, confirmed = native_search_rate_usd(model_vendor, agent_model)
        if not confirmed:
            # Do not let an unrecognized model be priced at the cheaper rate by
            # default: OpenAI bills non-reasoning models through
            # web_search_preview at $25/1k instead of $10/1k.
            print(f"WARNING: {agent_model} is not a recognized reasoning model; "
                  f"pricing native search at ${rate:.3f}/search "
                  "(web_search_preview rate). Verify before reporting cost.")
        if model_vendor == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "--model-vendor anthropic requires the anthropic SDK: "
                    "pip install -r requirements.txt") from None


def build_judges(specs: list[str]):
    """Parse --judge specs into (client, model) pairs.

    A spec is `model` (OpenAI, OPENAI_API_KEY) or `model@base_url` for any
    OpenAI-compatible route, keyed by JUDGE_API_KEY. The second form is what
    lets the panel span vendors, which is the point — a single-vendor judge
    grading a single-vendor agent cannot rule out self-preference.

    Under gateway routing the bare form follows the agent onto the gateway: a
    judge is part of the measuring instrument, and leaving it on a direct route
    while the agents move would mean the scores for a gateway study came from a
    differently-served grader. An explicit `model@base_url` still wins — that
    form names a route on purpose.
    """
    gw = agents.gateway_config()
    judges = []
    for spec in specs:
        model, _, base_url = spec.partition("@")
        if base_url:
            key = os.environ.get("JUDGE_API_KEY")
            if not key:
                raise SystemExit(
                    f"--judge {spec} needs JUDGE_API_KEY for {base_url}")
            judges.append((OpenAI(base_url=base_url, api_key=key), model))
        elif gw is not None:
            judges.append((OpenAI(base_url=gw.openai_base_url, api_key=gw.api_key,
                                  default_headers=gw.headers() or None), model))
        else:
            judges.append((OpenAI(), model))
    return judges


def _eval_error_summary(eval_result) -> tuple[int, int, float]:
    """Return failed rows, total rows, and failure rate for the launcher gate."""
    failed = 0
    for result in eval_result.results:
        metadata = result.metadata or {}
        if (result.error is not None or metadata.get("scorer_errors")
                or metadata.get("classifier_errors")):
            failed += 1
    total = len(eval_result.results)
    return failed, total, failed / total if total else 1.0


def _enforce_eval_result_gate(
    eval_result,
    expected_results: int | None,
    max_row_error_rate: float,
) -> tuple[int, int, float]:
    """Reject incomplete conditions and rows with task or scorer failures."""
    failed, total, error_rate = _eval_error_summary(eval_result)
    print(
        f"Error gate: {failed}/{total} rows "
        f"({error_rate:.2%}); allowed {max_row_error_rate:.2%}"
    )
    if expected_results is not None and total != expected_results:
        raise SystemExit(
            f"condition returned {total} evaluation results; "
            f"expected {expected_results}"
        )
    if error_rate > max_row_error_rate:
        raise SystemExit(
            f"condition failed row-error gate: {error_rate:.2%} > "
            f"{max_row_error_rate:.2%}"
        )
    return failed, total, error_rate


def _write_completion_marker(path: Path, experiment_name: str) -> None:
    """Atomically mark a condition complete after its error gate passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        f"experiment={experiment_name}\ncompleted_at="
        f"{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(arm: str, dataset_name: str, dataset_version: str | None,
        trials: int, judge_specs: list[str], agent_model: str, study_id: str,
        env_path: Path, search_mode: str = SEARCH_MODE_HARNESS,
        model_vendor: str = DEFAULT_MODEL_VENDOR,
        split: str | None = None, limit: int | None = None,
        matrix_order_seed: str | None = None,
        matrix_order_index: int | None = None,
        matrix_condition_concurrency: int | None = None,
        matrix_schedule_policy: str | None = None,
        condition_attempt: int = 1,
        expected_rows: int | None = None,
        eval_timeout_s: float | None = None,
        max_row_error_rate: float = 0.0,
        max_concurrency: int = 8,
        ydc_requests_per_second: float = DEFAULT_YDC_REQUESTS_PER_SECOND,
        completion_marker: Path | None = None):
    api_key, project_id = load_runtime_env(env_path)
    _preflight(arm, search_mode, model_vendor, agent_model)
    spec = vendor_of(model_vendor)

    dataset = init_dataset(project_id=project_id, name=dataset_name,
                           version=dataset_version, api_key=api_key)
    project_name = dataset.project.name
    # braintrust 0.25 exposes Dataset.version as a string property.
    resolved_version = dataset_version or dataset.version
    print(f"Dataset: {project_name}/{dataset.name} @ version {resolved_version}"
          f"{'' if dataset_version else '  (latest; pass --dataset-version to pin)'}")

    # Matrix launches always provide expected_rows. Materialize an unlimited
    # pinned dataset in that case so the pre-launch count guard is real rather
    # than comparing the expected count with the lazy-path sentinel (None).
    data, subset = select_rows(
        dataset, split, limit, count_full=expected_rows is not None
    )
    if expected_rows is not None and subset["n_rows"] != expected_rows:
        raise SystemExit(
            f"row-count guard failed: selected {subset['n_rows']} rows, "
            f"expected {expected_rows}. Refusing to launch {agent_model}/{search_mode}."
        )
    if subset["subset_applied"]:
        print(f"Rows: {subset['n_rows']} of {subset['n_available']} "
              f"(split={subset['split'] or 'all'}, limit={limit}) "
              f"subset_id={subset['subset_id']}")

    judges = build_judges(judge_specs)
    # One judge keeps SimpleQA parity with LiveNewsBench's published numbers;
    # several convene a jury. Both emit exactly one score per row.
    judge = (make_simpleqa_grader(judges[0][0], judge_model=judges[0][1])
             if len(judges) == 1 else make_jury_grader(judges))
    if search_mode == SEARCH_MODE_NONE:
        print("Control arm: no tools. Subtract this from the search arms of the "
              "SAME vendor to get retrieval's marginal value.")
    if search_mode == SEARCH_MODE_NATIVE:
        print(f"Native arm: {model_vendor} runs the search server-side. "
              "Decision-surface metrics are partially unobservable here — see "
              "decision_surface in the row metadata; compare only against this "
              "vendor's own harness and no_search arms.")

    condition = condition_label(search_mode, arm, model_vendor)
    condition_id = f"{model_vendor}:{agent_model}::{condition}"
    model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", agent_model)
    experiment_name = f"{study_id}-{dataset.name}-{model_slug}-{condition}"
    if condition_attempt > 1:
        # A retry is a new experiment, not an append into a partial attempt.
        # This keeps denominators clean and makes the checkpoint's chosen
        # successful attempt explicit.
        experiment_name += f"-retry-{condition_attempt:02d}"
    native_rate, rate_confirmed = native_search_rate_usd(model_vendor, agent_model)
    # Sampling parity is NOT achievable across vendors: both frontier vendors
    # reject sampling parameters outright and no vendor here supports `seed`.
    # Record what each arm received instead of implying a frozen config.
    sampling = dict(spec.sampling)

    eval_result = Eval(
        project_name,
        experiment_name=experiment_name,
        data=data,
        task=make_task(
            arm,
            agent_model,
            study_id,
            dataset.name,
            str(resolved_version),
            search_mode,
            model_vendor,
            ydc_requests_per_second,
        ),
        scores=[judge, *DETERMINISTIC_SCORERS],
        trial_count=trials,          # web nondeterminism > model nondeterminism
        timeout=eval_timeout_s,
        max_concurrency=max_concurrency,
        metadata={
            # --- matrix axes ---
            "model_class": spec.model_class,
            "model_vendor": model_vendor,
            "search_mode": search_mode,
            "search_provider": (
                SEARCH_PROVIDER if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else "none"),
            "freshness_treatment": (
                arm if search_mode == SEARCH_MODE_HARNESS else None),
            # --- legacy field names, kept so older experiments still join ---
            "provider": (
                SEARCH_PROVIDER if search_mode == SEARCH_MODE_HARNESS else "none"
                if search_mode == SEARCH_MODE_NONE else f"{model_vendor}_native"),
            "arm": arm, "agent_model": agent_model,
            "ydc_setup": (
                experiment_ydc_setup(arm, dataset.name)
                if search_mode == SEARCH_MODE_HARNESS
                else None),
            "study_id": study_id,
            "condition_id": condition_id,
            "matrix_order_seed": matrix_order_seed,
            "matrix_order_index": matrix_order_index,
            "matrix_condition_concurrency": matrix_condition_concurrency,
            "matrix_schedule_policy": matrix_schedule_policy,
            "condition_attempt": condition_attempt,
            "eval_timeout_s": eval_timeout_s,
            "max_row_error_rate": max_row_error_rate,
            "max_concurrency": max_concurrency,
            "ydc_requests_per_second": (
                ydc_requests_per_second
                if search_mode == SEARCH_MODE_HARNESS else None),
            "judge_models": [m for _, m in judges],
            "judge_mode": "single" if len(judges) == 1 else "jury",
            "dataset_name": dataset.name,
            "dataset_version": resolved_version,
            "dataset_version_pinned": bool(dataset_version),
            # Two runs are only comparable if these match: every contrast is
            # paired by task_key, so differing subsets lose the pairing silently.
            **{f"row_{k}": v for k, v in subset.items()},
            "budget": {"searches": MAX_SEARCHES, "clicks": MAX_CLICKS},
            # --- what the agent was actually configured with ---
            # No frontier vendor permits temperature/seed: gpt-5-family models
            # reject `temperature` with a 400 and support no `seed`, and Sonnet
            # 5 rejects non-default temperature/top_p/top_k. Only the OSS arm
            # pins sampling, so
            # sampling_pinned is False on both frontier vendors by necessity.
            "sampling_params": sampling,
            "sampling_pinned": bool(sampling),
            "reasoning_effort": spec.reasoning_effort,
            "reasoning_effort_pinned": spec.reasoning_effort is not None,
            # Which endpoint served this arm. Recorded because it is a real
            # difference between arms of the same vendor family — the OSS rows run
            # chat completions and the OpenAI rows run Responses — and because the
            # endpoint determines whether reasoning_effort could be pinned at all.
            "harness_protocol": (
                spec.harness_protocol if search_mode != SEARCH_MODE_NATIVE
                else agents.PROTOCOL_RESPONSES if spec.name == "openai"
                else agents.PROTOCOL_MESSAGES),
            "agent_base_url": agents.effective_base_url(spec),
            # Direct-to-vendor or proxied through the Braintrust gateway. A
            # contrast whose two runs disagree here is confounded: the serving
            # stack moved along with whatever the contrast was testing. Recorded
            # so that mix is detectable in analysis instead of invisible.
            "serving_path": agents.serving_path(),
            "gateway_project": (
                gateway.project if (gateway := agents.gateway_config()) else None),
            # Whether the five-call budget is API-enforced. Anthropic receives
            # the remaining search max_uses on each pause_turn continuation;
            # OpenAI receives max_tool_calls, which also covers page actions.
            "search_budget_enforced": (
                agents.NATIVE_BUDGET_ENFORCED.get(model_vendor, False)
                if search_mode == SEARCH_MODE_NATIVE
                else search_mode == SEARCH_MODE_HARNESS),
            # Provider date fields are not one construct. New results carry a
            # per-result date_semantics label; temporal_grounding uses only
            # explicit publication timestamps. Prefer Corvus-QA's recency_rung
            # as dataset ground truth about when a fact changed.
            "date_field_semantics": agents.DATE_FIELD_SEMANTICS.get(
                SEARCH_PROVIDER if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else None),
            # What truncation the harness applied. See SNIPPET_TRUNCATION.
            "snippet_truncation": (
                SNIPPET_TRUNCATION if search_mode == SEARCH_MODE_HARNESS else None),
            # Requested result count PER SECTION for this setup. Not a global
            # constant: the `wide` setup varies it, which is the point of that
            # arm. Kept under the legacy field name for older experiment joins.
            "n_results": (
                ydc_setup(arm)["count"] if search_mode == SEARCH_MODE_HARNESS
                else None),
            # Registered native-vs-harness target: the normalized You.com arm
            # enforces a maximum of 5 results in each of two sections. Native
            # APIs expose neither that split nor an exact source-count control,
            # so 10 is a declared target there and observed trajectory volume is
            # the measurement. Never truncate native results after the model has
            # already seen them; that would change metrics, not treatment.
            "result_count_target_per_search": (
                2 * ydc_setup(arm)["count"]
                if search_mode == SEARCH_MODE_HARNESS
                else NATIVE_RESULT_COUNT_TARGET if search_mode == SEARCH_MODE_NATIVE
                else None),
            "result_count_control": (
                "per_section_max" if search_mode == SEARCH_MODE_HARNESS
                else "unavailable_observed_only"
                if search_mode == SEARCH_MODE_NATIVE else None),
            "youdotcom_freshness": (
                experiment_ydc_setup(arm, dataset.name)["freshness"]
                if search_mode == SEARCH_MODE_HARNESS
                else None),
            "ydc_setup_name": (
                arm if search_mode == SEARCH_MODE_HARNESS else None),
            "search_user_location": (
                SEARCH_USER_LOCATION if search_mode != SEARCH_MODE_NONE else None),
            "ydc_language": (
                YDC_LANGUAGE if search_mode == SEARCH_MODE_HARNESS else None),
            "ydc_safesearch": (
                YDC_SAFESEARCH if search_mode == SEARCH_MODE_HARNESS else None),
            "ydc_usd_per_call": (
                YDC_USD_PER_CALL if search_mode == SEARCH_MODE_HARNESS else None),
            # --- native-arm configuration, declared so the arm is reproducible ---
            "native_search_tool": (
                agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE if (
                    search_mode == SEARCH_MODE_NATIVE
                    and model_vendor == "anthropic")
                else agents.OPENAI_WEB_SEARCH_TOOL_TYPE if (
                    search_mode == SEARCH_MODE_NATIVE
                    and model_vendor == "openai")
                else None),
            "native_search_usd_per_call": (
                native_rate if search_mode == SEARCH_MODE_NATIVE else None),
            "native_search_rate_confirmed": (
                rate_confirmed if search_mode == SEARCH_MODE_NATIVE else None),
            "openai_search_context_size": (
                agents.OPENAI_SEARCH_CONTEXT_SIZE if (
                    search_mode == SEARCH_MODE_NATIVE
                    and model_vendor == "openai") else None),
            "anthropic_thinking": (
                agents.ANTHROPIC_THINKING if model_vendor == "anthropic" else None),
            # --- token pricing assumptions, recorded so a cost number is
            # auditable without knowing which commit produced it ---
            "model_usd_per_mtok": list(
                agents.MODEL_USD_PER_MTOK.get(agent_model, ())) or None,
            "cached_input_multiplier": (
                agents.cached_input_multiplier(agent_model)
                if agent_model in agents.MODEL_USD_PER_MTOK else None),
            # OpenAI caches automatically above ~1k prompt tokens; Anthropic
            # caches only on explicit cache_control, which these requests never
            # send. So cache hits are expected on the OpenAI rows and expected to
            # be zero everywhere else — agent_cache_hit_rate is the per-row check.
            "prompt_caching_expected": model_vendor == "openai",
            "git_commit": _git_commit(),
            "prompt_version": PROMPT_VERSIONS[search_mode],
            # Native retrieval configuration, so each arm is reproducible from
            # the row. allowed_callers ["direct"] means dynamic filtering was
            # off and every result the model saw is observable to us.
            "native_search_allowed_callers": (
                agents.ANTHROPIC_WEB_SEARCH_ALLOWED_CALLERS
                if (search_mode == SEARCH_MODE_NATIVE
                    and model_vendor == "anthropic") else None),
            "native_search_context_size": (
                agents.OPENAI_SEARCH_CONTEXT_SIZE
                if (search_mode == SEARCH_MODE_NATIVE
                    and model_vendor == "openai") else None),
            "question_transform_version": (
                "retrievalqa-answer-as-of-v1"
                if dataset.name == RETRIEVALQA_DATASET_NAME else "identity-v1"),
        },
    )
    expected_results = (
        subset["n_rows"] * trials if subset["n_rows"] is not None else None
    )
    _enforce_eval_result_gate(
        eval_result, expected_results, max_row_error_rate
    )
    if completion_marker is not None:
        _write_completion_marker(completion_marker, experiment_name)
    return eval_result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--search-mode", choices=list(SEARCH_MODES),
                   default=SEARCH_MODE_HARNESS,
                   help="none = no tools (parametric control); harness = our "
                        "search_web tool over a search API; native = the model "
                        "vendor's own server-side search. NOTE: this is a "
                        "different axis from --arm native_fresh, which is a "
                        "search API's freshness parameter.")
    r.add_argument("--model-vendor", choices=sorted(VENDORS),
                   default=DEFAULT_MODEL_VENDOR,
                   help="baseten = OSS models (OpenAI-compatible); openai and "
                        "anthropic = frontier, both of which support "
                        "--search-mode native.")
    r.add_argument("--arm", default=DEFAULT_ARM,
                   choices=[*YDC_SETUPS, NO_SEARCH_ARM],
                   help="You.com setup for --search-mode harness: "
                        + "; ".join(
                            f"{name} (count={cfg['count']}, "
                            f"freshness={cfg['freshness'] or 'none'})"
                            for name, cfg in YDC_SETUPS.items())
                        + ". no_search is a deprecated alias for "
                        "--search-mode none.")
    r.add_argument("--dataset-name", default=DATASET_NAME)
    r.add_argument("--dataset-version", default=None,
                   help="Pin a dataset version so every provider/arm sees the "
                        "same rows. Required unless --allow-latest is passed.")
    r.add_argument("--allow-latest", action="store_true",
                   help="Exploratory runs only: allow an unpinned dataset head.")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--study-id", default="freshness-v1",
                   help="Shared identifier for every condition in one experiment "
                        "matrix. Reuse it across providers, arms, and models.")
    r.add_argument("--agent-model", default=None,
                   help="Defaults to the chosen vendor's pinned model "
                        + ", ".join(f"{v}={s.default_model}"
                                    for v, s in sorted(VENDORS.items()))
                        + ". Provider rankings can be model-dependent; re-run "
                        "the matrix under a second model to check that yours "
                        "generalizes.")
    r.add_argument("--judge", action="append", dest="judges", metavar="MODEL[@BASE_URL]",
                   help="Repeatable. One judge emits one semantic score; three "
                        "or more convene a majority-vote jury. Use model@base_url "
                        "with JUDGE_API_KEY for non-OpenAI routes.")
    r.add_argument("--split", default=None,
                   help="Restrict to one dataset split (LiveNewsBench: val, "
                        "train, test, human_verified_test; Corvus-QA: dev, "
                        "test). Applied identically to every arm or the pairing "
                        "breaks.")
    r.add_argument("--limit", type=int, default=None,
                   help="Cap rows per run, taken deterministically after "
                        "sorting by row id. Required for a cost-bounded pilot: "
                        "the full LiveNewsBench matrix is 14 conditions and "
                        "18,606 row executions at one trial.")
    r.add_argument("--env-file", type=Path, default=Path(".env"))
    r.add_argument("--matrix-order-seed", default=None,
                   help="Condition-order seed supplied by run_matrix.py.")
    r.add_argument("--matrix-order-index", type=int, default=None,
                   help="One-based execution position supplied by run_matrix.py.")
    r.add_argument("--matrix-condition-concurrency", type=int, default=None,
                   help="Concurrent-condition cap supplied by run_matrix.py.")
    r.add_argument("--matrix-schedule-policy", default=None,
                   help="Compatibility policy supplied by run_matrix.py.")
    r.add_argument("--condition-attempt", type=int, default=1,
                   help="One-based condition attempt supplied by run_matrix.py.")
    r.add_argument("--expected-rows", type=int, default=None,
                   help="Abort before Eval if the selected row count differs.")
    r.add_argument("--eval-timeout-s", type=float, default=None,
                   help="Whole-condition timeout; Braintrust flushes in finally.")
    r.add_argument("--max-row-error-rate", type=float, default=0.0,
                   help="Exit nonzero when task/scorer failures exceed this rate.")
    r.add_argument("--max-concurrency", type=int, default=8)
    r.add_argument(
        "--ydc-requests-per-second", type=float,
        default=DEFAULT_YDC_REQUESTS_PER_SECOND,
        help="Shared You.com request rate across local processes (default: 1; "
             f"hard cap: {MAX_YDC_REQUESTS_PER_SECOND:g}).",
    )
    r.add_argument("--completion-marker", type=Path, default=None,
                   help=argparse.SUPPRESS)

    g = sub.add_parser(
        "gateway-check",
        help="Verify that the configured Braintrust gateway can serve a "
             "vendor/model pair, including its hosted web-search tool. Spends "
             "one billed search on the native check; run it once per "
             "gateway/model pair, not per experiment.")
    g.add_argument("--model-vendor", choices=sorted(VENDORS),
                   default=DEFAULT_MODEL_VENDOR)
    g.add_argument("--agent-model", default=None,
                   help="Defaults to the vendor's pinned model.")
    g.add_argument("--env-file", type=Path, default=Path(".env"))

    args = ap.parse_args()
    if args.cmd == "gateway-check":
        gateway_check(args.env_file, args.model_vendor, args.agent_model)
        return
    if not args.dataset_version and not args.allow_latest:
        ap.error("--dataset-version is required for paired comparisons; pass "
                 "--allow-latest only for exploratory runs")
    if args.trials < 1:
        ap.error("--trials must be at least 1")
    if args.condition_attempt < 1:
        ap.error("--condition-attempt must be at least 1")
    if args.expected_rows is not None and args.expected_rows < 1:
        ap.error("--expected-rows must be at least 1")
    if args.eval_timeout_s is not None and args.eval_timeout_s <= 0:
        ap.error("--eval-timeout-s must be positive")
    if not 0 <= args.max_row_error_rate <= 1:
        ap.error("--max-row-error-rate must be between 0 and 1")
    if args.max_concurrency < 1:
        ap.error("--max-concurrency must be at least 1")
    if not 0 < args.ydc_requests_per_second <= MAX_YDC_REQUESTS_PER_SECOND:
        ap.error("--ydc-requests-per-second must be greater than zero and no "
                 f"more than {MAX_YDC_REQUESTS_PER_SECOND:g}")

    search_mode = args.search_mode
    # `--arm no_search` predates the search_mode axis. Honor it so existing
    # scripts keep working, but map it onto the axis rather than carrying two
    # ways to express the same condition.
    if args.arm == NO_SEARCH_ARM:
        if search_mode not in (SEARCH_MODE_HARNESS, SEARCH_MODE_NONE):
            ap.error(f"--arm no_search conflicts with --search-mode "
                     f"{search_mode}; drop one")
        search_mode = SEARCH_MODE_NONE
    if search_mode == SEARCH_MODE_NATIVE and args.arm != DEFAULT_ARM:
        # native_fresh is a search-API parameter; there is no such knob on a
        # model vendor's server-side search, so accepting it would imply a
        # treatment that was never applied.
        ap.error("--arm applies only to --search-mode harness; the model "
                 "vendor's native search exposes no freshness parameter")

    agent_model = args.agent_model or VENDORS[args.model_vendor].default_model
    # Luna handles the high-volume semantic check. Use --judge gpt-4.1 when a
    # run needs direct parity with LiveNewsBench's published judge.
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    run(args.arm, args.dataset_name, args.dataset_version,
        args.trials, args.judges or [DEFAULT_JUDGE_MODEL], agent_model,
        args.study_id, args.env_file, search_mode, args.model_vendor,
        args.split, args.limit, args.matrix_order_seed, args.matrix_order_index,
        args.matrix_condition_concurrency, args.matrix_schedule_policy,
        args.condition_attempt, args.expected_rows, args.eval_timeout_s,
        args.max_row_error_rate, args.max_concurrency,
        args.ydc_requests_per_second, args.completion_marker)


if __name__ == "__main__":
    main()
