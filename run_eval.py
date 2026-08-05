"""Starter Braintrust harness for web-search API x LLM freshness experiments.

Two treatment axes, defined in agents.py and selected on the command line:

    --model-vendor   baseten (OSS) | openai (frontier) | anthropic (frontier)
    --search-mode    none (parametric) | harness (our tool) | native (vendor's own)

Within one condition block everything else is frozen (model snapshot, prompt,
tool contract, and budget). Sampling is the one thing that cannot be frozen
across vendors — Claude Opus 5 rejects temperature/top_p/top_k, and Baseten does
not document seed support — so each run records what it actually sent rather than
implying parity.

Braintrust-native throughout:

  * reads a versioned LiveNewsBench or RetrievalQA dataset from the project
    named by BRAINTRUST_PROJECT_ID; real comparisons require a pinned version
  * agent LLM calls auto-traced via wrap_openai
  * every approved search is a `tool` span with native metrics; raw provider
    payloads are never retained
  * trial_count for retrieval nondeterminism (temp 0 doesn't freeze the web)

Dataset contract (set by the importers, consumed by scorers.py):
    input    = {"question": str}
    expected = answer string or list of acceptable answer strings
    metadata = upstream row fields (link, articles, event_date, ...) plus
               livenewsbench_release / livenewsbench_split / source_commit
Leakage excludes and temporal grounding both read from metadata, not expected.

Usage:
    # .env supplies BRAINTRUST_API_KEY + BRAINTRUST_PROJECT_ID and always wins
    # All credentials must be in .env; ambient credentials are ignored.

    # dataset push lives in import_livenewsbench.py, not here:
    python import_livenewsbench.py <datasets_root> --source-commit <sha>

    # one command per matrix cell; interleave them in time, don't run days apart
    python run_eval.py run --model-vendor openai --search-mode harness \
      --provider exa --arm native_fresh \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 3
    python run_eval.py run --model-vendor openai --search-mode native \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 3
    python run_eval.py run --model-vendor baseten --search-mode none \
      --dataset-version <xact-id> --study-id matrix-v1 --trials 3

Native search is only attributable WITHIN a vendor: run each frontier vendor's
none/harness/native arms together, and compare its native arm to its own harness
arm — not to the other vendor's. See README "The test matrix".
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import time
from datetime import datetime, timezone
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
from corvus.sources import SharedHostLimiter, retry_after_seconds
from scorers import (DETERMINISTIC_SCORERS, iter_source_urls,
                     make_jury_grader, make_simpleqa_grader)

# ---------------------------------------------------------------------------
# Credentials: values from .env always override ambient credentials.
# ---------------------------------------------------------------------------


RUNTIME_ENV_NAMES = (
    "BRAINTRUST_API_KEY",
    "BRAINTRUST_PROJECT_ID",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BASETEN_API_KEY",
    "EXA_API_KEY",
    "PARALLEL_API_KEY",
    "YDC_API_KEY",
    "JUDGE_API_KEY",
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
SNIPPET_CHARS = 400                    # normalized-arm snippet truncation
N_RESULTS = 8

# Exa search tier. Their default is "auto", which routes per query — pinning it
# keeps the retrieval tier a declared experiment condition. Documented values:
# instant | fast | auto | deep-lite | deep | deep-reasoning.
EXA_SEARCH_TYPE = "auto"
# Exa content freshness. maxAgeHours REPLACED the livecrawl string parameter
# (Feb 2026): positive N serves cache younger than N hours and livecrawls
# otherwise, 0 always livecrawls, -1 is cache-only, omitted livecrawls only when
# no cache exists. Accepts -1..720. 24 = the declared "fresh within a day" arm.
EXA_MAX_AGE_HOURS = 24
# Exa's documented default is 10000 ms, max 90000. Pinned so a livecrawl stall
# is a declared condition, and kept under the 30 s client timeout in _http.
EXA_LIVECRAWL_TIMEOUT_MS = 10_000
# Exa caps includeDomains + excludeDomains at 1200 items each.
EXA_MAX_DOMAINS = 1200
# Parallel's GA Search API modes are turbo | basic | advanced. Pin one mode for
# both arms so freshness is the only treatment that changes.
PARALLEL_MODE = "basic"
# Match Exa's one-day cache-age treatment. This controls content fetch age, not
# publication date, so it is closer to Exa's maxAgeHours than after_date is.
PARALLEL_MAX_AGE_SECONDS = 24 * 60 * 60
# Combined include_domains + exclude_domains ceiling in a Parallel source_policy.
PARALLEL_MAX_DOMAINS = 200

FROZEN_SYSTEM_PROMPT = """\
You are a web research agent answering a time-sensitive factual question. You have one tool:
search_web(query). You may use at most 5 searches and may not fetch result pages.
Search results show rank, title, url, snippet, and
published date when available. When you know the answer, stop calling tools
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
search. You may use at most 5 searches. When you know the answer, stop searching
and reply with ONLY the final answer, as concisely as possible. If you cannot
determine the answer within budget, reply exactly: I could not find this."""

PROMPT_VERSIONS = {
    SEARCH_MODE_HARNESS: "frozen-v2-generic-factual",
    SEARCH_MODE_NONE: "no-search-v2-parametric",
    SEARCH_MODE_NATIVE: "native-search-v1-generic-factual",
}

# The harness tool schema lives in agents.SEARCH_TOOL_* — one definition
# translated per wire protocol, so the OpenAI and Anthropic harness arms cannot
# drift into offering subtly different tools.

# Search pricing (USD) for native cost metrics. A flat per-call constant is
# wrong for any vendor that bills content per page, so each entry carries three
# documented terms and search_cost_usd() applies them to the results actually
# returned.
#
# Exa (https://exa.ai/docs/reference/pricing, checked 2026-07-30):
#   $7/1k requests for instant|fast|auto covering the first 10 results,
#   +$1/1k results beyond 10, +$1/1k pages PER CONTENT TYPE. We request one
#   content type (highlights), so 8 results add $0.008 — the old flat $0.005
#   understated an Exa call by roughly 3x. Livecrawling via maxAgeHours is not
#   billed separately from the content charge.
# You.com (https://you.com/docs/api-reference/search, checked 2026-07-30):
#   $5.00/1k calls, independent of `count`. livecrawl would add $1/1k pages,
#   and we leave it off in both arms.
# Parallel (https://docs.parallel.ai/getting-started/pricing, checked 2026-08-05):
#   basic/advanced are $5/1k requests for the first 10 results, plus $1/1k for
#   each additional result and excerpt. This harness requests only 8 results.
SEARCH_PRICING = {
    ("exa", "normalized"):         {"per_call": 0.007, "per_result_content": 0.001, "per_result_over_10": 0.001},
    ("exa", "native_fresh"):       {"per_call": 0.007, "per_result_content": 0.001, "per_result_over_10": 0.001},
    ("parallel", "normalized"):    {"per_call": 0.005, "per_result_content": 0.0,   "per_result_over_10": 0.001},
    ("parallel", "native_fresh"):  {"per_call": 0.005, "per_result_content": 0.0,   "per_result_over_10": 0.001},
    ("youdotcom", "normalized"):   {"per_call": 0.005, "per_result_content": 0.0,   "per_result_over_10": 0.0},
    ("youdotcom", "native_fresh"): {"per_call": 0.005, "per_result_content": 0.0,   "per_result_over_10": 0.0},
}


def search_cost_usd(provider: str, arm: str, n_results: int) -> float:
    """Documented per-call price plus per-page content charges."""
    rate = SEARCH_PRICING.get((provider, arm))
    if rate is None:
        return 0.0
    return (rate["per_call"]
            + n_results * rate["per_result_content"]
            + max(0, n_results - 10) * rate["per_result_over_10"])

ARCHIVE_EXCLUDES = ["web.archive.org", "archive.org", "archive.is",
                    "archive.ph", "archive.today"]

# Env var each provider adapter reads, for preflight validation.
PROVIDER_KEYS = {"exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY",
                 "youdotcom": "YDC_API_KEY"}


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


def _provider_json(method: str, url: str, **kwargs):
    """Conservative serial provider request with bounded retry/backoff."""
    host = (urlparse(url).hostname or "").lower()
    approved_hosts = {"api.exa.ai", "api.parallel.ai", "ydc-index.io"}
    if host not in approved_hosts:
        raise ValueError(f"unapproved provider API host: {host!r}")
    limiter = _provider_limiters.setdefault(
        host, SharedHostLimiter(f"provider-{host}", 1.0)
    )
    for attempt in range(5):
        limiter.wait()
        response = _http.request(method, url, **kwargs)
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
        delay = max(retry_after_seconds(response.headers.get("Retry-After")),
                    min(60.0, 5.0 * (2 ** attempt)))
        time.sleep(delay)
    raise AssertionError("unreachable")


def exa_search(query: str, arm: str, exclude_domains: list[str]):
    body = {
        # `type` defaults to "auto", which routes per query and is therefore NOT
        # frozen across runs. Pinned here so the search tier is a declared
        # condition; "fast"/"deep" are the other reproducible choices.
        "type": EXA_SEARCH_TYPE,
        "query": query,
        "numResults": N_RESULTS,
        "category": "news",
        "excludeDomains": exclude_domains[:EXA_MAX_DOMAINS],
        # numSentences/highlightsPerUrl are deprecated; maxCharacters is current
        # and matches the SNIPPET_CHARS budget enforced below.
        "contents": {"highlights": {"maxCharacters": SNIPPET_CHARS}},
    }
    if arm == "native_fresh":
        # The whole `livecrawl` string parameter is deprecated in favor of
        # maxAgeHours, not just its "preferred" value. Sending both was also
        # self-contradictory: livecrawl="always" says never use cache while
        # maxAgeHours=24 accepts cache up to a day old. maxAgeHours alone is the
        # documented way to express this arm's treatment.
        body["contents"]["maxAgeHours"] = EXA_MAX_AGE_HOURS
        body["contents"]["livecrawlTimeout"] = EXA_LIVECRAWL_TIMEOUT_MS
    raw = _provider_json(
        "POST",
        "https://api.exa.ai/search",
        json=body,
        headers={"x-api-key": os.environ["EXA_API_KEY"]},
    )
    results = []
    for i, res in enumerate(raw.get("results", []), start=1):
        highlights = res.get("highlights") or []
        results.append({
            "rank": i,
            "url": res.get("url", ""),
            "title": res.get("title") or "",
            "snippet": (highlights[0] if highlights else res.get("text", "") or "")[:SNIPPET_CHARS],
            "published_date": res.get("publishedDate"),
        })
    return results, raw


def parallel_search(query: str, arm: str, exclude_domains: list[str]):
    advanced_settings = {
        # Keep the decision surface and result count identical across vendors.
        "excerpt_settings": {"max_chars_per_result": SNIPPET_CHARS},
        "max_results": N_RESULTS,
        # Ceiling is 200 combined include+exclude domains. exclude_domains is
        # source-first, so any truncation drops archive mirrors, not gold sources.
        "source_policy": {
            "exclude_domains": exclude_domains[:PARALLEL_MAX_DOMAINS]},
    }
    if arm == "native_fresh":
        advanced_settings["fetch_policy"] = {
            "max_age_seconds": PARALLEL_MAX_AGE_SECONDS}
    body = {
        "objective": query,
        "search_queries": [query[:200]],
        "mode": PARALLEL_MODE,
        "advanced_settings": advanced_settings,
    }
    raw = _provider_json(
        "POST",
        "https://api.parallel.ai/v1/search",
        json=body,
        headers={
            "x-api-key": os.environ["PARALLEL_API_KEY"],
            "content-type": "application/json",
        },
    )
    results = []
    for i, res in enumerate(raw.get("results", []), start=1):
        excerpts = res.get("excerpts") or []
        snippet = " ".join(excerpts)[:SNIPPET_CHARS]
        results.append({
            "rank": i,
            "url": res.get("url", ""),
            "title": res.get("title") or "",
            "snippet": snippet,
            "published_date": res.get("publish_date"),
        })
    return results, raw


def youdotcom_search(query: str, arm: str, exclude_domains: list[str]):
    # GET https://ydc-index.io/v1/search — the documented base host is
    # ydc-index.io (no api. prefix), and GET is the only shape You.com publishes
    # a full parameter spec for. Their docs do recommend POST for domain lists
    # (up to 500 as a JSON array) because GET passes them comma-separated and is
    # bounded by URL length, but the POST body schema is undocumented. Our list
    # is a handful of domains, so GET stays well inside the documented path.
    # exclude_domains is one comma-separated string and is mutually exclusive
    # with include_domains (sending both returns 422).
    params = {"query": query, "count": N_RESULTS}
    if exclude_domains:
        # Omit rather than send an empty string, which is not a documented value.
        params["exclude_domains"] = ",".join(exclude_domains)
    if arm == "native_fresh":
        # day|week|month|year or YYYY-MM-DDtoYYYY-MM-DD. Use one day to match
        # the declared 24-hour treatment; never derive this from event_date.
        params["freshness"] = "day"
        # NB: You.com also exposes a `livecrawl` param, but it only populates
        # result.contents — it would not change the snippet decision surface we
        # measure, so enabling it would add latency and $1/1k pages for no signal.
    raw = _provider_json(
        "GET",
        "https://ydc-index.io/v1/search",
        params=params,
        headers={
            "X-API-Key": os.environ["YDC_API_KEY"],
            # You.com documents that GET responses are cacheable at CDN and
            # proxy layers while POST responses are not. Freshness is the
            # quantity under measurement here, so ask intermediaries not to
            # serve a stale hit.
            "Cache-Control": "no-cache",
        },
    )
    # results.web[] carries `snippets`; results.news[] does not, so web is the
    # only shape comparable to the other providers' snippets.
    hits = (raw.get("results") or {}).get("web") or []
    results = []
    for i, res in enumerate(hits, start=1):
        snippets = res.get("snippets") or []
        results.append({
            "rank": i,
            "url": res.get("url", ""),
            "title": res.get("title") or "",
            "snippet": (snippets[0] if snippets else res.get("description", "") or "")[:SNIPPET_CHARS],
            # `page_age` is the documented timestamp field on a web result;
            # there is no `published_date` in this response shape.
            "published_date": res.get("page_age"),
        })
    return results, raw


PROVIDERS = {"exa": exa_search, "parallel": parallel_search,
             "youdotcom": youdotcom_search}


def _provider_request_id(provider: str, raw: dict) -> str | None:
    """Return each vendor's safe correlation ID without retaining raw payloads."""
    if provider == "exa":
        return raw.get("requestId")
    if provider == "parallel":
        return raw.get("search_id")
    if provider == "youdotcom":
        return (raw.get("metadata") or {}).get("search_uuid")
    return None


# ---------------------------------------------------------------------------
# Traced tools. type="tool" spans; native metrics without raw payload retention.
# notrace_io=True on both: @traced otherwise auto-logs the return value over
# the explicit output= below, and these return tuples, not the payload we want.
# ---------------------------------------------------------------------------

@traced(type="tool", name="search_web", notrace_io=True)
def run_search(provider: str, arm: str, query: str, exclude_domains: list[str]):
    t0 = time.perf_counter()
    results, raw = PROVIDERS[provider](query, arm, exclude_domains)
    latency = time.perf_counter() - t0
    rendered = "\n".join(
        f"[{r['rank']}] {r['title']}\n    {r['url']}\n    "
        f"published: {r['published_date'] or 'unknown'}\n    {r['snippet']}"
        for r in results) or "No results."
    current_span().log(
        input={"query": query, "provider": provider, "arm": arm,
               "exclude_domains": exclude_domains},
        output=results,
        metadata={"provider": provider, "arm": arm,
                  "provider_request_id": _provider_request_id(provider, raw),
                  "raw_payload_retained": False},
        metrics={"tokens": _tok(rendered),
                 "latency_s": latency,
                 "search_cost_usd": search_cost_usd(provider, arm, len(results)),
                 "n_results": len(results)},
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


def condition_label(search_mode: str, provider: str, arm: str,
                    model_vendor: str) -> str:
    """One slug per matrix cell. Distinguishes the two collision-prone names:
    `arm=native_fresh` is a SEARCH VENDOR's freshness parameter, while
    `search_mode=native` is the MODEL vendor's own server-side search."""
    if search_mode == SEARCH_MODE_NONE:
        return "no_search"
    if search_mode == SEARCH_MODE_NATIVE:
        return f"native-{model_vendor}"
    return f"harness-{provider}-{arm}"


def make_task(
    provider: str,
    arm: str,
    agent_model: str = AGENT_MODEL,
    study_id: str = "freshness-v1",
    dataset_name: str = DATASET_NAME,
    dataset_version: str | None = None,
    search_mode: str = SEARCH_MODE_HARNESS,
    model_vendor: str = DEFAULT_MODEL_VENDOR,
):
    spec = vendor_of(model_vendor)
    system_prompt = _system_prompt_for(search_mode)
    condition = condition_label(search_mode, provider, arm, model_vendor)
    condition_id = f"{model_vendor}:{agent_model}::{condition}"

    def task(input: dict, hooks) -> dict:
        # Leakage excludes come from row METADATA (link + articles[*]), which is
        # where the importer puts them and where leakage_guard reads them.
        # `expected` is the answer string, so it has no .get().
        row_metadata = hooks.metadata or {}
        source_domains = source_domains_of(row_metadata, hooks.expected)
        excludes = source_domains + ARCHIVE_EXCLUDES
        question = input["question"]
        task_key = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
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
        # Wall-clock per row. Vals AI's Web Search Index reports an independent
        # API completing FASTER than native search despite issuing more search
        # calls, so speed is a real axis and not derivable from search count.
        # Per-search latency already lands on the tool spans; this is the total a
        # user would actually wait, including model turns.
        t0 = time.perf_counter()
        if search_mode == SEARCH_MODE_NATIVE:
            outcome = _run_native(client, spec, agent_model, system_prompt,
                                  question, excludes)
        else:
            outcome = _run_harness(client, spec, agent_model, system_prompt,
                                   question, excludes, provider, arm,
                                   search_mode)
        wall_clock_s = time.perf_counter() - t0

        surfaced = agents.surfaced_domains(outcome["trajectory"])
        # The confound that quietly voids a search arm: the tool was available
        # and the model never used it, so the row is a no-search row wearing a
        # search arm's label. Weak OSS tool-calling and a native model that
        # decides not to search both land here. Filter on it in analysis — do
        # not average it away.
        search_available = search_mode != SEARCH_MODE_NONE
        zero_search_row = bool(search_available and outcome["used_searches"] == 0)

        hooks.metadata.update({
            # --- matrix axes: the four fields any slice should key on ---
            "model_class": spec.model_class,
            "model_vendor": model_vendor,
            "search_mode": search_mode,
            "search_provider": (
                provider if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else "none"
            ),
            # Freshness treatment applies to harness arms only; None elsewhere
            # keeps `native_fresh` from being read as "native search".
            "freshness_treatment": (
                arm if search_mode == SEARCH_MODE_HARNESS else None),
            # --- kept for continuity with runs made before the axes existed ---
            "provider": (
                provider if search_mode == SEARCH_MODE_HARNESS else "none"
                if search_mode == SEARCH_MODE_NONE else f"{model_vendor}_native"),
            "arm": arm,
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
            "exclusion_enforced": outcome["exclusion_enforced"],
            "excluded_source_domains": source_domains,
            "search_budget": MAX_SEARCHES,
            # --- per-row integrity flags ---
            "zero_search_row": zero_search_row,
            "bad_tool_calls": outcome["bad_tool_calls"],
            "refused_searches": outcome["refused_searches"],
            "refused_clicks": outcome["refused_clicks"],
            "search_errors": outcome["search_errors"],
            "model_refused": outcome["refused"],
            "answer_truncated": outcome["truncated"],
            "pause_turns": outcome["pause_turns"],
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
            agent_model, outcome["prompt_tokens"], outcome["completion_tokens"])
        cost_metrics = {"search_cost_usd": outcome["search_cost"]}
        if inference_cost is not None:
            cost_metrics["model_cost_usd"] = inference_cost
            cost_metrics["total_cost_usd"] = inference_cost + outcome["search_cost"]
            # The share of spend that is NOT search. Vals found this dominates;
            # logging it per row makes that checkable here rather than assumed.
            total = inference_cost + outcome["search_cost"]
            cost_metrics["search_share_of_cost"] = (
                outcome["search_cost"] / total if total else 0.0)
        hooks.metadata["model_cost_confirmed"] = cost_confirmed
        current_span().log(metrics={
            **cost_metrics,
            "latency_s": wall_clock_s,
            "used_searches": outcome["used_searches"],
            "used_clicks": outcome["used_clicks"],
            "refused_tool_calls": (
                outcome["refused_searches"] + outcome["refused_clicks"]),
            "agent_prompt_tokens": outcome["prompt_tokens"],
            "agent_completion_tokens": outcome["completion_tokens"],
            "agent_total_tokens": (
                outcome["prompt_tokens"] + outcome["completion_tokens"]),
            "answer_words": len(final.split()),
            "answer_chars": len(final),
            "distinct_surfaced_domains": len(surfaced),
            "zero_search_row": int(zero_search_row),
            "n_search_errors": len(outcome["search_errors"]),
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


def _blank_outcome(decision_surface: str) -> dict:
    return {
        "final_answer": "", "trajectory": [], "used_searches": 0,
        "used_clicks": 0, "refused_searches": 0, "refused_clicks": 0,
        "bad_tool_calls": 0, "search_cost": 0.0, "prompt_tokens": 0,
        "completion_tokens": 0, "decision_surface": decision_surface,
        "exclusion_enforced": False, "citations": [], "search_errors": [],
        "refused": False, "truncated": False, "pause_turns": 0,
    }


def _run_harness(client, spec, agent_model, system_prompt, question, excludes,
                 provider, arm, search_mode) -> dict:
    """Tool-calling loop for the harness arm and the no-tool control arm.

    Identical driver for both: the control arm is this loop with the tool never
    offered, so the two differ only in tool availability.
    """
    out = _blank_outcome(
        SURFACE_FULL if search_mode == SEARCH_MODE_HARNESS else SURFACE_NONE)
    out["exclusion_enforced"] = search_mode == SEARCH_MODE_HARNESS
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
        out["prompt_tokens"] += turn.prompt_tokens
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
                if out["used_searches"] >= MAX_SEARCHES:
                    out["refused_searches"] += 1
                    content = "Search budget exhausted."
                else:
                    out["used_searches"] += 1
                    query = call["arguments"].get("query", "")
                    results, content, tok = run_search(
                        provider, arm, query, excludes)
                    # Accumulated per call, not searches x flat rate: Exa bills
                    # content per page returned, so two calls returning
                    # different result counts do not cost the same.
                    out["search_cost"] += search_cost_usd(
                        provider, arm, len(results))
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
            client, agent_model, system_prompt, question, excludes)
    else:
        raise SystemExit(
            f"--search-mode native is unavailable for --model-vendor {spec.name}: "
            "no server-side search exists for this vendor.")

    rate, _confirmed = native_search_rate_usd(spec.name, agent_model)
    out = _blank_outcome(run.surface)
    out.update({
        "final_answer": run.final_answer or "I could not find this.",
        "trajectory": run.trajectory,
        "used_searches": run.n_searches,
        # Native search enforces max_uses server-side where the API supports it,
        # so a per-call refusal count is not observable. Recorded as 0 rather
        # than fabricated — budget_economy reads used_searches, not this.
        "search_cost": rate * run.n_searches,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "exclusion_enforced": run.exclusion_enforced,
        "citations": run.citations,
        "search_errors": run.search_errors,
        "refused": run.refused,
        "truncated": run.truncated,
        "pause_turns": run.pause_turns,
    })
    return out


# ---------------------------------------------------------------------------
# Eval run. Dataset push lives in import_livenewsbench.py — a second pusher
# here produced an incompatible schema (link/event_date under `expected`, no
# livenewsbench_release) that silently disabled leakage_guard and
# temporal_grounding. Do not reintroduce it.
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _preflight(provider: str, arm: str, search_mode: str, model_vendor: str,
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

    needed = [spec.api_key_env]
    if search_mode == SEARCH_MODE_HARNESS:
        needed.append(PROVIDER_KEYS[provider])
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing env var(s): {', '.join(missing)}")

    if search_mode == SEARCH_MODE_HARNESS and (provider, arm) not in SEARCH_PRICING:
        raise SystemExit(f"No SEARCH_PRICING entry for ({provider}, {arm}); "
                         "cost metrics would silently be $0.")

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
    """
    judges = []
    for spec in specs:
        model, _, base_url = spec.partition("@")
        if base_url:
            key = os.environ.get("JUDGE_API_KEY")
            if not key:
                raise SystemExit(
                    f"--judge {spec} needs JUDGE_API_KEY for {base_url}")
            judges.append((OpenAI(base_url=base_url, api_key=key), model))
        else:
            judges.append((OpenAI(), model))
    return judges


def run(provider: str, arm: str, dataset_name: str, dataset_version: str | None,
        trials: int, judge_specs: list[str], agent_model: str, study_id: str,
        env_path: Path, search_mode: str = SEARCH_MODE_HARNESS,
        model_vendor: str = DEFAULT_MODEL_VENDOR):
    api_key, project_id = load_runtime_env(env_path)
    _preflight(provider, arm, search_mode, model_vendor, agent_model)
    spec = vendor_of(model_vendor)

    dataset = init_dataset(project_id=project_id, name=dataset_name,
                           version=dataset_version, api_key=api_key)
    project_name = dataset.project.name
    # braintrust 0.25 exposes Dataset.version as a string property.
    resolved_version = dataset_version or dataset.version
    print(f"Dataset: {project_name}/{dataset.name} @ version {resolved_version}"
          f"{'' if dataset_version else '  (latest; pass --dataset-version to pin)'}")

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

    condition = condition_label(search_mode, provider, arm, model_vendor)
    condition_id = f"{model_vendor}:{agent_model}::{condition}"
    model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", agent_model)
    experiment_name = f"{study_id}-{dataset.name}-{model_slug}-{condition}"
    native_rate, rate_confirmed = native_search_rate_usd(model_vendor, agent_model)
    # Sampling parity is NOT achievable across vendors: Claude Opus 5 rejects
    # temperature/top_p/top_k, and Baseten does not document seed support. Record
    # what each arm received instead of implying a frozen sampling config.
    sampling = dict(spec.sampling)
    if spec.seed_supported:
        sampling["seed"] = agents.AGENT_SEED

    Eval(
        project_name,
        experiment_name=experiment_name,
        data=dataset,
        task=make_task(
            provider,
            arm,
            agent_model,
            study_id,
            dataset.name,
            str(resolved_version),
            search_mode,
            model_vendor,
        ),
        scores=[judge, *DETERMINISTIC_SCORERS],
        trial_count=trials,          # web nondeterminism > model nondeterminism
        max_concurrency=8,
        metadata={
            # --- matrix axes ---
            "model_class": spec.model_class,
            "model_vendor": model_vendor,
            "search_mode": search_mode,
            "search_provider": (
                provider if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else "none"),
            "freshness_treatment": (
                arm if search_mode == SEARCH_MODE_HARNESS else None),
            # --- legacy field names, kept so older experiments still join ---
            "provider": (
                provider if search_mode == SEARCH_MODE_HARNESS else "none"
                if search_mode == SEARCH_MODE_NONE else f"{model_vendor}_native"),
            "arm": arm, "agent_model": agent_model,
            "study_id": study_id,
            "condition_id": condition_id,
            "judge_models": [m for _, m in judges],
            "judge_mode": "single" if len(judges) == 1 else "jury",
            "dataset_name": dataset.name,
            "dataset_version": resolved_version,
            "dataset_version_pinned": bool(dataset_version),
            "budget": {"searches": MAX_SEARCHES, "clicks": MAX_CLICKS},
            # --- what the agent was actually configured with ---
            # No frontier vendor permits temperature/seed: gpt-5-family models
            # reject `temperature` with a 400 and support no `seed`, and Opus 5
            # rejects temperature/top_p/top_k. Only the OSS arm pins sampling, so
            # sampling_pinned is False on both frontier vendors by necessity.
            "sampling_params": sampling,
            "sampling_pinned": bool(sampling),
            "reasoning_effort": spec.reasoning_effort,
            "reasoning_effort_pinned": spec.reasoning_effort is not None,
            "agent_base_url": spec.base_url,
            # Whether the 5-search budget is API-enforced or only observed.
            # OpenAI's hosted web_search exposes no max_uses, so its native arm
            # can exceed the cap every other arm is held to — a real limit on the
            # native-vs-harness contrast within that vendor.
            "search_budget_enforced": (
                agents.NATIVE_BUDGET_ENFORCED.get(model_vendor, False)
                if search_mode == SEARCH_MODE_NATIVE
                else search_mode == SEARCH_MODE_HARNESS),
            # Publication date vs last-modified: two different constructs. Any
            # freshness claim must not pool arms with different semantics here.
            "date_field_semantics": agents.DATE_FIELD_SEMANTICS.get(
                provider if search_mode == SEARCH_MODE_HARNESS
                else f"{model_vendor}_native" if search_mode == SEARCH_MODE_NATIVE
                else None),
            "snippet_chars": (
                SNIPPET_CHARS if search_mode == SEARCH_MODE_HARNESS else None),
            "n_results": (
                N_RESULTS if search_mode == SEARCH_MODE_HARNESS else None),
            "exa_search_type": EXA_SEARCH_TYPE,
            "parallel_mode": PARALLEL_MODE,
            # The freshness treatment each provider actually received, so a run
            # is interpretable without reading the adapter source. Harness-only.
            "exa_max_age_hours": (
                EXA_MAX_AGE_HOURS
                if (search_mode, provider, arm) == (
                    SEARCH_MODE_HARNESS, "exa", "native_fresh") else None),
            "parallel_max_age_seconds": (
                PARALLEL_MAX_AGE_SECONDS
                if (search_mode, provider, arm) == (
                    SEARCH_MODE_HARNESS, "parallel", "native_fresh") else None),
            "youdotcom_freshness": (
                "day" if (search_mode, provider, arm) == (
                    SEARCH_MODE_HARNESS, "youdotcom", "native_fresh") else None),
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
            "git_commit": _git_commit(),
            "prompt_version": PROMPT_VERSIONS[search_mode],
        },
    )


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
    r.add_argument("--provider", choices=list(PROVIDERS), default="exa",
                   help="Search API for --search-mode harness. Ignored "
                        "otherwise.")
    r.add_argument("--arm", default="normalized",
                   choices=["normalized", "native_fresh", NO_SEARCH_ARM],
                   help="Freshness treatment for --search-mode harness. "
                        "no_search is a deprecated alias for "
                        "--search-mode none.")
    r.add_argument("--dataset-name", default=DATASET_NAME)
    r.add_argument("--dataset-version", default=None,
                   help="Pin a dataset version so every provider/arm sees the "
                        "same rows. Required unless --allow-latest is passed.")
    r.add_argument("--allow-latest", action="store_true",
                   help="Exploratory runs only: allow an unpinned dataset head.")
    r.add_argument("--trials", type=int, default=3)
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
                   help="Repeatable. One judge keeps SimpleQA parity; three or "
                        "more convene a majority-vote jury. Use model@base_url "
                        "with JUDGE_API_KEY for non-OpenAI routes.")
    r.add_argument("--env-file", type=Path, default=Path(".env"))

    args = ap.parse_args()
    if not args.dataset_version and not args.allow_latest:
        ap.error("--dataset-version is required for paired comparisons; pass "
                 "--allow-latest only for exploratory runs")
    if args.trials < 1:
        ap.error("--trials must be at least 1")

    search_mode = args.search_mode
    # `--arm no_search` predates the search_mode axis. Honor it so existing
    # scripts keep working, but map it onto the axis rather than carrying two
    # ways to express the same condition.
    if args.arm == NO_SEARCH_ARM:
        if search_mode not in (SEARCH_MODE_HARNESS, SEARCH_MODE_NONE):
            ap.error(f"--arm no_search conflicts with --search-mode "
                     f"{search_mode}; drop one")
        search_mode = SEARCH_MODE_NONE
    if search_mode == SEARCH_MODE_NATIVE and args.arm != "normalized":
        # native_fresh is a search-API parameter; there is no such knob on a
        # model vendor's server-side search, so accepting it would imply a
        # treatment that was never applied.
        ap.error("--arm applies only to --search-mode harness; the model "
                 "vendor's native search exposes no freshness parameter")

    agent_model = args.agent_model or VENDORS[args.model_vendor].default_model
    # gpt-4.1 default keeps parity with LiveNewsBench's published grading.
    run(args.provider, args.arm, args.dataset_name, args.dataset_version,
        args.trials, args.judges or ["gpt-4.1"], agent_model,
        args.study_id, args.env_file, search_mode, args.model_vendor)


if __name__ == "__main__":
    main()
