"""Starter Braintrust harness for web-search API x LLM freshness experiments.

Within one condition block, the agent is frozen (model snapshot, temperature,
prompt, tools, and budget); provider + arm are the treatment variables.
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

    # per (provider, arm): interleave these in time, don't run days apart
    python run_eval.py run --provider exa --arm native_fresh \
      --dataset-version <xact-id> --study-id freshness-v1 --trials 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from import_livenewsbench import DATASET_NAME, load_env
from corvus.compliance import require_provider_permission
from corvus.sources import SharedHostLimiter, retry_after_seconds
from scorers import (DETERMINISTIC_SCORERS, make_jury_grader,
                     make_simpleqa_grader)

# ---------------------------------------------------------------------------
# Credentials. AGENTS.md: the .env key always overrides any ambient credential.
# ---------------------------------------------------------------------------


RUNTIME_ENV_NAMES = (
    "BRAINTRUST_API_KEY",
    "BRAINTRUST_PROJECT_ID",
    "OPENAI_API_KEY",
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

AGENT_MODEL = "gpt-4o-2024-11-20"      # pin an exact snapshot; judge = gpt-4.1 (different lineage at minimum — ideally use a non-OpenAI judge)
MAX_SEARCHES, MAX_CLICKS = 5, 0
SNIPPET_CHARS = 400                    # normalized-arm snippet truncation
N_RESULTS = 8

# Exa search tier. Their default is "auto", which routes per query — pinning it
# keeps the retrieval tier a declared experiment condition.
EXA_SEARCH_TYPE = "auto"
# Parallel's Search API is beta and requires this version header.
PARALLEL_BETA = "search-extract-2025-10-10"
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
NO_SEARCH_ARM = "no_search"
NO_SEARCH_SYSTEM_PROMPT = """\
You are answering a time-sensitive factual question from memory. You have no tools and no web
access. When you know the answer, reply with ONLY the final answer, as concisely
as possible. If you do not know, reply exactly: I could not find this."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for news. Returns ranked results.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]

# Approximate per-call search pricing (USD) for native cost metrics.
# Verify against each vendor's current pricing page before publishing numbers.
SEARCH_COST = {
    ("exa", "normalized"): 0.005, ("exa", "native_fresh"): 0.005,       # + $1/1k pages if text requested
    ("parallel", "normalized"): 0.004, ("parallel", "native_fresh"): 0.009,  # base vs pro tier
    ("youdotcom", "normalized"): 0.004, ("youdotcom", "native_fresh"): 0.004,
}

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
    urls = []
    if isinstance(metadata, dict):
        if metadata.get("link"):
            urls.append(metadata["link"])
        for article in metadata.get("articles") or []:
            if isinstance(article, dict):
                url = article.get("link") or article.get("url")
                if url:
                    urls.append(url)
    if not urls and isinstance(expected, dict) and expected.get("link"):
        urls.append(expected["link"])

    seen, domains = set(), []
    for url in urls:
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
        "excludeDomains": exclude_domains,
        # numSentences/highlightsPerUrl are deprecated; maxCharacters is current
        # and matches the SNIPPET_CHARS budget enforced below.
        "contents": {"highlights": {"maxCharacters": SNIPPET_CHARS}},
    }
    if arm == "native_fresh":
        # Exa's two documented freshness knobs. livecrawl="preferred" is
        # deprecated, so this uses "always"; maxAgeHours accepts -1..720.
        body["contents"]["livecrawl"] = "always"
        body["contents"]["maxAgeHours"] = 24
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
    body = {
        "objective": query,
        "search_queries": [query[:200]],
        "processor": "pro" if arm == "native_fresh" else "base",
        "max_results": N_RESULTS,
        # Nested under `excerpts`; as a top-level field this was simply ignored.
        "excerpts": {"max_chars_per_result": 1500},
        # Ceiling is 200 combined include+exclude domains. exclude_domains is
        # source-first, so any truncation drops archive mirrors, not gold sources.
        "source_policy": {
            "exclude_domains": exclude_domains[:PARALLEL_MAX_DOMAINS]},
    }
    raw = _provider_json(
        "POST",
        "https://api.parallel.ai/v1beta/search",
        json=body,
        headers={
            "x-api-key": os.environ["PARALLEL_API_KEY"],
            "parallel-beta": PARALLEL_BETA,
            "content-type": "application/json",
        },
    )
    results = []
    for i, res in enumerate(raw.get("results", []), start=1):
        excerpts = res.get("excerpts") or []
        snippet = " ".join(excerpts)
        if arm == "normalized":
            # Truncating Parallel's long excerpts removes its core product
            # feature — a deliberate, documented limitation of this arm.
            snippet = snippet[:SNIPPET_CHARS]
        results.append({
            "rank": i,
            "url": res.get("url", ""),
            "title": res.get("title") or "",
            "snippet": snippet,
            "published_date": res.get("publish_date"),
        })
    return results, raw


def youdotcom_search(query: str, arm: str, exclude_domains: list[str]):
    # GET with query params against ydc-index.io — NOT a JSON POST to api.you.com.
    # exclude_domains is a single comma-separated string (repeated params are
    # unsupported) and is mutually exclusive with include_domains.
    params = {"query": query, "count": N_RESULTS,
              "exclude_domains": ",".join(exclude_domains)}
    if arm == "native_fresh":
        # day|week|month|year or YYYY-MM-DDtoYYYY-MM-DD. Broadest window that
        # still filters news; never derive this from event_date.
        params["freshness"] = "week"
        # NB: You.com also exposes a `livecrawl` param, but it only populates
        # result.contents — it would not change the snippet decision surface we
        # measure, so enabling it would add latency and cost for no signal.
    raw = _provider_json(
        "GET",
        "https://ydc-index.io/v1/search",
        params=params,
        headers={"X-API-Key": os.environ["YDC_API_KEY"]},
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
            "published_date": res.get("page_age") or res.get("published_date"),
        })
    return results, raw


PROVIDERS = {"exa": exa_search, "parallel": parallel_search,
             "youdotcom": youdotcom_search}


# ---------------------------------------------------------------------------
# Traced tools. type="tool" spans; native metrics without raw payload retention.
# notrace_io=True on both: @traced otherwise auto-logs the return value over
# the explicit output= below, and these return tuples, not the payload we want.
# ---------------------------------------------------------------------------

@traced(type="tool", name="search_web", notrace_io=True)
def run_search(provider: str, arm: str, query: str, exclude_domains: list[str]):
    t0 = time.perf_counter()
    results, _raw = PROVIDERS[provider](query, arm, exclude_domains)
    latency = time.perf_counter() - t0
    rendered = "\n".join(
        f"[{r['rank']}] {r['title']}\n    {r['url']}\n    "
        f"published: {r['published_date'] or 'unknown'}\n    {r['snippet']}"
        for r in results) or "No results."
    current_span().log(
        input={"query": query, "provider": provider, "arm": arm,
               "exclude_domains": exclude_domains},
        output=results,
        metrics={"tokens": _tok(rendered),
                 "latency_s": latency,
                 "search_cost_usd": SEARCH_COST.get((provider, arm), 0.0),
                 "n_results": len(results)},
    )
    return results, rendered, _tok(rendered)


# ---------------------------------------------------------------------------
# Frozen agent loop
# ---------------------------------------------------------------------------

# Built lazily: constructing OpenAI() at import time raises on a missing
# OPENAI_API_KEY, which would crash `--help` and preempt _preflight's clear
agent_client = None


def get_agent_client():
    global agent_client
    if agent_client is None:
        agent_client = wrap_openai(OpenAI())
    return agent_client


def make_task(
    provider: str,
    arm: str,
    agent_model: str = AGENT_MODEL,
    study_id: str = "freshness-v1",
    dataset_name: str = DATASET_NAME,
    dataset_version: str | None = None,
):

    def task(input: dict, hooks) -> dict:
        # Leakage excludes come from row METADATA (link + articles[*]), which is
        # where the importer puts them and where leakage_guard reads them.
        # `expected` is the answer string, so it has no .get().
        row_metadata = hooks.metadata or {}
        source_domains = source_domains_of(row_metadata, hooks.expected)
        excludes = source_domains + ARCHIVE_EXCLUDES
        no_search = arm == NO_SEARCH_ARM
        condition = "no_search" if no_search else f"{provider}-{arm}"
        condition_id = f"{agent_model}::{condition}"
        question = input["question"]
        task_key = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        benchmark_category = (
            row_metadata.get("category")
            or row_metadata.get("event_category")
            or row_metadata.get("data_source")
            or "uncategorized"
        )

        messages = [{"role": "system", "content":
                     NO_SEARCH_SYSTEM_PROMPT if no_search else FROZEN_SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
        trajectory, searches, clicks = [], 0, 0
        prompt_tokens, completion_tokens = 0, 0
        # Attempts REFUSED because the search budget was already spent. The cap
        # is the benchmark protocol, but it also hides tool-call runaway, which is a
        # provider-attributable cost failure mode elsewhere in the literature.
        # Counting refusals recovers that signal without relaxing the cap.
        refused_searches, refused_clicks = 0, 0
        bad_tool_calls = 0
        final = None

        for _ in range(2 * (MAX_SEARCHES + MAX_CLICKS) + 2):
            # Once both budgets are spent there is nothing left to call, so drop
            # the tools and force a final answer instead of burning turns on
            # "budget exhausted" replies until the loop cap trips.
            out_of_budget = searches >= MAX_SEARCHES and clicks >= MAX_CLICKS
            kwargs = {} if (no_search or out_of_budget) else {"tools": TOOLS}
            resp = get_agent_client().chat.completions.create(
                model=agent_model, temperature=0, seed=42,
                messages=messages, **kwargs)
            usage = getattr(resp, "usage", None)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                final = (msg.content or "").strip()
                break
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                if name == "search_web":
                    if searches >= MAX_SEARCHES:
                        refused_searches += 1
                        content = "Search budget exhausted."
                    else:
                        searches += 1
                        query = args.get("query", "")
                        results, content, tok = run_search(
                            provider, arm, query, excludes)
                        trajectory.append({"type": "search", "query": query,
                                           "tokens": tok, "results": results})
                else:
                    bad_tool_calls += 1
                    content = f"Unknown tool: {name}. Use search_web."
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": content})
        if final is None:
            final = "I could not find this."

        surfaced_domains = {
            _domain_of(result.get("url", ""))
            for step in trajectory if step.get("type") == "search"
            for result in step.get("results", [])
            if _domain_of(result.get("url", ""))
        }
        hooks.metadata.update({
            "provider": "none" if no_search else provider,
            "arm": arm, "agent_model": agent_model,
            "study_id": study_id,
            "condition_id": condition_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "trial_index": hooks.trial_index,
            "task_key": task_key,
            "benchmark_category": benchmark_category,
            "excluded_source_domains": source_domains,
            "bad_tool_calls": bad_tool_calls,
            "refused_searches": refused_searches,
            "refused_clicks": refused_clicks,
            "as_of": datetime.now(timezone.utc).isoformat(),
        })
        # Task-span metrics; Braintrust rolls these into the experiment summary.
        # search_cost_usd is search spend ONLY — model inference cost comes from
        # the wrapped LLM spans, so keep the two decomposable rather than summing
        # them here. Total cost per row is an analysis-side join of the two.
        current_span().log(metrics={
            "search_cost_usd": searches * SEARCH_COST.get((provider, arm), 0.0),
            "used_searches": searches, "used_clicks": clicks,
            "refused_tool_calls": refused_searches + refused_clicks,
            "agent_prompt_tokens": prompt_tokens,
            "agent_completion_tokens": completion_tokens,
            "agent_total_tokens": prompt_tokens + completion_tokens,
            "answer_words": len(final.split()),
            "answer_chars": len(final),
            "distinct_surfaced_domains": len(surfaced_domains),
        })
        return {"final_answer": final, "trajectory": trajectory,
                "used_searches": searches, "used_clicks": clicks,
                "refused_searches": refused_searches,
                "refused_clicks": refused_clicks,
                "agent_prompt_tokens": prompt_tokens,
                "agent_completion_tokens": completion_tokens}

    return task


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


def _preflight(provider: str, arm: str) -> None:
    """Fail before spending money, not on row 1 of N."""
    needed = ["OPENAI_API_KEY"]
    if arm != NO_SEARCH_ARM:
        require_provider_permission(provider)
        needed.append(PROVIDER_KEYS[provider])
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing env var(s): {', '.join(missing)}")
    if arm != NO_SEARCH_ARM and (provider, arm) not in SEARCH_COST:
        raise SystemExit(f"No SEARCH_COST entry for ({provider}, {arm}); "
                         "cost metrics would silently be $0.")


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
        env_path: Path):
    api_key, project_id = load_runtime_env(env_path)
    _preflight(provider, arm)

    dataset = init_dataset(project_id=project_id, name=dataset_name,
                           version=dataset_version, api_key=api_key)
    project_name = dataset.project.name
    resolved_version = dataset_version or dataset.version()
    print(f"Dataset: {project_name}/{dataset.name} @ version {resolved_version}"
          f"{'' if dataset_version else '  (latest; pass --dataset-version to pin)'}")

    judges = build_judges(judge_specs)
    # One judge keeps SimpleQA parity with LiveNewsBench's published numbers;
    # several convene a jury. Both emit exactly one score per row.
    judge = (make_simpleqa_grader(judges[0][0], judge_model=judges[0][1])
             if len(judges) == 1 else make_jury_grader(judges))
    if arm == NO_SEARCH_ARM:
        print("Control arm: no tools. Subtract this from provider arms to get "
              "retrieval's marginal value.")
    condition = "no_search" if arm == NO_SEARCH_ARM else f"{provider}-{arm}"
    condition_id = f"{agent_model}::{condition}"
    model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", agent_model)
    experiment_name = f"{study_id}-{dataset.name}-{model_slug}-{condition}"

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
        ),
        scores=[judge, *DETERMINISTIC_SCORERS],
        trial_count=trials,          # web nondeterminism > model nondeterminism
        max_concurrency=8,
        metadata={
            "provider": "none" if arm == NO_SEARCH_ARM else provider,
            "arm": arm, "agent_model": agent_model,
            "study_id": study_id,
            "condition_id": condition_id,
            "judge_models": [m for _, m in judges],
            "judge_mode": "single" if len(judges) == 1 else "jury",
            "dataset_name": dataset.name,
            "dataset_version": resolved_version,
            "dataset_version_pinned": bool(dataset_version),
            "budget": {"searches": MAX_SEARCHES, "clicks": MAX_CLICKS},
            "snippet_chars": SNIPPET_CHARS, "n_results": N_RESULTS,
            "exa_search_type": EXA_SEARCH_TYPE,
            "git_commit": _git_commit(),
            "prompt_version": "frozen-v2-generic-factual",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--provider", choices=list(PROVIDERS), default="exa",
                   help="Ignored when --arm no_search.")
    r.add_argument("--arm", required=True,
                   choices=["normalized", "native_fresh", NO_SEARCH_ARM])
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
    r.add_argument("--agent-model", default=AGENT_MODEL,
                   help="Provider rankings can be model-dependent; re-run the "
                        "matrix under a second agent model to check that yours "
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
    # gpt-4.1 default keeps parity with LiveNewsBench's published grading.
    run(args.provider, args.arm, args.dataset_name, args.dataset_version,
        args.trials, args.judges or ["gpt-4.1"], args.agent_model,
        args.study_id, args.env_file)


if __name__ == "__main__":
    main()
