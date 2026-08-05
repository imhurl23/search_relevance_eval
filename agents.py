"""Agent instrumentation across model vendors and search modalities.

This module exists so one frozen benchmark protocol can be run under four
materially different agent configurations:

    model_class  x  search_mode
    -----------------------------------------------------------------
    oss          x  harness   (Baseten model, our search_web tool)
    oss          x  none      (Baseten model, parametric memory only)
    frontier     x  harness   (OpenAI/Anthropic model, our search_web tool)
    frontier     x  native    (the model provider's own server-side search)
    frontier     x  none      (parametric baseline for the frontier arms)

`native` is only attributable *within* a vendor: if GPT gets native search and
Claude gets the harness tool, model identity is confounded with search mode. So
each frontier vendor runs all three modes and `native` is compared to that same
vendor's `harness` and `none` arms.

The hard instrumentation problem this module solves
---------------------------------------------------
Native (server-side) search does not produce `search_web` tool calls, so the
harness sees no trajectory. Six of the ten deterministic scorers read
`output["trajectory"][*]["results"]`, and with an empty trajectory they return
*passing* scores: leakage_guard 1.0 (nothing leaked because nothing was
observed), budget_economy 1.0 (zero searches is within budget), and
search_cost_usd $0.00. A native arm would therefore look free, compliant, and
leak-clean purely by construction.

The fix has two halves:

  1. Normalize every native search into the SAME trajectory schema the harness
     emits, so the scorers that can still be computed keep working.
  2. Publish a `decision_surface` tier alongside it, declaring which fields of
     that schema are actually populated. scorers.py gates on this tier and
     returns None — never a passing score — for anything it cannot observe.

Decision-surface tiers, and why they differ
-------------------------------------------
    full        rank, url, title, snippet, published_date   harness arms
    no_snippet  rank, url, title, published_date            Anthropic native
    urls_only   rank, url, (title from citations)           OpenAI native
    none        nothing                                     no_search arms

Anthropic's `web_search_result` carries url/title/page_age but no snippet.
OpenAI's `web_search_call.action.sources` carries urls only, with titles
recoverable from `url_citation` annotations and no dates at all. That asymmetry
is a property of the vendors, not a bug here, and it is why native and harness
arms cannot share a single headline decision-surface number.

Deliberately NOT done: Anthropic citations carry `cited_text` (<=150 chars), and
mapping that into the `snippet` field would make the snippet scorers appear
computable on the native arm. It would also guarantee a near-perfect score,
because cited_text is selected *precisely because* it supports the answer. That
is a spurious win, so citations are recorded separately and `snippet` stays
empty.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from openai import OpenAI

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

SEARCH_MODE_NONE = "none"
SEARCH_MODE_HARNESS = "harness"
SEARCH_MODE_NATIVE = "native"
SEARCH_MODES = (SEARCH_MODE_NONE, SEARCH_MODE_HARNESS, SEARCH_MODE_NATIVE)

# Decision-surface observability tiers. scorers.py imports these names.
SURFACE_FULL = "full"
SURFACE_NO_SNIPPET = "no_snippet"
SURFACE_URLS_ONLY = "urls_only"
SURFACE_NONE = "none"


# ---------------------------------------------------------------------------
# Anthropic request tuning. Declared above the vendor registry because the
# registry records the effort level as part of the frozen condition.
# ---------------------------------------------------------------------------

# web_search_20250305 is the BASIC tool, chosen deliberately over the newer
# _20260209/_20260318 versions. Those add "dynamic filtering", which runs the
# search inside code execution and drops results before they reach the context
# window — the model would then answer from a surface we cannot observe and that
# does not correspond to the harness arms, where every returned result is
# rendered into the prompt. _20260318's response_inclusion can also omit result
# blocks entirely. Basic search keeps the decision surface both complete and
# comparable, at the cost of more input tokens.
ANTHROPIC_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
# Thinking stays on (adaptive). Disabling it on Opus 5 has a documented failure
# mode where a tool call is written into the visible text instead of emitted as
# a tool_use block: the turn succeeds, the search never runs, and nothing
# errors. On a search eval that would silently produce no-search rows.
ANTHROPIC_THINKING = {"type": "adaptive"}
ANTHROPIC_EFFORT = "high"
# max_tokens caps thinking + text together on Opus 5, so leave headroom or the
# answer truncates mid-sentence and scores as a wrong answer.
ANTHROPIC_MAX_TOKENS = 8192
ANTHROPIC_MAX_PAUSE_TURNS = 4


# ---------------------------------------------------------------------------
# Vendor registry
#
# Every field here is an experiment condition, not a convenience. In particular
# `sampling` and `seed_supported` differ by vendor and CANNOT be equalized.
# Neither frontier vendor accepts sampling parameters at all: Claude Opus 5
# rejects temperature/top_p/top_k with a 400, and gpt-5-family models reject
# `temperature` ("only the default (1) is supported") and offer no `seed`. Only
# the OSS arm can pin them. So "temperature 0, seed 42" is not a property the
# frozen agent can hold across vendors — it is unavailable, not declined. The run
# records what each arm actually received (`sampling_pinned`) rather than
# implying parity, and `seed_supported` stays False everywhere as a result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VendorSpec:
    name: str
    model_class: str            # "oss" | "frontier"
    api_key_env: str
    default_model: str
    base_url: str | None = None
    supports_native_search: bool = False
    # Sampling params safe to send to this vendor's chat/messages endpoint.
    sampling: dict = field(default_factory=dict)
    seed_supported: bool = False
    # Reasoning depth, where the vendor lets us pin it. None = vendor default,
    # left unpinned. Recorded either way so a run is never ambiguous about it.
    reasoning_effort: str | None = None
    notes: str = ""


VENDORS: dict[str, VendorSpec] = {
    # Baseten Model APIs are OpenAI-Chat-Completions-compatible at
    # https://inference.baseten.co/v1 (docs.baseten.co/inference/model-apis/overview,
    # checked 2026-08-05). All catalog models support tool calling. gpt-oss-120b
    # is the open-weights default because the OSS+harness cell depends entirely
    # on reliable function calling — a model that never emits a tool call
    # silently collapses that cell into the OSS+none cell.
    "baseten": VendorSpec(
        name="baseten",
        model_class="oss",
        api_key_env="BASETEN_API_KEY",
        default_model="openai/gpt-oss-120b",
        base_url="https://inference.baseten.co/v1",
        supports_native_search=False,
        sampling={"temperature": 0},
        seed_supported=False,
        notes="Server-side search is structurally unavailable; native arm N/A.",
    ),
    # gpt-5.6-sol is pinned rather than the bare `gpt-5.6` alias, which points at
    # sol today and will move. Sol is the tier-matched counterpart to
    # claude-opus-5 ($5/$30 vs $5/$25); terra and luna are cheaper tiers and
    # would confound the frontier comparison with a capability-tier difference.
    #
    # NOTE the empty sampling dict: gpt-5-family reasoning models reject
    # `temperature` with a 400 ("only the default (1) is supported"), and do not
    # support `seed`. "temperature 0, seed 42" is therefore unavailable on this
    # vendor too — it is not a knob we chose not to use.
    #
    # reasoning_effort is left unpinned on purpose. Reasoning models reject
    # `reasoning_effort` alongside function tools on /v1/chat/completions, which
    # the harness arm uses; pinning it on the native arm alone would make effort
    # differ between this vendor's own native and harness arms — the one contrast
    # the matrix exists to measure. Both arms therefore run the vendor default.
    "openai": VendorSpec(
        name="openai",
        model_class="frontier",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5.6-sol",
        supports_native_search=True,
        sampling={},
        seed_supported=False,
        reasoning_effort=None,
        notes="Native search via the Responses API hosted web_search tool.",
    ),
    "anthropic": VendorSpec(
        name="anthropic",
        model_class="frontier",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-opus-5",
        supports_native_search=True,
        # Opus 5 removed temperature/top_p/top_k — sending any of them is a 400.
        sampling={},
        seed_supported=False,
        # Pinnable here because effort and tools coexist fine on the Messages
        # API, so both this vendor's arms get the same declared depth.
        reasoning_effort=ANTHROPIC_EFFORT,
        notes="Native search via the server-side web_search tool.",
    ),
}


# Whether the 5-search budget is ENFORCED by the API, or merely observed.
# Anthropic's web_search takes max_uses; OpenAI's hosted web_search publishes no
# equivalent, so its native arm can exceed the budget every other arm is held to.
# That is a real threat to the native-vs-harness contrast within OpenAI, so it is
# recorded per run rather than papered over. budget_economy still scores the
# observed count, which is what surfaces a violation.
NATIVE_BUDGET_ENFORCED = {"anthropic": True, "openai": False}

# What each search layer's date field actually MEANS. temporal_grounding treats
# `published_date` as a publication timestamp, but two of these surfaces report
# last-modified instead, which is a different construct: a re-rendered page can
# look fresh without carrying new information. Rows record which semantic they
# got so a freshness claim is not pooled across the two.
DATE_FIELD_SEMANTICS = {
    "exa": "publication",             # publishedDate
    "parallel": "publication",        # publish_date
    "youdotcom": "last_modified",     # page_age
    "anthropic_native": "last_modified",  # page_age
    "openai_native": None,            # no date field at all
}


def vendor_of(name: str) -> VendorSpec:
    try:
        return VENDORS[name]
    except KeyError:
        raise ValueError(
            f"unknown model vendor {name!r}; expected one of {sorted(VENDORS)}"
        ) from None


# ---------------------------------------------------------------------------
# Native search pricing
#
# Both vendors publish $10 per 1,000 searches, which makes the two native arms
# directly cost-comparable to each other. Search *content* tokens are billed at
# model rates on top, and those already land on the traced LLM spans — keep the
# two decomposable rather than folding them together here.
#
#   Anthropic (platform.claude.com/docs/en/agents-and-tools/tool-use/
#              web-search-tool, checked 2026-08-05): $10/1k searches.
#   OpenAI    (developers.openai.com/api/docs/pricing, checked 2026-08-05):
#              $10/1k calls on reasoning models; the non-reasoning
#              `web_search_preview` path is $25/1k instead. We pin a reasoning
#              model so the $10 rate is the one that applies, and record the
#              rate in run metadata so the assumption is auditable.
# ---------------------------------------------------------------------------

NATIVE_SEARCH_USD_PER_CALL = {"openai": 0.010, "anthropic": 0.010}

# Models on which OpenAI bills the standard `web_search` rate. A non-reasoning
# model routes to `web_search_preview` at 2.5x the price, so an unrecognized
# model is flagged rather than silently priced at the cheaper rate.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def native_search_rate_usd(vendor: str, model: str) -> tuple[float, bool]:
    """Return (usd_per_search, rate_is_confirmed_for_this_model)."""
    rate = NATIVE_SEARCH_USD_PER_CALL.get(vendor, 0.0)
    if vendor != "openai":
        return rate, True
    confirmed = model.startswith(_OPENAI_REASONING_PREFIXES)
    return (rate if confirmed else 0.025), confirmed


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def get_client(vendor: str, wrap):
    """Build (and memoize) one traced client per vendor.

    `wrap` is braintrust's wrap_openai / wrap_anthropic, injected so this module
    has no braintrust import of its own and stays unit-testable offline.
    """
    if vendor in _clients:
        return _clients[vendor]
    spec = vendor_of(vendor)
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise SystemExit(f"{spec.api_key_env} is required for --model-vendor {vendor}")
    if vendor == "anthropic":
        import anthropic  # imported lazily so the OSS/OpenAI paths need no install

        client = wrap(anthropic.Anthropic(api_key=key))
    else:
        client = wrap(OpenAI(api_key=key, base_url=spec.base_url))
    _clients[vendor] = client
    return client


def reset_clients() -> None:
    """Test hook — the memo above would otherwise leak between cases."""
    _clients.clear()


# ---------------------------------------------------------------------------
# Native search results
# ---------------------------------------------------------------------------


@dataclass
class NativeRun:
    """One native-search turn, normalized onto the harness trajectory schema."""

    final_answer: str
    # [{"type": "search", "query": str, "tokens": int, "results": [...]}]
    trajectory: list = field(default_factory=list)
    n_searches: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    surface: str = SURFACE_NONE
    # Vendor-reported search count, when the API publishes one. Preferred over
    # counting parsed blocks: a search whose results were dropped from the
    # response still consumed budget and still costs money.
    vendor_search_count: int | None = None
    exclusion_enforced: bool = False
    citations: list = field(default_factory=list)
    search_errors: list = field(default_factory=list)
    refused: bool = False
    pause_turns: int = 0
    truncated: bool = False


def _domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# Anthropic caps a domain filter by request size rather than a documented count
# (an over-long list returns `request_too_large`). Ordered source-domains-first
# upstream, so truncation drops archive mirrors and never a gold source.
ANTHROPIC_MAX_BLOCKED_DOMAINS = 50
# OpenAI documents up to 100 entries in filters.blocked_domains.
OPENAI_MAX_BLOCKED_DOMAINS = 100


# ---------------------------------------------------------------------------
# Anthropic native search
# ---------------------------------------------------------------------------

def anthropic_native_search(
    client,
    model: str,
    system_prompt: str,
    question: str,
    exclude_domains: list[str],
    max_searches: int,
) -> NativeRun:
    """Run one question through Claude's server-side web_search tool."""
    tool = {
        "type": ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        # A hard cap, matching the harness arms' search budget exactly.
        "max_uses": max_searches,
    }
    blocked = exclude_domains[:ANTHROPIC_MAX_BLOCKED_DOMAINS]
    if blocked:
        # Gold-source exclusion IS enforceable here, so the leakage rule applies
        # to this arm on the same terms as the harness arms.
        tool["blocked_domains"] = blocked

    run = NativeRun(
        final_answer="",
        surface=SURFACE_NO_SNIPPET,
        exclusion_enforced=bool(blocked),
    )
    messages = [{"role": "user", "content": question}]
    texts: list[str] = []

    for _ in range(ANTHROPIC_MAX_PAUSE_TURNS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=[tool],
            thinking=ANTHROPIC_THINKING,
            output_config={"effort": ANTHROPIC_EFFORT},
        )
        usage = _attr(response, "usage")
        run.prompt_tokens += int(_attr(usage, "input_tokens") or 0)
        run.completion_tokens += int(_attr(usage, "output_tokens") or 0)
        server_use = _attr(usage, "server_tool_use")
        requests = _attr(server_use, "web_search_requests")
        if requests is not None:
            run.vendor_search_count = (run.vendor_search_count or 0) + int(requests)

        _parse_anthropic_content(response, run, texts)

        stop = _attr(response, "stop_reason")
        if stop == "refusal":
            # A 200 with an empty/partial body. Recording it as a refusal keeps
            # it out of the accuracy numerator instead of scoring as a wrong
            # answer, which would misattribute a policy decline to retrieval.
            run.refused = True
            break
        if stop == "max_tokens":
            run.truncated = True
            break
        if stop != "pause_turn":
            break
        # A paused server-tool turn resumes by sending the assistant message
        # back unchanged — encrypted_content included, or the API 400s.
        run.pause_turns += 1
        messages = messages + [
            {"role": "assistant", "content": _attr(response, "content")}]

    run.final_answer = " ".join(t for t in texts if t).strip()
    run.n_searches = (
        run.vendor_search_count
        if run.vendor_search_count is not None
        else len(run.trajectory)
    )
    return run


def _parse_anthropic_content(response, run: NativeRun, texts: list[str]) -> None:
    """Walk content blocks, pairing server_tool_use queries with their results."""
    pending: dict[str, str] = {}
    for block in _attr(response, "content") or []:
        btype = _attr(block, "type")
        if btype == "text":
            texts.append(_attr(block, "text") or "")
            for citation in _attr(block, "citations") or []:
                run.citations.append({
                    "url": _attr(citation, "url") or "",
                    "title": _attr(citation, "title") or "",
                    # cited_text is intentionally NOT promoted into a snippet;
                    # see this module's docstring.
                    "cited_text": _attr(citation, "cited_text") or "",
                })
        elif btype == "server_tool_use" and _attr(block, "name") == "web_search":
            args = _attr(block, "input") or {}
            query = args.get("query", "") if isinstance(args, dict) else ""
            pending[_attr(block, "id") or ""] = query
        elif btype == "web_search_tool_result":
            query = pending.pop(_attr(block, "tool_use_id") or "", "")
            content = _attr(block, "content")
            # On an error `content` is a single object, not a list. Branching
            # here matters: an error round still spent budget, and treating it
            # as "no results" would read as a clean empty SERP.
            if isinstance(content, dict) or _attr(content, "error_code"):
                run.search_errors.append({
                    "query": query,
                    "error_code": (
                        content.get("error_code")
                        if isinstance(content, dict)
                        else _attr(content, "error_code")
                    ),
                })
                continue
            results = []
            for i, item in enumerate(content or [], start=1):
                results.append({
                    "rank": i,
                    "url": _attr(item, "url") or "",
                    "title": _attr(item, "title") or "",
                    # No snippet field exists on web_search_result.
                    "snippet": "",
                    "published_date": _attr(item, "page_age"),
                })
            run.trajectory.append({
                "type": "search",
                "query": query,
                # Token cost of native search results is billed on the model
                # spans, not attributable per-search here. 0 keeps the
                # token-discounted scorers from reading a fabricated number;
                # they are gated off on this surface anyway.
                "tokens": 0,
                "results": results,
            })


# ---------------------------------------------------------------------------
# OpenAI native search
# ---------------------------------------------------------------------------

# `web_search` is the current type for new Responses API integrations
# (developers.openai.com/api/docs/guides/tools-web-search, checked 2026-08-05).
OPENAI_WEB_SEARCH_TOOL_TYPE = "web_search"
# low | medium | high. Pinned so the retrieval depth is a declared condition
# rather than a per-request default that can shift under us.
OPENAI_SEARCH_CONTEXT_SIZE = "medium"


def openai_native_search(
    client,
    model: str,
    system_prompt: str,
    question: str,
    exclude_domains: list[str],
) -> NativeRun:
    """Run one question through OpenAI's hosted Responses web_search tool."""
    tool: dict = {
        "type": OPENAI_WEB_SEARCH_TOOL_TYPE,
        "search_context_size": OPENAI_SEARCH_CONTEXT_SIZE,
    }
    blocked = exclude_domains[:OPENAI_MAX_BLOCKED_DOMAINS]
    if blocked:
        tool["filters"] = {"blocked_domains": blocked}

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=question,
        tools=[tool],
        # Without this include, the response carries only inline url_citation
        # annotations — the subset the answer happened to cite, not the surface
        # the model actually consulted. Scoring leakage or diversity off
        # citations alone would understate both.
        include=["web_search_call.action.sources"],
    )

    run = NativeRun(
        final_answer="",
        # URLs only: no per-result dates and no snippets, so temporal grounding
        # and every snippet-derived metric are unobservable on this arm.
        surface=SURFACE_URLS_ONLY,
        exclusion_enforced=bool(blocked),
    )
    usage = _attr(response, "usage")
    run.prompt_tokens = int(_attr(usage, "input_tokens") or 0)
    run.completion_tokens = int(_attr(usage, "output_tokens") or 0)

    titles: dict[str, str] = {}
    texts: list[str] = []
    for item in _attr(response, "output") or []:
        itype = _attr(item, "type")
        if itype == "web_search_call":
            action = _attr(item, "action") or {}
            query = _attr(action, "query") or ""
            sources = _attr(action, "sources") or []
            results = []
            for i, source in enumerate(sources, start=1):
                results.append({
                    "rank": i,
                    "url": _attr(source, "url") or "",
                    "title": "",
                    "snippet": "",
                    "published_date": None,
                })
            if _attr(item, "status") not in (None, "completed"):
                run.search_errors.append({
                    "query": query, "error_code": _attr(item, "status")})
            run.trajectory.append({
                "type": "search", "query": query, "tokens": 0,
                "results": results,
            })
        elif itype == "message":
            for part in _attr(item, "content") or []:
                texts.append(_attr(part, "text") or "")
                for note in _attr(part, "annotations") or []:
                    if _attr(note, "type") != "url_citation":
                        continue
                    url = _attr(note, "url") or ""
                    title = _attr(note, "title") or ""
                    if url and title:
                        titles[url] = title
                    run.citations.append({"url": url, "title": title})

    # Backfill the titles the citations disclosed. Partial by construction — an
    # uncited result stays untitled — which is why this arm is urls_only.
    for step in run.trajectory:
        for result in step["results"]:
            if not result["title"]:
                result["title"] = titles.get(result["url"], "")

    run.final_answer = " ".join(t for t in texts if t).strip()
    run.vendor_search_count = len(run.trajectory)
    run.n_searches = len(run.trajectory)
    return run


def _attr(obj, name):
    """Read a field from an SDK model or a plain dict.

    The adapters are exercised offline against dict fixtures and online against
    pydantic response objects; one accessor keeps a single code path under test.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------
# Harness search mode: one tool-calling loop, two wire protocols
#
# The harness arm has to run on OpenAI/Baseten chat completions AND on
# Anthropic's Messages API, because "native vs harness" is only attributable
# within a vendor — Claude needs a harness arm of its own to compare its native
# arm against. Rather than branch inside the agent loop (which would make "the
# agent is frozen across arms" hard to substantiate), each vendor supplies a
# session object with the same three operations and run_eval drives one loop.
# ---------------------------------------------------------------------------

SEARCH_TOOL_NAME = "search_web"
SEARCH_TOOL_DESCRIPTION = "Search the web for news. Returns ranked results."
SEARCH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


@dataclass
class Turn:
    """One assistant turn, normalized across wire protocols."""

    text: str = ""
    # [{"id": str, "name": str, "arguments": dict, "malformed": bool}]
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop: str | None = None
    refused: bool = False
    truncated: bool = False


class HarnessSession:
    """Base class holding the vendor-neutral parts of the loop."""

    surface = SURFACE_FULL

    def __init__(self, client, spec: VendorSpec, model: str, system_prompt: str):
        self.client = client
        self.spec = spec
        self.model = model
        self.system_prompt = system_prompt

    def add_user(self, question: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, tools_enabled: bool) -> Turn:  # pragma: no cover - interface
        raise NotImplementedError

    def add_tool_results(self, results: list) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def sampling_used(self) -> dict:
        """Sampling parameters actually sent, for run metadata."""
        params = dict(self.spec.sampling)
        if self.spec.seed_supported:
            params["seed"] = AGENT_SEED
        return params


AGENT_SEED = 42


class OpenAIHarnessSession(HarnessSession):
    """Chat Completions tool calling — OpenAI and Baseten (OpenAI-compatible)."""

    def __init__(self, client, spec, model, system_prompt):
        super().__init__(client, spec, model, system_prompt)
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]
        self.tools = [{
            "type": "function",
            "function": {
                "name": SEARCH_TOOL_NAME,
                "description": SEARCH_TOOL_DESCRIPTION,
                "parameters": SEARCH_TOOL_PARAMETERS,
            },
        }]

    def add_user(self, question: str) -> None:
        self.messages.append({"role": "user", "content": question})

    def step(self, tools_enabled: bool) -> Turn:
        kwargs = dict(self.spec.sampling)
        if self.spec.seed_supported:
            kwargs["seed"] = AGENT_SEED
        if tools_enabled:
            kwargs["tools"] = self.tools
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, **kwargs)
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        turn = Turn(
            text=(getattr(message, "content", None) or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop=getattr(response.choices[0], "finish_reason", None),
        )
        turn.truncated = turn.stop == "length"
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            self.messages.append(message)
        for call in calls:
            raw = getattr(call.function, "arguments", None) or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args, malformed = {}, True
            else:
                malformed = not isinstance(args, dict)
                if malformed:
                    args = {}
            turn.tool_calls.append({
                "id": call.id, "name": call.function.name,
                "arguments": args, "malformed": malformed,
            })
        return turn

    def add_tool_results(self, results: list) -> None:
        for call_id, content in results:
            self.messages.append({
                "role": "tool", "tool_call_id": call_id, "content": content})


class AnthropicHarnessSession(HarnessSession):
    """Messages API tool calling, so Claude can run the harness search arm."""

    def __init__(self, client, spec, model, system_prompt):
        super().__init__(client, spec, model, system_prompt)
        self.messages: list = []
        self._used_tools = False
        self.tools = [{
            "name": SEARCH_TOOL_NAME,
            "description": SEARCH_TOOL_DESCRIPTION,
            "input_schema": SEARCH_TOOL_PARAMETERS,
        }]

    def add_user(self, question: str) -> None:
        self.messages.append({"role": "user", "content": question})

    def step(self, tools_enabled: bool) -> Turn:
        kwargs: dict = {}
        if tools_enabled:
            kwargs["tools"] = self.tools
        elif self._used_tools:
            # The OpenAI path forces a final answer by dropping `tools`. That is
            # not safe here: history already contains tool_use/tool_result blocks,
            # and a Messages request carrying those without the tool declared is
            # rejected. Keep the declaration and forbid further calls instead —
            # same effect on the agent, valid request.
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = {"type": "none"}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=self.system_prompt,
            messages=self.messages,
            thinking=ANTHROPIC_THINKING,
            output_config={"effort": self.spec.reasoning_effort},
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        stop = getattr(response, "stop_reason", None)
        turn = Turn(
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop=stop,
            refused=stop == "refusal",
            truncated=stop == "max_tokens",
        )
        texts = []
        content = getattr(response, "content", None) or []
        for block in content:
            btype = _attr(block, "type")
            if btype == "text":
                texts.append(_attr(block, "text") or "")
            elif btype == "tool_use":
                args = _attr(block, "input")
                turn.tool_calls.append({
                    "id": _attr(block, "id") or "",
                    "name": _attr(block, "name") or "",
                    "arguments": args if isinstance(args, dict) else {},
                    "malformed": not isinstance(args, dict),
                })
        turn.text = " ".join(t for t in texts if t).strip()
        if turn.tool_calls:
            self._used_tools = True
            # Echo the assistant content back unchanged — thinking blocks
            # included. Editing or dropping them breaks the next turn.
            self.messages.append({"role": "assistant", "content": content})
        return turn

    def add_tool_results(self, results: list) -> None:
        # All tool_result blocks for one assistant turn go back in a SINGLE user
        # message. Splitting them across messages trains the model out of
        # parallel tool calls, which would change the arm's behavior.
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content}
                for call_id, content in results
            ],
        })


def make_harness_session(client, spec: VendorSpec, model: str, system_prompt: str):
    """Session for both the harness arm and the no-tool control arm.

    The control arm is the same session driven with tools_enabled=False on every
    step, so the two arms differ ONLY in whether the tool is offered — not in
    client, history handling, or sampling.
    """
    if spec.name == "anthropic":
        return AnthropicHarnessSession(client, spec, model, system_prompt)
    return OpenAIHarnessSession(client, spec, model, system_prompt)


def surfaced_domains(trajectory) -> set[str]:
    return {
        _domain_of(result.get("url", ""))
        for step in trajectory or []
        if isinstance(step, dict) and step.get("type") == "search"
        for result in step.get("results", []) or []
        if _domain_of(result.get("url", ""))
    }
