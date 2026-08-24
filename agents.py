"""Agent instrumentation across model vendors and search modalities.

This module exists so one frozen benchmark protocol can run under configurations
that differ in two ways at once — the model's class and where the search happens:

    model_class  x  search_mode
    -----------------------------------------------------------------
    oss          x  harness   (Baseten model, our search_web tool -> You.com)
    oss          x  none      (Baseten model, parametric memory only)
    frontier     x  harness   (OpenAI/Anthropic model, our tool -> You.com)
    frontier     x  native    (the model provider's own server-side search)
    frontier     x  none      (parametric baseline for the frontier arms)

Baseten runs no search of its own, so `oss x native` does not exist. Its harness
arms work because the harness owns the search: our code calls You.com and the
model only has to emit a tool call.

`native` is only attributable *within* a vendor: if GPT gets native search and
Claude gets the harness tool, model identity is confounded with search mode. So
each frontier vendor runs all three modes and `native` is compared to that same
vendor's `harness` and `none` arms.

Vendors vs model rows
---------------------
VENDORS says HOW to reach an endpoint (auth, base URL, which parameters that API
accepts). MATRIX_MODELS says WHAT the study runs. They are separate because the
OSS side needs several models from one vendor — a single open model cannot
distinguish "retrieval substitutes for capability" from "this one model happens
to be good at tool calling."

The hard instrumentation problem this module solves
---------------------------------------------------
Native (server-side) search does not produce `search_web` tool calls, so the
harness sees no trajectory. Most deterministic scorers read
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
# _20260209/_20260318 versions, which add "dynamic filtering": the search runs
# inside code execution and the model writes code that drops results before they
# reach the context window. The candidate set stays visible either way — at
# response_inclusion's default "full" the raw result blocks still come back — but
# WHICH SUBSET survived into the model's context does not, because that lives in
# model-authored code-execution output. Retrieval precision, snippet-derived
# scores, and leakage over the consumed surface all read exactly that quantity.
ANTHROPIC_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
# Pinned, not inherited: ["direct"] is already the default here, but flips to
# ["code_execution_20260120"] on _20260209 and later, so without this a one-word
# version bump would enable dynamic filtering with nothing in the diff to show
# it. Stating it makes a future bump a no-op.
ANTHROPIC_WEB_SEARCH_ALLOWED_CALLERS = ["direct"]
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
# OpenAI request tuning.
#
# The harness arm runs on /v1/responses, not /v1/chat/completions. This is not a
# style preference — chat completions REJECTS the combination this study needs:
#
#   400 Function tools with reasoning_effort are not supported for gpt-5.6-terra
#       in /v1/chat/completions. To use function tools, use /v1/responses or set
#       reasoning_effort to 'none'.
#
# and the model's DEFAULT effort is not 'none', so the rejection stands even when
# the request omits reasoning_effort entirely. An earlier revision left effort
# unpinned expecting that to sidestep the incompatibility; it does not.
#
# Being forced onto Responses is a net gain: it is also where the native arm
# already lives, so all three OpenAI arms (none / harness / native) now share one
# endpoint and one declared effort, which is what makes native-vs-harness within
# this vendor a one-variable contrast.
OPENAI_EFFORT = "high"
# Reasoning tokens count against this ceiling, so it needs more headroom than a
# non-reasoning answer would. Truncation is recorded, never silently scored.
OPENAI_MAX_OUTPUT_TOKENS = 16384

# How to hold a tool-calling conversation with a vendor. Declared rather than
# inferred from the vendor name because two vendors here speak the same protocol
# for opposite reasons: Baseten is chat-completions-only (no Responses API
# exists), while OpenAI is Responses-only (chat completions rejects tools plus
# reasoning). Collapsing that into `if vendor == "anthropic"` is what hid the
# incompatibility above until it failed live.
PROTOCOL_CHAT_COMPLETIONS = "chat_completions"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_MESSAGES = "messages"


# ---------------------------------------------------------------------------
# Vendor registry
#
# Every field here is an experiment condition, not a convenience. In particular
# `sampling` differs by vendor and CANNOT be equalized. Neither frontier vendor
# accepts sampling parameters at all: Claude rejects temperature/top_p/top_k with
# a 400, and gpt-5-family models reject `temperature` ("only the default (1) is
# supported"). Only the OSS arm can pin them.
#
# There is no `seed` field here because no vendor in this study supports one:
# gpt-5-family reasoning models dropped it, Anthropic never had it, and Baseten
# does not document it. So "temperature 0, seed 42" is unavailable, not declined.
# The run records what each arm actually received (`sampling_pinned`) rather than
# implying parity. Add the field back only alongside a vendor that honors it.
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
    # Reasoning depth, where the vendor lets us pin it. None = vendor default,
    # left unpinned. Recorded either way so a run is never ambiguous about it.
    reasoning_effort: str | None = None
    # Which wire protocol the harness arm speaks to this vendor. See the
    # PROTOCOL_* constants: this is a hard endpoint capability per vendor, not a
    # preference.
    harness_protocol: str = PROTOCOL_CHAT_COMPLETIONS
    notes: str = ""


VENDORS: dict[str, VendorSpec] = {
    # Baseten Model APIs are OpenAI-Chat-Completions-compatible at
    # https://inference.baseten.co/v1 (docs.baseten.co/inference/model-apis/overview,
    # checked 2026-08-05). All catalog models support tool calling, which is what
    # the OSS harness arm needs — Baseten runs no search of its own, so the
    # harness calls You.com directly and the model only has to emit the tool call.
    # A model that does not do that reliably collapses the OSS harness cell into
    # the OSS none cell, which is what zero_search_row detects.
    #
    # Default is the current-generation mid-price model; see MATRIX_MODELS for the
    # OSS rows the study actually runs.
    "baseten": VendorSpec(
        name="baseten",
        model_class="oss",
        api_key_env="BASETEN_API_KEY",
        default_model="zai-org/GLM-5.2",
        base_url="https://inference.baseten.co/v1",
        supports_native_search=False,
        sampling={"temperature": 0},
        # Baseten Model APIs expose Chat Completions only — there is no Responses
        # endpoint to move to, which is why both protocols have to stay supported.
        harness_protocol=PROTOCOL_CHAT_COMPLETIONS,
        notes="Server-side search is structurally unavailable; native arm N/A.",
    ),
    # gpt-5.6-terra is pinned rather than the moving `gpt-5.6` alias.
    #
    # Terra and Opus 5 are NOT asserted to be capability-equivalent. Terra is
    # OpenAI's cost-balanced tier while Opus is Anthropic's higher-capability
    # tier. The pairing is an operational choice, not a cross-vendor control.
    #
    # This is tolerable because no primary contrast depends on it: native-vs-
    # harness and search-vs-none both hold the model fixed within a vendor, and
    # cross-vendor native comparisons are already ruled out as two-variable. The
    # pairing only bounds how far an oss-vs-frontier result generalizes — read
    # that contrast as "vs this frontier model", never "vs frontier models".
    # NOTE the empty sampling dict: gpt-5-family reasoning models reject
    # `temperature` with a 400 ("only the default (1) is supported"), and do not
    # support `seed`. "temperature 0, seed 42" is therefore unavailable on this
    # vendor too — it is not a knob we chose not to use.
    #
    # reasoning_effort IS pinned here, which it could not be while the harness arm
    # ran on chat completions: that endpoint rejects reasoning_effort alongside
    # function tools, so pinning it on the native arm alone would have made effort
    # differ between this vendor's own native and harness arms — the one contrast
    # the matrix exists to measure. Both arms now run on /v1/responses, where
    # effort and function tools coexist, so all three OpenAI arms declare the same
    # depth and match Anthropic's ANTHROPIC_EFFORT.
    "openai": VendorSpec(
        name="openai",
        model_class="frontier",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5.6-terra",
        supports_native_search=True,
        sampling={},
        reasoning_effort=OPENAI_EFFORT,
        harness_protocol=PROTOCOL_RESPONSES,
        notes="Native search via the Responses API hosted web_search tool.",
    ),
    # Opus 5 is the cost-conscious Anthropic choice for this matrix. Comparisons
    # against Terra remain descriptive; the causal contrasts hold the model fixed
    # and compare no search, harness search, and native search within a vendor.
    "anthropic": VendorSpec(
        name="anthropic",
        model_class="frontier",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-opus-5",
        supports_native_search=True,
        # Opus 5 removed temperature/top_p/top_k — sending any of them is a 400.
        sampling={},
        # Pinnable here because effort and tools coexist fine on the Messages
        # API, so both this vendor's arms get the same declared depth.
        reasoning_effort=ANTHROPIC_EFFORT,
        harness_protocol=PROTOCOL_MESSAGES,
        notes="Native search via the server-side web_search tool.",
    ),
}


# Whether the five-search budget is enforced by the provider API. Anthropic's
# max_uses is per request, so pause_turn continuations receive only the remaining
# budget. OpenAI's Responses API applies max_tool_calls across its request.
NATIVE_BUDGET_ENFORCED = {"anthropic": True, "openai": True}

# What each search layer's date field actually MEANS. temporal_grounding treats
# `published_date` as a publication timestamp. You.com now returns BOTH web and
# news results: web page_age is last-modified, but news page_age is a
# publication timestamp. The merged result list carries mixed semantics, so the
# construct is stronger than before for news-intent queries but not uniform.
# Anthropic native's page_age is last-modified, and OpenAI native has no date
# field. A re-rendered web page still looks fresh without carrying new
# information.
#
# Freshness conclusions should rest on Corvus-QA's recency_rung — dataset ground
# truth about when the fact actually changed — rather than on vendor metadata.
# Rows still record the semantic so this stays visible.
DATE_FIELD_SEMANTICS = {
    "youdotcom": "mixed",                # web: last_modified, news: publication
    "anthropic_native": "last_modified",  # page_age
    "openai_native": None,                # no date field at all
}


# ---------------------------------------------------------------------------
# Model rows the study runs.
#
# A vendor says HOW to talk to an endpoint; a model row says WHAT to run. They are
# separate because the OSS side needs several models from one vendor: a
# single-model OSS arm cannot distinguish "retrieval substitutes for capability"
# from "this one open model happens to be good at tool calling."
#
# Both OSS rows are current-generation and chosen to span the cost range, so the
# substitution question gets two points on the frontier instead of one. Their
# order here is cheapest-first, which is also the order to smoke-test them in:
# the cheap row is where a tool-calling failure costs least to discover.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRow:
    vendor: str
    model: str
    note: str

    @property
    def spec(self) -> "VendorSpec":
        return VENDORS[self.vendor]


MATRIX_MODELS: tuple[ModelRow, ...] = (
    ModelRow("baseten", "deepseek-ai/DeepSeek-V4-Flash-0731",
             "OSS, cheapest row ($0.13/$0.26) — the extreme end of the "
             "cost-substitution frontier"),
    ModelRow("baseten", "zai-org/GLM-5.2",
             "OSS, mid-price row ($1.40/$4.40) — current-generation, stronger "
             "reasoning than the Flash row"),
    ModelRow("openai", "gpt-5.6-terra",
             "frontier, also runs a native-search arm"),
    ModelRow("anthropic", "claude-opus-5",
             "frontier, also runs a native-search arm"),
)


def oss_models() -> tuple[ModelRow, ...]:
    return tuple(r for r in MATRIX_MODELS if r.spec.model_class == "oss")


def frontier_models() -> tuple[ModelRow, ...]:
    return tuple(r for r in MATRIX_MODELS if r.spec.model_class == "frontier")


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

# ---------------------------------------------------------------------------
# Model token pricing, USD per million tokens: (input, output).
#
# This exists because search fees are the SMALL half of the bill. Work the
# arithmetic: five native searches cost 5 x $0.010 = $0.05, while one
# search-heavy turn that pulls 60k input tokens through a $10/MTok model costs
# $0.60 — roughly 12x the search spend. A search-fee-only comparison therefore
# does not measure a slightly incomplete quantity, it measures the wrong one.
#
# The native arms are worst affected, because their search-result tokens are
# billed as input tokens on the model call rather than as a search fee. Without
# this table the arm with the highest hidden token cost is the one that reports
# the lowest cost.
#
# Prices are list prices, checked 2026-08-05. Deliberately NOT using Sonnet 5's
# promotional $2/$10 intro rate (expires 2026-08-31), so a run's recorded cost
# does not silently change meaning when the promotion lapses.
#
# A model absent from this table yields None, and the row records
# model_cost_confirmed=False rather than a fabricated $0.00 — the same discipline
# applied to the native search rate.
# ---------------------------------------------------------------------------

MODEL_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI (developers.openai.com/api/docs/pricing)
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    # Anthropic (platform.claude.com/docs/en/pricing)
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Baseten Model APIs (baseten.co/pricing). Pricing the OSS arm is what makes
    # the capability-substitution question answerable rather than rhetorical:
    # gpt-oss-120b is 100x cheaper per input token than claude-fable-5, so
    # "OSS + retrieval vs frontier without it" is a cost-ratio claim, and a
    # cost-ratio claim needs both sides priced.
    "openai/gpt-oss-120b": (0.10, 0.50),
    "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.13, 0.26),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "moonshotai/Kimi-K2.6": (0.95, 4.00),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "zai-org/GLM-4.7": (0.60, 2.20),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": (0.60, 2.40),
}


# Cached input is billed at a fraction of the base input rate, and both frontier
# vendors publish the same 0.1x multiplier (OpenAI: $0.20 cached on $2.00 base
# for Terra; Anthropic: $0.50 cached on $5.00 base for Opus). It is
# expressed as a multiplier rather than eight more hand-entered numbers so the
# cached rate cannot drift out of sync with the base rate it is derived from.
#
# Baseten publishes no cached-input rate, so its rows price cached tokens at the
# full input rate. That is the conservative direction — it can only overstate the
# OSS arm's cost, never understate it — and Baseten reports cached_tokens=0 today
# anyway, so the multiplier is currently unexercised on that vendor.
FRONTIER_CACHED_INPUT_MULTIPLIER = 0.1
OSS_CACHED_INPUT_MULTIPLIER = 1.0


def cached_input_multiplier(model: str) -> float:
    """Baseten catalog paths carry an org prefix (`zai-org/GLM-5.2`); the two
    frontier vendors' model names never do. That split is what separates the
    priced-cache rows from the unpriced ones, and test_price_table_name_shapes
    pins it so a future entry cannot quietly land on the wrong side."""
    return (OSS_CACHED_INPUT_MULTIPLIER if "/" in model
            else FRONTIER_CACHED_INPUT_MULTIPLIER)


def model_cost_usd(model: str, prompt_tokens: int, completion_tokens: int,
                   cached_prompt_tokens: int = 0) -> tuple[float | None, bool]:
    """Return (usd, confirmed). None when the model has no pinned price.

    Returning None rather than 0.0 keeps an unpriced arm out of a cost frontier
    instead of placing it at the origin, which would read as "free".

    `cached_prompt_tokens` is the SUBSET of prompt_tokens served from a prompt
    cache. Callers must normalize to that convention before calling: OpenAI and
    the Chat Completions vendors already count cached tokens inside their input
    total, whereas Anthropic reports cache reads OUTSIDE `input_tokens`, so its
    adapter adds them in. Getting that backwards is a silent 10x error in either
    direction, which is why the normalization lives at each adapter and this
    function takes one unambiguous convention.

    Why this matters here rather than being a rounding detail: OpenAI caches
    automatically above ~1k prompt tokens, with no opt-in. The harness arm
    re-sends a growing conversation on every tool turn, so turns 2..6 are almost
    entirely cache hits, while the native arm makes a single uncached call.
    Charging every input token at the base rate therefore inflated the harness
    arm far more than the native arm — biasing the one cost comparison this
    matrix is built to make. Anthropic caching stays off (the requests send no
    cache_control), so its rows are unaffected and its cached count is 0.
    """
    rate = MODEL_USD_PER_MTOK.get(model)
    if rate is None:
        return None, False
    cached = max(0, min(cached_prompt_tokens, prompt_tokens))
    uncached = prompt_tokens - cached
    cached_rate = rate[0] * cached_input_multiplier(model)
    usd = (uncached * rate[0] + cached * cached_rate
           + completion_tokens * rate[1]) / 1_000_000
    return usd, True

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
# Serving path: direct-to-vendor, or through the Braintrust gateway
#
# The gateway (gateway.braintrust.dev) is an OpenAI/Anthropic-compatible proxy
# that holds the vendor keys org-side and logs every call to a Braintrust
# project. Setting BRAINTRUST_GATEWAY_URL routes EVERY vendor through it; there
# is deliberately no per-vendor switch, because a study whose arms sit behind
# different serving paths has a second variable moving inside every contrast the
# matrix exists to measure. Move all arms or none, and re-run a study after
# switching rather than joining across the boundary — `serving_path` is recorded
# in run metadata so a mixed study is at least detectable after the fact.
#
# What the gateway changes, and what it does not:
#   * Auth. One Braintrust key replaces the per-vendor keys, and the vendor keys
#     live in the org's AI Providers settings. A model with no provider
#     configured for the org 404s at the gateway rather than reaching a vendor.
#   * Model names. The gateway resolves a model name to a provider, so a name
#     the org's registry does not carry fails even if the vendor serves it. This
#     is the likeliest failure for the OSS rows, whose names are Baseten paths.
#   * Server-side search tools are NOT verified to survive the proxy. Both native
#     arms depend on a hosted tool (OpenAI's Responses `web_search`, Anthropic's
#     `web_search_20250305`) whose results come back as provider-specific block
#     types that the normalizers below parse. If the gateway drops or reshapes
#     those blocks, the native arms degrade to an empty trajectory — which
#     decision-surface gating reports as unobservable rather than as a passing
#     score, but which still costs a run. Verify before trusting a native arm on
#     the gateway: `python run_eval.py gateway-check`.
# ---------------------------------------------------------------------------


GATEWAY_URL_ENV = "BRAINTRUST_GATEWAY_URL"
GATEWAY_KEY_ENV = "BRAINTRUST_GATEWAY_API_KEY"
GATEWAY_PROJECT_ENV = "BRAINTRUST_GATEWAY_PROJECT"
GATEWAY_ORG_ENV = "BRAINTRUST_GATEWAY_ORG"

SERVING_PATH_DIRECT = "direct"
SERVING_PATH_GATEWAY = "gateway"


@dataclass(frozen=True)
class GatewayConfig:
    """Resolved gateway routing, or None-valued when routing direct."""

    root: str                       # no trailing /v1 — SDKs differ on the suffix
    api_key: str
    project: str | None = None
    org: str | None = None

    @property
    def openai_base_url(self) -> str:
        """openai-python appends /chat/completions or /responses to base_url."""
        return f"{self.root}/v1"

    @property
    def anthropic_base_url(self) -> str:
        """anthropic-python appends /v1/messages to base_url."""
        return self.root

    def headers(self) -> dict:
        h = {}
        if self.project:
            h["x-bt-project-name"] = self.project
        if self.org:
            h["x-bt-org-name"] = self.org
        return h


def gateway_config() -> GatewayConfig | None:
    """Read gateway routing from the environment. None means direct-to-vendor.

    The key falls back to BRAINTRUST_API_KEY because the same token authenticates
    the gateway and the logging SDK; BRAINTRUST_GATEWAY_API_KEY exists for the
    case where the vendor credentials live in a different org from the one the
    experiments are written to, so the two calls need two tokens.
    """
    root = (os.environ.get(GATEWAY_URL_ENV) or "").strip().rstrip("/")
    if not root:
        return None
    # Accept either form of the URL — the doc writes the base as .../v1 in some
    # places and bare in others, and the two SDKs want different suffixes.
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    key = (os.environ.get(GATEWAY_KEY_ENV)
           or os.environ.get("BRAINTRUST_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            f"{GATEWAY_URL_ENV} is set, so {GATEWAY_KEY_ENV} or "
            "BRAINTRUST_API_KEY is required to authenticate to the gateway")
    return GatewayConfig(
        root=root,
        api_key=key,
        project=(os.environ.get(GATEWAY_PROJECT_ENV) or "").strip() or None,
        org=(os.environ.get(GATEWAY_ORG_ENV) or "").strip() or None,
    )


def serving_path() -> str:
    return SERVING_PATH_GATEWAY if gateway_config() else SERVING_PATH_DIRECT


def effective_base_url(spec: VendorSpec) -> str | None:
    """The base URL this vendor's arm will actually be served from.

    Recorded in run metadata: `agent_base_url` has to name the endpoint that
    served the run, not the direct-to-vendor default it would have used.
    """
    gw = gateway_config()
    if gw is None:
        return spec.base_url
    return (gw.anthropic_base_url if spec.name == "anthropic"
            else gw.openai_base_url)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def get_client(vendor: str, wrap):
    """Build (and memoize) one traced client per vendor.

    `wrap` is braintrust's wrap_openai / wrap_anthropic, injected so this module
    has no braintrust import of its own and stays unit-testable offline.

    Under gateway routing the per-vendor key is not read at all — the vendor
    credential lives in the Braintrust org, so requiring a local one here would
    reject a correctly configured gateway run.
    """
    if vendor in _clients:
        return _clients[vendor]
    spec = vendor_of(vendor)
    gw = gateway_config()
    if gw is None:
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise SystemExit(
                f"{spec.api_key_env} is required for --model-vendor {vendor}")
        headers = None
    else:
        key = gw.api_key
        headers = gw.headers() or None
    if vendor == "anthropic":
        import anthropic  # imported lazily so the OSS/OpenAI paths need no install

        client = wrap(anthropic.Anthropic(
            api_key=key, base_url=effective_base_url(spec),
            default_headers=headers))
    else:
        client = wrap(OpenAI(
            api_key=key, base_url=effective_base_url(spec),
            default_headers=headers))
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
    # Subset of prompt_tokens served from a prompt cache, normalized to the
    # convention model_cost_usd documents (cached is INSIDE prompt_tokens).
    cached_prompt_tokens: int = 0
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
# Cached-input accounting
#
# The two wire families disagree about whether cached tokens are inside the
# input total, and the disagreement is invisible until it shows up as a 10x cost
# error. Normalize here, once, to the convention model_cost_usd documents:
# `cached` is always a SUBSET of the returned prompt-token count.
#
#   OpenAI (Responses) and Chat Completions: `input_tokens` / `prompt_tokens`
#       ALREADY include cached tokens; the details block names the cached subset.
#       Verified live: a repeated 2,811-token prefix returned input_tokens=2814
#       with cached_tokens=2811.
#   Anthropic (Messages): `input_tokens` EXCLUDES both cache reads and cache
#       writes, which are reported as sibling fields. So the reads must be added
#       back into the total before being named as its cached subset, or the
#       tokens vanish from the bill entirely.
#
# Cache WRITES are folded into full-price input rather than given Anthropic's
# 1.25x write multiplier. These requests send no cache_control, so writes are 0
# and the simplification is exact today; if caching is ever enabled here, this is
# the line to revisit.
# ---------------------------------------------------------------------------


def _openai_cached_tokens(usage) -> int:
    """Cached subset of an OpenAI/Chat-Completions input total."""
    for field_name in ("input_tokens_details", "prompt_tokens_details"):
        details = _attr(usage, field_name)
        if details is not None:
            return int(_attr(details, "cached_tokens") or 0)
    return 0


def _anthropic_token_split(usage) -> tuple[int, int]:
    """Return (billable_input_tokens, cached_subset) for a Messages response."""
    base = int(_attr(usage, "input_tokens") or 0)
    cache_read = int(_attr(usage, "cache_read_input_tokens") or 0)
    cache_write = int(_attr(usage, "cache_creation_input_tokens") or 0)
    return base + cache_read + cache_write, cache_read


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
    tool_base = {
        "type": ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "allowed_callers": ANTHROPIC_WEB_SEARCH_ALLOWED_CALLERS,
    }
    blocked = exclude_domains[:ANTHROPIC_MAX_BLOCKED_DOMAINS]
    if blocked:
        # Gold-source exclusion IS enforceable here, so the leakage rule applies
        # to this arm on the same terms as the harness arms.
        tool_base["blocked_domains"] = blocked

    run = NativeRun(
        final_answer="",
        surface=SURFACE_NO_SNIPPET,
        exclusion_enforced=bool(blocked),
    )
    messages = [{"role": "user", "content": question}]
    texts: list[str] = []
    searches_used = 0

    for _ in range(ANTHROPIC_MAX_PAUSE_TURNS + 1):
        remaining_searches = max_searches - searches_used
        if remaining_searches <= 0:
            break
        # max_uses is scoped to one Messages request. Supplying the remaining
        # cumulative budget prevents a pause_turn continuation from resetting
        # the arm to another five searches.
        tool = {**tool_base, "max_uses": remaining_searches}
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
        billable_input, cached_input = _anthropic_token_split(usage)
        run.prompt_tokens += billable_input
        run.cached_prompt_tokens += cached_input
        run.completion_tokens += int(_attr(usage, "output_tokens") or 0)
        server_use = _attr(usage, "server_tool_use")
        requests = _attr(server_use, "web_search_requests")
        if requests is not None:
            request_count = int(requests)
            run.vendor_search_count = (run.vendor_search_count or 0) + request_count
            searches_used += request_count

        trajectory_before = len(run.trajectory)
        _parse_anthropic_content(response, run, texts)
        if requests is None:
            searches_used += len(run.trajectory) - trajectory_before

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
# OpenAI exposes context depth, not a result-count or web/news split. `high` is
# the closest supported treatment to the harness arm's target surface of up to
# 10 results per search; observed sources remain the auditable quantity because
# the API does not guarantee a source count for this setting.
OPENAI_SEARCH_CONTEXT_SIZE = "high"


def openai_native_search(
    client,
    model: str,
    system_prompt: str,
    question: str,
    exclude_domains: list[str],
    effort: str | None = None,
    max_searches: int = 5,
) -> NativeRun:
    """Run one question through OpenAI's hosted Responses web_search tool.

    `effort` must be the same value the harness arm sends, or native-vs-harness
    within this vendor stops being a one-variable contrast. run_eval passes
    spec.reasoning_effort to both.
    """
    tool: dict = {
        "type": OPENAI_WEB_SEARCH_TOOL_TYPE,
        "search_context_size": OPENAI_SEARCH_CONTEXT_SIZE,
    }
    blocked = exclude_domains[:OPENAI_MAX_BLOCKED_DOMAINS]
    if blocked:
        tool["filters"] = {"blocked_domains": blocked}
    extra: dict = {}
    if effort:
        extra["reasoning"] = {"effort": effort}

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=question,
        tools=[tool],
        max_tool_calls=max_searches,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        **extra,
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
    run.cached_prompt_tokens = _openai_cached_tokens(usage)
    run.completion_tokens = int(_attr(usage, "output_tokens") or 0)
    # Now that this arm pins max_output_tokens, a long reasoning trace can hit the
    # ceiling and return a partial answer. Recorded so answer_truncated explains a
    # low score instead of it reading as a wrong answer.
    incomplete = _attr(response, "incomplete_details") or {}
    run.truncated = _attr(incomplete, "reason") == "max_output_tokens"

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
                if _attr(part, "type") == "refusal":
                    # Recorded, never retried and never scored as a wrong answer.
                    run.refused = True
                    continue
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
    cached_prompt_tokens: int = 0
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
        if tools_enabled:
            kwargs["tools"] = self.tools
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, **kwargs)
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        turn = Turn(
            text=(getattr(message, "content", None) or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            cached_prompt_tokens=_openai_cached_tokens(usage),
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
        billable_input, cached_input = _anthropic_token_split(usage)
        turn = Turn(
            prompt_tokens=billable_input,
            cached_prompt_tokens=cached_input,
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


class OpenAIResponsesHarnessSession(HarnessSession):
    """Responses API function calling — the only OpenAI path that takes both.

    Chat completions rejects function tools alongside reasoning effort on
    gpt-5-family models (see OPENAI_EFFORT), so this is not an alternative to
    OpenAIHarnessSession for this vendor, it is the only option. It also puts the
    harness arm on the same endpoint as the native arm.
    """

    def __init__(self, client, spec, model, system_prompt):
        super().__init__(client, spec, model, system_prompt)
        # Responses keeps the system prompt out of the turn list, in
        # `instructions`. The prompt text is byte-identical to the other
        # protocols' system message; only its position on the wire differs.
        self.input: list = []
        self._used_tools = False
        # Function tools are FLAT here — name/description/parameters at the top
        # level — where chat completions nests them under "function". The schema
        # itself is the shared SEARCH_TOOL_PARAMETERS, unchanged.
        #
        # strict stays False so that schema can be reused verbatim. Strict mode
        # would require additionalProperties:false and every property required,
        # i.e. a different tool declaration on this arm than on the Baseten and
        # Anthropic arms — a difference in the agent, not just the transport.
        self.tools = [{
            "type": "function",
            "name": SEARCH_TOOL_NAME,
            "description": SEARCH_TOOL_DESCRIPTION,
            "parameters": SEARCH_TOOL_PARAMETERS,
            "strict": False,
        }]

    def add_user(self, question: str) -> None:
        self.input.append({"role": "user", "content": question})

    def step(self, tools_enabled: bool) -> Turn:
        kwargs: dict = dict(self.spec.sampling)
        if tools_enabled:
            kwargs["tools"] = self.tools
        elif self._used_tools:
            # Same constraint as the Anthropic path: the input list already holds
            # function_call items, and a request carrying those without the tool
            # declared is rejected. Keep the declaration and forbid further calls.
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "none"
        if self.spec.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.spec.reasoning_effort}
        response = self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=self.input,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            **kwargs,
        )
        usage = _attr(response, "usage")
        turn = Turn(
            prompt_tokens=int(_attr(usage, "input_tokens") or 0),
            cached_prompt_tokens=_openai_cached_tokens(usage),
            completion_tokens=int(_attr(usage, "output_tokens") or 0),
            stop=_attr(response, "status"),
        )
        incomplete = _attr(response, "incomplete_details") or {}
        turn.truncated = _attr(incomplete, "reason") == "max_output_tokens"

        texts: list[str] = []
        items = _attr(response, "output") or []
        for item in items:
            itype = _attr(item, "type")
            if itype == "function_call":
                raw = _attr(item, "arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args, malformed = {}, True
                else:
                    malformed = not isinstance(args, dict)
                    if malformed:
                        args = {}
                turn.tool_calls.append({
                    # call_id, not id: call_id is the token function_call_output
                    # correlates on. `id` identifies the item, and echoing it
                    # back in place of call_id is accepted and then never matched.
                    "id": _attr(item, "call_id") or "",
                    "name": _attr(item, "name") or "",
                    "arguments": args,
                    "malformed": malformed,
                })
            elif itype == "message":
                for part in _attr(item, "content") or []:
                    if _attr(part, "type") == "refusal":
                        turn.refused = True
                        continue
                    texts.append(_attr(part, "text") or "")
        turn.text = " ".join(t for t in texts if t).strip()
        if turn.tool_calls:
            self._used_tools = True
            # Echo EVERY output item back, reasoning items included and
            # unmodified, exactly as the Anthropic path echoes thinking blocks.
            # Dropping them is accepted by the API but discards the reasoning
            # context between tool calls, which would make this arm's agent
            # weaker than the Anthropic harness arm for a reason unrelated to
            # search — a confound in the one contrast this vendor pair exists to
            # support.
            self.input.extend(items)
        return turn

    def add_tool_results(self, results: list) -> None:
        for call_id, content in results:
            self.input.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            })


# Protocol -> session class. Keyed on the declared protocol rather than the
# vendor name so adding a vendor cannot silently inherit the wrong endpoint.
HARNESS_SESSIONS = {
    PROTOCOL_CHAT_COMPLETIONS: OpenAIHarnessSession,
    PROTOCOL_RESPONSES: OpenAIResponsesHarnessSession,
    PROTOCOL_MESSAGES: AnthropicHarnessSession,
}


def make_harness_session(client, spec: VendorSpec, model: str, system_prompt: str):
    """Session for both the harness arm and the no-tool control arm.

    The control arm is the same session driven with tools_enabled=False on every
    step, so the two arms differ ONLY in whether the tool is offered — not in
    client, history handling, endpoint, or sampling.
    """
    try:
        session_class = HARNESS_SESSIONS[spec.harness_protocol]
    except KeyError:
        raise SystemExit(
            f"vendor {spec.name!r} declares harness_protocol "
            f"{spec.harness_protocol!r}, which has no session class. "
            f"Known protocols: {sorted(HARNESS_SESSIONS)}."
        ) from None
    return session_class(client, spec, model, system_prompt)


def surfaced_domains(trajectory) -> set[str]:
    return {
        _domain_of(result.get("url", ""))
        for step in trajectory or []
        if isinstance(step, dict) and step.get("type") == "search"
        for result in step.get("results", []) or []
        if _domain_of(result.get("url", ""))
    }
