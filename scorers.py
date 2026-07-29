"""Braintrust scorers shared by LiveNewsBench and RetrievalQA evaluations.

Expected record shapes
----------------------
input:
    {"question": str}

expected:
    LiveNewsBench: str
    RetrievalQA: list[str]

output (a plain answer string or the structured agent payload below):
    {
      "final_answer": str,
      "trajectory": [
        {"type": "search", "query": str, "tokens": int,
         "results": [{"rank": int, "url": str, "title": str,
                      "snippet": str, "published_date": "ISO8601|None"}]},
        {"type": "fetch", "url": str, "tokens": int},
        ...
      ],
      "used_searches": int,
      "used_clicks": int,
    }

metadata:
    The complete benchmark source fields plus importer provenance. Evaluation
    metadata such as provider, arm, and as_of may be merged in at run time.

All scorers return braintrust-style dicts: {"name", "score" (0-1 or None), "metadata"}.
Deterministic scorers are pure functions; the SimpleQA judge is the only LLM call.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import braintrust
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

ARCHIVE_DOMAINS = {
    "web.archive.org", "archive.org", "archive.is", "archive.ph",
    "archive.today", "cachedview.nl", "timetravel.mementoweb.org",
}


def _host(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _apex_match(host: str, domain: str) -> bool:
    """True if host == domain or host is a subdomain of it."""
    return host == domain or host.endswith("." + domain)


def _parse_date(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", str(s))
        if not m:
            return None
        dt = datetime.fromisoformat(m.group(1))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _output_payload(output) -> dict[str, Any]:
    return output if isinstance(output, dict) else {}


def _output_answer(output) -> str:
    if isinstance(output, str):
        return output
    if not isinstance(output, dict):
        return "" if output is None else str(output)
    for key in ("final_answer", "answer", "output"):
        value = output.get(key)
        if value is not None:
            return value if isinstance(value, str) else str(value)
    return ""


def _expected_answers(expected) -> list[str]:
    """Normalize both imported benchmark shapes into acceptable answer strings."""
    value = expected
    if isinstance(expected, dict):
        value = expected.get("answer", expected.get("ground_truth"))
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _metadata(metadata, kwargs) -> dict[str, Any]:
    value = metadata if isinstance(metadata, dict) else kwargs.get("metadata")
    return value if isinstance(value, dict) else {}


def _searches(output):
    payload = _output_payload(output)
    trajectory = payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        return []
    return [
        step
        for step in trajectory
        if isinstance(step, dict) and step.get("type") == "search"
    ]


def _all_results(output):
    for s in _searches(output):
        results = s.get("results", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if isinstance(result, dict):
                yield result


def _answer_aliases(expected):
    """Gold answer plus trivial variants for cheap snippet-containment checks.

    Deliberately conservative (string-level). The SimpleQA judge remains the
    authority on the *answer*; this is only used for surface/evidence metrics,
    where a false negative is acceptable and a false positive is not.
    """
    aliases = set()
    for gold in _expected_answers(expected):
        aliases.update({gold, gold.lower()})
        aliases.add(re.sub(r"[^\w\s]", "", gold.lower()))
        aliases.add(re.sub(r"\s+", " ", gold.lower()).strip())
        num = re.sub(r"(?<=\d),(?=\d{3}\b)", "", gold)
        aliases.add(num.lower())
    return {a for a in aliases if len(a) >= 3}


def _snippet_contains_gold(result, aliases) -> bool:
    text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    text_nopunct = re.sub(r"[^\w\s]", " ", text)
    return any(a in text or a in text_nopunct for a in aliases)


_ARTICLES = {"a", "an", "the"}


def _normalize_answer(value: str) -> str:
    """SQuAD-style normalization: NFKC, casefold, drop punctuation and leading
    articles, strip thousands separators, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


REFUSAL_SENTINEL = "i could not find this"


# ---------------------------------------------------------------------------
# 1. Answer layer — deterministic string match + SimpleQA-style judge
#    (judge keeps parity with evals/grade_answers.py)
# ---------------------------------------------------------------------------

def qa_answer_match(input, output, expected, **kwargs):
    """Deterministic answer correctness: normalized exact or containment match
    against any acceptable gold answer.

    The cheap, reproducible floor under the LLM judges — it needs no model, so
    it cannot drift between runs and it costs nothing to recompute when
    comparing providers. Short-form factoid answers (both LiveNewsBench and
    RetrievalQA, which supplies a *list* of acceptable answers) are the regime
    where string matching is trustworthy.

    Deliberately asymmetric: gold-in-prediction counts, prediction-in-gold does
    not, so "Jane Doe, on Tuesday" matches gold "Jane Doe" while a bare "Jane"
    does not match. Semantic paraphrase is out of scope by design — that is what
    the judge is for. Disagreement between this and the judge is the interesting
    signal, so keep both.
    """
    golds = _expected_answers(expected)
    prediction = _output_answer(output)
    pred_norm = _normalize_answer(prediction)

    if not golds:
        return {"name": "qa_answer_match", "score": None,
                "metadata": {"applicable": False,
                             "reason": "no gold answer on this row"}}

    attempted = bool(pred_norm) and REFUSAL_SENTINEL not in pred_norm
    if not attempted:
        return {"name": "qa_answer_match", "score": 0.0,
                "metadata": {"applicable": True, "attempted": False,
                             "match_type": "none", "n_acceptable": len(golds)}}

    match_type, matched = "none", None
    for gold in golds:
        gold_norm = _normalize_answer(gold)
        if not gold_norm:
            continue
        if gold_norm == pred_norm:
            match_type, matched = "exact", gold
            break
        # Token-boundary containment, so "5" does not match inside "1958".
        if re.search(rf"(?<!\w){re.escape(gold_norm)}(?!\w)", pred_norm):
            if match_type == "none":
                match_type, matched = "contains", gold

    return {
        "name": "qa_answer_match",
        "score": 1.0 if match_type != "none" else 0.0,
        "metadata": {
            "applicable": True,
            "attempted": True,
            "match_type": match_type,
            "matched_gold": matched,
            "n_acceptable": len(golds),
            "prediction_normalized": pred_norm[:200],
        },
    }

SIMPLEQA_GRADER_TEMPLATE = """\
Your job is to look at a question, a gold target, and a predicted answer, and \
then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

- CORRECT: the predicted answer fully contains the gold target's important \
information and does not contradict it. Formatting, hedging, and extra detail \
are fine.
- INCORRECT: the predicted answer contradicts the gold target in any way, \
even with hedging.
- NOT_ATTEMPTED: the important information of the gold target is neither \
confirmed nor contradicted (e.g. refusal, "I couldn't find this").

Question: {question}
Gold target: {target}
Predicted answer: {predicted}

Reply with exactly one word: CORRECT, INCORRECT, or NOT_ATTEMPTED."""


def make_simpleqa_grader(client, judge_model: str = "gpt-4.1"):
    """LLM judge. Pass an OpenAI-compatible client; keep the judge model in a
    different family from the frozen agent. LiveNewsBench's own grading uses
    the SimpleQA prompt with GPT-4.1, so this stays comparable to their
    published numbers while living in Braintrust's native scores.

    Emits score 1.0 / 0.0 / None(NOT_ATTEMPTED scored as 0 but flagged), plus
    the categorical grade in metadata so you can compute the SimpleQA-style
    "correct given attempted" slice later.
    """

    def simpleqa_grade(input, output, expected, **kwargs):
        targets = _expected_answers(expected)
        prompt = SIMPLEQA_GRADER_TEMPLATE.format(
            question=input.get("question", "") if isinstance(input, dict) else str(input),
            target=json.dumps(targets, ensure_ascii=False),
            predicted=_output_answer(output),
        )
        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        grade = resp.choices[0].message.content.strip().upper()
        grade = grade if grade in {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"} else "INCORRECT"
        return {
            "name": "simpleqa_grade",
            "score": 1.0 if grade == "CORRECT" else 0.0,
            "metadata": {"grade": grade, "attempted": grade != "NOT_ATTEMPTED"},
        }

    return simpleqa_grade


# ---------------------------------------------------------------------------
# 2. Validity — LiveNewsBench leakage rule as a hard scorer
# ---------------------------------------------------------------------------

def leakage_guard(input, output, expected, metadata=None, **kwargs):
    """1.0 iff no surfaced result comes from the source article's domain or a
    web-archive mirror. LiveNewsBench requires excluding these; a violation
    means the row's accuracy is contaminated, so treat 0 here as invalidating
    the row (filter on it in analysis, don't average it away).
    """
    metadata = _metadata(metadata, kwargs)
    release = metadata.get("livenewsbench_release")
    if not release:
        return {
            "name": "leakage_guard",
            "score": None,
            "metadata": {
                "applicable": False,
                "reason": "LiveNewsBench-only rule",
            },
        }

    source_urls = []
    if metadata.get("link"):
        source_urls.append(metadata["link"])
    for article in metadata.get("articles", []):
        if isinstance(article, dict):
            url = article.get("link") or article.get("url")
            if url:
                source_urls.append(url)
    source_domains = sorted({_host(url) for url in source_urls if _host(url)})
    bad = []
    for r in _all_results(output):
        h = _host(r.get("url", ""))
        if not h:
            continue
        source_match = any(_apex_match(h, domain) for domain in source_domains)
        archive_match = any(_apex_match(h, domain) for domain in ARCHIVE_DOMAINS)
        if source_match or archive_match:
            bad.append(r.get("url"))
    return {
        "name": "leakage_guard",
        "score": 1.0 if not bad else 0.0,
        "metadata": {
            "applicable": True,
            "leaked_urls": bad[:10],
            "source_domains": source_domains,
            "source_domain_available": bool(source_domains),
        },
    }


# ---------------------------------------------------------------------------
# 3. Economy — budget compliance and spend
# ---------------------------------------------------------------------------

def budget_economy(input, output, expected, **kwargs):
    """Compliance with the 5-search / 5-click default budget, with raw spend
    in metadata. Score = 1 if within budget, else 0. Searches, clicks, and
    tokens land in metadata for the cost/quality quadrant and for fitting the
    provider-specific patience/fetch-rate curves offline.
    """
    payload = _output_payload(output)
    trajectory = payload.get("trajectory", [])
    trajectory = trajectory if isinstance(trajectory, list) else []
    s, c = payload.get("used_searches", 0), payload.get("used_clicks", 0)
    s = s if isinstance(s, (int, float)) else 0
    c = c if isinstance(c, (int, float)) else 0
    tokens = sum(
        step.get("tokens", 0)
        for step in trajectory
        if isinstance(step, dict) and isinstance(step.get("tokens", 0), (int, float))
    )
    return {
        "name": "budget_economy",
        "score": 1.0 if (s <= 5 and c <= 5) else 0.0,
        "metadata": {"used_searches": s, "used_clicks": c,
                     "trajectory_tokens": tokens},
    }


# ---------------------------------------------------------------------------
# 4. Temporal grounding — exact, thanks to event_date
# ---------------------------------------------------------------------------

def temporal_grounding(input, output, expected, metadata=None, **kwargs):
    """Freshness of the decision surface, anchored to the item's event_date.

    - pre_event_rate: fraction of dated results published BEFORE the event.
      These structurally cannot contain the answer; a high rate at low ranks
      is a stale-index signature, independent of any judge.
    - evidence_lag_days: median (published_date - event_date) over post-event
      results — how quickly the provider's surfaced coverage follows events.
    - date_coverage: fraction of results carrying a parseable date at all.
      An agent can't reason about freshness it can't see, so date metadata is
      itself a decision-surface property and differs by provider.

    Headline score = fraction of *dated* results that are post-event
    (1 - pre_event_rate). None if no results carried dates.
    """
    metadata = _metadata(metadata, kwargs)
    event = _parse_date(metadata.get("event_date") or metadata.get("date"))
    if event is None:
        return {
            "name": "temporal_grounding",
            "score": None,
            "metadata": {
                "applicable": False,
                "reason": "No event date in benchmark metadata",
            },
        }
    dated, pre, lags = 0, 0, []
    total = 0
    pre_at_top3 = 0
    for r in _all_results(output):
        total += 1
        pd = _parse_date(r.get("published_date"))
        if pd is None or event is None:
            continue
        dated += 1
        if pd < event:
            pre += 1
            if r.get("rank", 99) <= 3:
                pre_at_top3 += 1
        else:
            lags.append((pd - event).total_seconds() / 86400.0)
    lags.sort()
    return {
        "name": "temporal_grounding",
        "score": (1.0 - pre / dated) if dated else None,
        "metadata": {
            "applicable": True,
            "event_date": event.date().isoformat(),
            "pre_event_rate": (pre / dated) if dated else None,
            "pre_event_at_top3": pre_at_top3,
            "evidence_lag_days_median": lags[len(lags) // 2] if lags else None,
            "date_coverage": (dated / total) if total else None,
            "n_results": total,
        },
    }


# ---------------------------------------------------------------------------
# 5. Decision surface — snippet self-sufficiency & gold rank
# ---------------------------------------------------------------------------

def snippet_sufficiency(input, output, expected, **kwargs):
    """Was the gold answer visible in the snippet layer, without any click?

    Score = 1 if any (title+snippet) contains a gold alias. Metadata carries
    the best rank and the search-round index where gold first surfaced, giving
    a reciprocal-rank view of how each provider concentrates gold on the
    surface (the Tavily-at-rank-1 vs Brave-rich-snippets distinction from the
    decision-surface paper, transplanted to your three providers).

    String-containment is conservative; treat this as a floor. If you add a
    per-URL LLM oracle pass later, write its labels into result dicts as
    `oracle_snippet_gold` and this scorer will prefer them.
    """
    aliases = _answer_aliases(expected)
    best_rank, first_round, hits = None, None, 0
    for round_idx, s in enumerate(_searches(output)):
        for r in s.get("results", []):
            is_gold = r.get("oracle_snippet_gold")
            if is_gold is None:
                is_gold = _snippet_contains_gold(r, aliases)
            if is_gold:
                hits += 1
                rank = r.get("rank", 99)
                if best_rank is None or rank < best_rank:
                    best_rank, first_round = rank, round_idx
    return {
        "name": "snippet_sufficiency",
        "score": 1.0 if hits else 0.0,
        "metadata": {
            "gold_snippet_hits": hits,
            "gold_best_rank": best_rank,
            "gold_rr": (1.0 / best_rank) if best_rank else 0.0,
            "gold_first_search_round": first_round,
        },
    }


def token_discounted_gain(input, output, expected, tau: float = 4000.0, **kwargs):
    """Time-biased gain (Smucker & Clarke), with tokens as the clock.

    Walk the trajectory in order, accumulating tokens the agent had to ingest.
    Gain = exp(-cum_tokens / tau) at the first point gold becomes visible in
    a snippet (fetched-page gold requires the per-URL oracle; without it,
    fetches contribute cost but not gain — stated limitation).

    tau = 4000 tokens-to-half-ish; sweep it in analysis, it's a lens not a truth.
    Distinguishes 'gold at rank 1 of search 1' from 'gold after two searches
    and 9k tokens of excerpts' — invisible to accuracy, central to cost.
    """
    aliases = _answer_aliases(expected)
    cum = 0.0
    trajectory = _output_payload(output).get("trajectory", [])
    trajectory = trajectory if isinstance(trajectory, list) else []
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "search":
            per_result = (step.get("tokens", 0) / max(len(step.get("results", [])), 1))
            for r in sorted(step.get("results", []), key=lambda x: x.get("rank", 99)):
                cum += per_result
                gold = r.get("oracle_snippet_gold")
                if gold is None:
                    gold = _snippet_contains_gold(r, aliases)
                if gold:
                    return {"name": "token_discounted_gain",
                            "score": math.exp(-cum / tau),
                            "metadata": {"tokens_to_gold": cum, "tau": tau}}
        else:
            cum += step.get("tokens", 0)
    return {"name": "token_discounted_gain", "score": 0.0,
            "metadata": {"tokens_to_gold": None, "tau": tau,
                         "total_tokens": cum}}


# ---------------------------------------------------------------------------
# 6. Diversity — redundancy and source concentration
# ---------------------------------------------------------------------------

def compression_redundancy(input, output, expected, **kwargs):
    """Normalized-compression-distance trick: gzip the concatenated snippets.

    Breaking news is where syndication explodes — ten results that are one
    wire story compress extremely well. Score = compressed/raw ratio rescaled
    against a floor (pure-duplicate text compresses to ~0.05 of raw at these
    lengths), so higher = more marginal information per result.
    """
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in _all_results(output)]
    blob = "\n".join(t for t in texts if t.strip()).encode("utf-8")
    if len(blob) < 200:
        return {"name": "compression_redundancy", "score": None,
                "metadata": {"reason": "too little snippet text"}}
    ratio = len(gzip.compress(blob, 9)) / len(blob)
    score = max(0.0, min(1.0, (ratio - 0.05) / (0.60 - 0.05)))  # empirical band
    return {"name": "compression_redundancy", "score": score,
            "metadata": {"gzip_ratio": ratio, "n_snippets": len(texts)}}


def domain_entropy(input, output, expected, **kwargs):
    """Shannon entropy over result hostnames, normalized by log(n_results).
    1.0 = every result from a distinct host; 0.0 = single-source SERP.
    Read alongside compression_redundancy: distinct hosts syndicating one
    story score high here and low there — that gap IS the syndication effect.
    """
    hosts = [_host(r.get("url", "")) for r in _all_results(output)]
    hosts = [h for h in hosts if h]
    if len(hosts) < 2:
        return {"name": "domain_entropy", "score": None,
                "metadata": {"n_results": len(hosts)}}
    counts = Counter(hosts)
    n = len(hosts)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    return {"name": "domain_entropy",
            "score": H / math.log(n),
            "metadata": {"unique_hosts": len(counts), "n_results": n,
                         "top_host": counts.most_common(1)[0]}}


DETERMINISTIC_SCORERS = [
    qa_answer_match,
    leakage_guard,
    budget_economy,
    temporal_grounding,
    snippet_sufficiency,
    token_discounted_gain,
    compression_redundancy,
    domain_entropy,
]


# ---------------------------------------------------------------------------
# Braintrust deployment definitions
# ---------------------------------------------------------------------------

class EvalScorerParams(BaseModel):
    """Standard arguments Braintrust supplies to an experiment scorer."""

    input: Any = None
    output: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


project = braintrust.projects.create(name="search_evals")

project.scorers.create(
    name="QA Answer Match",
    slug="qa-answer-match",
    description=(
        "Deterministic answer correctness: a normalized exact or containment "
        "match of an acceptable gold answer inside the predicted final answer. "
        "Normalization is SQuAD-style — Unicode NFKC, case folding, punctuation "
        "and leading-article removal, thousands-separator stripping, whitespace "
        "collapse. It accepts either one gold answer or RetrievalQA's list of "
        "acceptable answers, and extracts final_answer from a structured agent "
        "payload. Containment is token-bounded and one-directional, so extra "
        "non-contradictory detail still matches while a truncated answer does "
        "not. Refusals score 0 and are flagged as unattempted. This is the "
        "model-free floor under the LLM judges: it cannot drift between runs, "
        "so provider comparisons stay reproducible. It does not capture "
        "semantic paraphrase — disagreement with the QA Answer Correctness "
        "judge is the signal worth inspecting, so report both."
    ),
    parameters=EvalScorerParams,
    handler=qa_answer_match,
    metadata={"__pass_threshold": 0.5},
    tags=["LiveNewsBench", "RetrievalQA", "answer-quality", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="LiveNewsBench Leakage Guard",
    slug="livenewsbench-leakage-guard",
    description=(
        "Enforces the LiveNewsBench anti-leakage rule. The scorer inspects every "
        "surfaced search-result URL and fails when a result comes from the source "
        "article's domain or a web-archive mirror. Source URLs are read from the "
        "dataset metadata imported with each row. It returns no score for "
        "RetrievalQA because that benchmark does not define this exclusion rule. "
        "For newer LiveNewsBench rows whose upstream release omits source URLs, "
        "archive domains are still checked and source_domain_available is reported "
        "in result metadata."
    ),
    parameters=EvalScorerParams,
    handler=leakage_guard,
    metadata={"__pass_threshold": 1.0},
    tags=["LiveNewsBench", "validity", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Search Budget Compliance",
    slug="search-budget-compliance",
    description=(
        "Checks whether an agent stays within the benchmark's default allowance "
        "of at most five searches and five full-page clicks. The score is 1 only "
        "when both limits are respected. Result metadata records searches, clicks, "
        "and total trajectory tokens so experiments can compare quality against "
        "retrieval cost. Compatible with both LiveNewsBench and RetrievalQA when "
        "the task returns the documented structured trajectory payload."
    ),
    parameters=EvalScorerParams,
    handler=budget_economy,
    metadata={"__pass_threshold": 1.0},
    tags=["LiveNewsBench", "RetrievalQA", "cost", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Temporal Grounding",
    slug="temporal-grounding",
    description=(
        "Measures whether dated search results were published on or after the "
        "question's event date. The headline score is the fraction of parseably "
        "dated results that are post-event. Metadata includes pre-event rate, "
        "pre-event results in the top three, median post-event evidence lag, date "
        "coverage, and result count. Event dates are read from LiveNewsBench "
        "metadata; rows without an event date, including RetrievalQA, return no "
        "score instead of being treated as failures."
    ),
    parameters=EvalScorerParams,
    handler=temporal_grounding,
    metadata={"__pass_threshold": 0.8},
    tags=["LiveNewsBench", "freshness", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Snippet Answer Sufficiency",
    slug="snippet-answer-sufficiency",
    description=(
        "Checks whether any search-result title or snippet exposes an acceptable "
        "gold answer before a page click. It supports both a single expected "
        "answer string and RetrievalQA's list of acceptable answers, using "
        "conservative punctuation, case, whitespace, and thousands-separator "
        "aliases. Metadata reports hit count, best rank, reciprocal rank, and the "
        "first search round containing an answer. This is a surface-evidence "
        "metric, not a substitute for semantic answer correctness."
    ),
    parameters=EvalScorerParams,
    handler=snippet_sufficiency,
    metadata={"__pass_threshold": 0.5},
    tags=["LiveNewsBench", "RetrievalQA", "retrieval", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Token-Discounted Answer Gain",
    slug="token-discounted-answer-gain",
    description=(
        "Rewards agents that surface an acceptable answer early with little text "
        "consumption. The scorer walks the trajectory in order and returns "
        "exp(-tokens_to_answer/4000) when a gold alias first appears in a snippet; "
        "otherwise it returns 0. It supports string and multi-answer expectations. "
        "Metadata exposes tokens_to_gold and the decay constant, making the "
        "quality-versus-cost tradeoff auditable."
    ),
    parameters=EvalScorerParams,
    handler=token_discounted_gain,
    metadata={"__pass_threshold": 0.1},
    tags=["LiveNewsBench", "RetrievalQA", "cost", "retrieval", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Snippet Information Diversity",
    slug="snippet-information-diversity",
    description=(
        "Estimates marginal information diversity across retrieved snippets using "
        "gzip compression. Repetitive or syndicated result sets compress strongly "
        "and receive lower scores; varied result text receives higher scores. "
        "The raw gzip ratio and snippet count are included in metadata. Fewer than "
        "200 bytes of snippet text returns no score because compression ratios are "
        "unstable at very small sizes."
    ),
    parameters=EvalScorerParams,
    handler=compression_redundancy,
    metadata={"__pass_threshold": 0.3},
    tags=["LiveNewsBench", "RetrievalQA", "diversity", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="Result Domain Diversity",
    slug="result-domain-diversity",
    description=(
        "Computes Shannon entropy over normalized search-result hostnames and "
        "divides by log(number of results). A score near 1 means results are spread "
        "across distinct domains; a score near 0 indicates concentration in one "
        "source. Metadata reports unique hosts, total valid hosts, and the most "
        "common host. Fewer than two valid hostnames returns no score."
    ),
    parameters=EvalScorerParams,
    handler=domain_entropy,
    metadata={"__pass_threshold": 0.5},
    tags=["LiveNewsBench", "RetrievalQA", "diversity", "deterministic"],
    if_exists="replace",
)

project.scorers.create(
    name="QA Answer Correctness",
    slug="qa-answer-correctness",
    description=(
        "Semantic LLM judge for final-answer correctness across LiveNewsBench and "
        "RetrievalQA. It accepts either one gold answer or a list of acceptable "
        "answers, extracts final_answer when the prediction is a structured agent "
        "payload, and grades the response as CORRECT, INCORRECT, or NOT_ATTEMPTED. "
        "CORRECT requires the important gold information without contradiction; "
        "both INCORRECT and NOT_ATTEMPTED score 0. The judge uses Baseten's "
        "OpenAI-compatible GPT-OSS 120B route with chain-of-thought enabled. This "
        "is the configured organization's lowest-cost verified model route at "
        "approximately $0.10 per million input tokens and $0.50 per million output "
        "tokens."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                "Grade this short-form question-answering result.\n\n"
                "Question payload:\n{{input}}\n\n"
                "Acceptable gold answer or answers:\n{{expected}}\n\n"
                "Predicted payload:\n{{output}}\n\n"
                "If the predicted payload is an object, judge only its "
                "`final_answer`, `answer`, or `output` field, in that priority "
                "order. Ignore its retrieval trajectory except when identifying "
                "the final answer.\n\n"
                "Return CORRECT when the prediction contains the important "
                "information from at least one acceptable gold answer and does "
                "not contradict it. Formatting, concise paraphrases, and extra "
                "non-contradictory detail are allowed. Return INCORRECT for a "
                "wrong or contradictory answer. Return NOT_ATTEMPTED when the "
                "prediction refuses, abstains, or neither confirms nor "
                "contradicts an acceptable answer."
            ),
        }
    ],
    model="openai/gpt-oss-120b",
    use_cot=True,
    choice_scores={"CORRECT": 1.0, "INCORRECT": 0.0, "NOT_ATTEMPTED": 0.0},
    metadata={"__pass_threshold": 0.5},
    tags=["LiveNewsBench", "RetrievalQA", "answer-quality", "LLM-judge"],
    if_exists="replace",
)
