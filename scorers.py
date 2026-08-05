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
      "decision_surface": "full" | "no_snippet" | "urls_only" | "none",
      "used_searches": int,
      "used_clicks": int,
    }

`decision_surface` declares which trajectory fields are actually populated, and
it is load-bearing. Server-side ("native") search arms normalize onto the same
trajectory schema but cannot fill every field: Anthropic returns no snippets,
OpenAI returns neither snippets nor dates. Every trajectory-reading scorer gates
on the tier and returns None rather than a score it cannot support — see the
"Decision-surface observability" section below for why a passing score on an
unobservable surface is the dangerous failure mode here.

metadata:
    The complete benchmark source fields plus importer provenance. Evaluation
    metadata merged in at run time: model_class, model_vendor, search_mode,
    search_provider, freshness_treatment, exclusion_enforced, search_budget,
    zero_search_row, as_of.

All scorers return braintrust-style dicts: {"name", "score" (0-1 or None), "metadata"}.
A None score means "not measurable on this arm" and is excluded from averages;
0.0 means measured and failed. Never substitute one for the other.
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
    return [item_str for item in values if (item_str := str(item).strip())]


def _metadata(metadata, kwargs) -> dict[str, Any]:
    value = metadata if isinstance(metadata, dict) else kwargs.get("metadata")
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Decision-surface observability
#
# Not every arm exposes the same view of what the agent saw. Server-side
# ("native") search returns citations or bare URLs, not the rank/title/snippet/
# date rows the harness tool produces. Without gating, every scorer that reads
# `trajectory` returns a PASSING score on those arms for the wrong reason:
# leakage_guard sees no leaked URLs because it sees no URLs, budget_economy
# sees zero searches, snippet_sufficiency sees no gold and scores 0.0 as though
# the provider failed to surface it.
#
# A score of None means "not measurable on this arm" and is excluded from
# averages. A score of 1.0 or 0.0 means "measured". Never conflate the two: an
# unobservable surface must never produce a number that reads as evidence.
#
# Tiers (set by run_eval from agents.SURFACE_*):
#   full        rank/url/title/snippet/published_date  — harness arms
#   no_snippet  rank/url/title/published_date          — Anthropic native
#   urls_only   rank/url + titles from citations        — OpenAI native
#   none        no retrieval at all                    — no_search arms
# ---------------------------------------------------------------------------

SURFACE_FULL = "full"
SURFACE_NO_SNIPPET = "no_snippet"
SURFACE_URLS_ONLY = "urls_only"
SURFACE_NONE = "none"
_KNOWN_SURFACES = (SURFACE_FULL, SURFACE_NO_SNIPPET, SURFACE_URLS_ONLY,
                   SURFACE_NONE)

# Which tiers populate which fields.
_URL_SURFACES = {SURFACE_FULL, SURFACE_NO_SNIPPET, SURFACE_URLS_ONLY}
_DATE_SURFACES = {SURFACE_FULL, SURFACE_NO_SNIPPET}
_SNIPPET_SURFACES = {SURFACE_FULL}


def _surface(output) -> str:
    """The decision-surface tier for this row.

    Rows written before the tier existed carried full harness results whenever
    they had a trajectory, so infer that rather than dropping historical rows
    out of every trajectory-based metric.
    """
    payload = _output_payload(output)
    declared = payload.get("decision_surface")
    if declared in _KNOWN_SURFACES:
        return declared
    return SURFACE_FULL if payload.get("trajectory") else SURFACE_NONE


def _not_measurable(name: str, surface: str, needs: str) -> dict[str, Any]:
    return {
        "name": name,
        "score": None,
        "metadata": {
            "applicable": False,
            "decision_surface": surface,
            "reason": f"{surface} surface exposes no {needs}",
        },
    }


def iter_source_urls(metadata, expected=None):
    """Yield source URLs from metadata first, then fall back to a legacy expected payload."""
    if isinstance(metadata, dict):
        if metadata.get("link"):
            yield metadata["link"]
        for article in metadata.get("articles") or []:
            if isinstance(article, dict):
                url = article.get("link") or article.get("url")
                if url:
                    yield url
        return
    if isinstance(expected, dict):
        link = expected.get("link")
        if link:
            yield link


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
    normalized_golds = [_normalize_answer(gold) for gold in golds]
    expected_refusal = REFUSAL_SENTINEL in normalized_golds
    if expected_refusal and REFUSAL_SENTINEL in pred_norm:
        return {
            "name": "qa_answer_match",
            "score": 1.0,
            "metadata": {
                "applicable": True,
                "attempted": False,
                "match_type": "expected_refusal",
                "matched_gold": next(
                    gold
                    for gold in golds
                    if _normalize_answer(gold) == REFUSAL_SENTINEL
                ),
                "n_acceptable": len(golds),
            },
        }
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


def make_jury_grader(judges, template: str = None):
    """Cross-vendor judge jury: majority vote over the SimpleQA 3-way grade.

    `judges` is a list of (client, model_name) pairs. A single-vendor judge
    grading a single-vendor agent invites self-preference bias, and one judge
    gives no way to tell a hard row from a judge that simply disagrees with
    itself. Majority vote fixes the first; recording every ballot plus
    `unanimous` in metadata surfaces the second, so rows where the jury split
    can be pulled out and inspected rather than silently averaged.

    Emits ONE score per row. Jury reliability across rows (agreement rates,
    per-judge bias) is an analysis question, computed from this metadata
    downstream — not something a scorer should try to summarize.
    """
    template = template or SIMPLEQA_GRADER_TEMPLATE
    valid = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}

    def jury_grade(input, output, expected, **kwargs):
        prompt = template.format(
            question=input.get("question", "") if isinstance(input, dict) else str(input),
            target=json.dumps(_expected_answers(expected), ensure_ascii=False),
            predicted=_output_answer(output),
        )
        ballots, errors = {}, {}
        for client, model in judges:
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
                grade = (resp.choices[0].message.content or "").strip().upper()
                ballots[model] = grade if grade in valid else "INCORRECT"
            except Exception as exc:            # a dead judge must not void the row
                errors[model] = f"{type(exc).__name__}: {exc}"

        if not ballots:
            return {"name": "jury_grade", "score": None,
                    "metadata": {"applicable": False, "judge_errors": errors}}

        tally = Counter(ballots.values())
        verdict, votes = tally.most_common(1)[0]
        # A tie on an even/degraded panel is not a majority for CORRECT.
        if tally.get("CORRECT", 0) * 2 <= len(ballots) and verdict == "CORRECT":
            verdict = tally.most_common(2)[1][0] if len(tally) > 1 else "INCORRECT"
        return {
            "name": "jury_grade",
            "score": 1.0 if verdict == "CORRECT" else 0.0,
            "metadata": {
                "applicable": True,
                "verdict": verdict,
                "ballots": ballots,
                "n_judges": len(ballots),
                "votes_for_verdict": votes,
                "unanimous": len(tally) == 1,
                "attempted": verdict != "NOT_ATTEMPTED",
                "judge_errors": errors or None,
            },
        }

    return jury_grade


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

    surface = _surface(output)
    if surface not in _URL_SURFACES:
        # The no_search arm surfaces nothing, so there is nothing that could
        # leak. Scoring 1.0 here would hand the control arm a free pass on a
        # rule it was never subject to, inflating any gated headline number.
        return _not_measurable("leakage_guard", surface, "result URLs")

    payload = _output_payload(output)
    results = list(_all_results(output))
    used_searches = payload.get("used_searches", 0)
    used_searches = used_searches if isinstance(used_searches, (int, float)) else 0
    if not results and used_searches:
        # Searches ran but no URLs came back to inspect — a dropped or filtered
        # response, not a clean SERP. Unmeasurable, not compliant.
        return {
            "name": "leakage_guard",
            "score": None,
            "metadata": {
                "applicable": False,
                "decision_surface": surface,
                "reason": (f"{used_searches} search(es) ran but surfaced no "
                           "inspectable URLs"),
            },
        }

    source_domains = sorted(
        {_host(url) for url in iter_source_urls(metadata, None) if _host(url)}
    )
    bad = []
    for r in results:
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
            "decision_surface": surface,
            "leaked_urls": bad[:10],
            "source_domains": source_domains,
            "source_domain_available": bool(source_domains),
            # Whether the exclusion list was actually sent to the search layer.
            # Every current arm can enforce it (harness excludeDomains, Anthropic
            # blocked_domains, OpenAI filters.blocked_domains) — but a future arm
            # that cannot must not be silently pooled with those that can.
            "exclusion_enforced": metadata.get("exclusion_enforced"),
            "n_results_inspected": len(results),
        },
    }


# ---------------------------------------------------------------------------
# 3. Economy — budget compliance and spend
# ---------------------------------------------------------------------------

def budget_economy(input, output, expected, metadata=None, **kwargs):
    """Compliance with the search/click budget, with raw spend in metadata.
    Score = 1 if within budget, else 0. Searches, clicks, and tokens land in
    metadata for the cost/quality quadrant and for fitting the provider-specific
    patience/fetch-rate curves offline.

    Not applicable to the no-tool control arm: an arm with no tools is
    trivially within budget, and scoring it 1.0 would credit compliance with a
    rule it could not violate.
    """
    surface = _surface(output)
    if surface == SURFACE_NONE:
        return _not_measurable("budget_economy", surface, "tool calls to budget")

    payload = _output_payload(output)
    row_metadata = _metadata(metadata, kwargs)
    trajectory = payload.get("trajectory", [])
    trajectory = trajectory if isinstance(trajectory, list) else []
    s, c = payload.get("used_searches", 0), payload.get("used_clicks", 0)
    s = s if isinstance(s, (int, float)) else 0
    c = c if isinstance(c, (int, float)) else 0
    # Read the cap from the run rather than hardcoding 5, so changing
    # MAX_SEARCHES cannot silently decouple the protocol from the scorer.
    cap = row_metadata.get("search_budget")
    cap = cap if isinstance(cap, (int, float)) else 5
    tokens = sum(
        step.get("tokens", 0)
        for step in trajectory
        if isinstance(step, dict) and isinstance(step.get("tokens", 0), (int, float))
    )
    return {
        "name": "budget_economy",
        "score": 1.0 if (s <= cap and c <= 5) else 0.0,
        "metadata": {"used_searches": s, "used_clicks": c,
                     "search_budget": cap,
                     "decision_surface": surface,
                     # Native arms bill search tokens on the model spans, so a 0
                     # here is an accounting boundary, not a free search.
                     "trajectory_tokens_measured": surface in _SNIPPET_SURFACES,
                     "trajectory_tokens": tokens},
    }


def dealbreaker_gate(input, output, expected, metadata=None, **kwargs):
    """Hard-constraint gate: did this row violate a rule that invalidates it?

    Answer quality and rule compliance currently sit side by side as unrelated
    columns, so a provider can top the answer score while leaking gold sources
    or blowing the budget. This collapses the non-negotiable rules into one
    0/1 flag that analysis multiplies the answer score by, which is what makes
    a gated headline number possible.

    Composed from the existing scorers rather than reimplementing them, so the
    rules cannot drift apart. A rule that does not apply to a row (the leakage
    rule on RetrievalQA) cannot fail it.
    """
    gates = {"leakage_guard": leakage_guard(input, output, expected,
                                            metadata=metadata, **kwargs),
             "budget_economy": budget_economy(input, output, expected,
                                              metadata=metadata, **kwargs)}
    violated = [name for name, res in gates.items() if res.get("score") == 0.0]
    skipped = [name for name, res in gates.items() if res.get("score") is None]
    checked = [n for n in gates if n not in skipped]
    if not checked:
        # Every constituent rule was inapplicable, so there is no gate to pass.
        # Returning 1.0 here is the specific bug that let a native or no-search
        # arm show a clean gated headline number without a single rule evaluated.
        return {
            "name": "dealbreaker_gate",
            "score": None,
            "metadata": {
                "applicable": False,
                "not_applicable": skipped,
                "gates_checked": [],
                "decision_surface": _surface(output),
                "reason": "no hard-constraint rule was measurable on this row",
            },
        }
    return {
        "name": "dealbreaker_gate",
        "score": 0.0 if violated else 1.0,
        "metadata": {
            "applicable": True,
            "violated": violated,
            "not_applicable": skipped,
            "gates_checked": checked,
        },
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
    surface = _surface(output)
    if surface not in _DATE_SURFACES:
        # OpenAI's native search returns URLs with no publication dates, so
        # freshness of its decision surface is genuinely unmeasurable — not 1.0,
        # and not 0.0 either. This is the arm asymmetry to disclose in any
        # writeup that compares native search across vendors.
        return _not_measurable(
            "temporal_grounding", surface, "per-result publication dates")
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
    surface = _surface(output)
    if surface not in _SNIPPET_SURFACES:
        # Without a snippet layer this scorer would report 0.0 — reading as "the
        # provider never surfaced gold" when in fact we cannot see what was
        # surfaced. Anthropic's citations carry cited_text, but that text is
        # selected because it supports the answer, so scoring it would guarantee
        # a near-perfect result. Both failure directions are avoided by None.
        return _not_measurable(
            "snippet_sufficiency", surface, "result snippets")
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
            "applicable": True,
            "decision_surface": surface,
            "gold_snippet_hits": hits,
            "gold_best_rank": best_rank,
            "gold_rr": (1.0 / best_rank) if best_rank else 0.0,
            "gold_first_search_round": first_round,
        },
    }


def evidence_precision(input, output, expected, **kwargs):
    """What FRACTION of the surfaced results actually carried the gold answer?

    snippet_sufficiency is a recall question — did gold appear anywhere. This is
    its precision counterpart: signal density on the decision surface. A
    provider returning gold at rank 1 plus seven irrelevant results and one
    returning gold in five of eight score identically on sufficiency, but the
    second hands the agent a far cheaper read. Pairs with
    compression_redundancy (are the results distinct?) to separate "diverse but
    useless" from "dense and on-topic".

    Per-search precision is in metadata because a provider that nails the first
    query and then degrades looks different from one that is uniformly mediocre.

    Expect low absolute values — most news results legitimately will not restate
    the answer. Read it comparatively between providers, never as an absolute
    quality bar. Uses the same conservative string containment as
    snippet_sufficiency, and prefers `oracle_snippet_gold` labels when present.
    """
    surface = _surface(output)
    if surface not in _SNIPPET_SURFACES:
        return _not_measurable("evidence_precision", surface, "result snippets")
    aliases = _answer_aliases(expected)
    per_search, total, hits = [], 0, 0
    for s in _searches(output):
        results = [r for r in s.get("results", []) if isinstance(r, dict)]
        if not results:
            continue
        n_gold = 0
        for r in results:
            is_gold = r.get("oracle_snippet_gold")
            if is_gold is None:
                is_gold = _snippet_contains_gold(r, aliases)
            n_gold += bool(is_gold)
        per_search.append(round(n_gold / len(results), 4))
        total += len(results)
        hits += n_gold

    if not total:
        return {"name": "evidence_precision", "score": None,
                "metadata": {"applicable": False,
                             "reason": "no results surfaced"}}
    return {
        "name": "evidence_precision",
        "score": hits / total,
        "metadata": {
            "applicable": True,
            "gold_results": hits,
            "n_results": total,
            "precision_per_search": per_search,
            "best_search_precision": max(per_search) if per_search else None,
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
    surface = _surface(output)
    if surface not in _SNIPPET_SURFACES:
        # Needs both a snippet layer (to detect gold) and per-search token
        # accounting (the clock). Native arms have neither: their search-result
        # tokens are billed on the model spans, not attributable per search.
        return _not_measurable(
            "token_discounted_gain", surface,
            "snippets or per-search token accounting")
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
    surface = _surface(output)
    if surface not in _SNIPPET_SURFACES:
        # Titles alone would compress very differently from title+snippet, so
        # running this on a no-snippet arm produces a number that is not
        # comparable to the harness arms even though it looks like one.
        return _not_measurable(
            "compression_redundancy", surface, "result snippets")
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
    surface = _surface(output)
    if surface not in _URL_SURFACES:
        return _not_measurable("domain_entropy", surface, "result URLs")
    hosts = [_host(r.get("url", "")) for r in _all_results(output)]
    hosts = [h for h in hosts if h]
    if len(hosts) < 2:
        return {"name": "domain_entropy", "score": None,
                "metadata": {"applicable": False, "decision_surface": surface,
                             "n_results": len(hosts)}}
    counts = Counter(hosts)
    n = len(hosts)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    return {"name": "domain_entropy",
            "score": H / math.log(n),
            "metadata": {"applicable": True, "decision_surface": surface,
                         "unique_hosts": len(counts), "n_results": n,
                         "top_host": counts.most_common(1)[0]}}


def gated_answer_match(input, output, expected, metadata=None, **kwargs):
    """Answer correctness with hard-rule violations zeroed — one headline number.

    `dealbreaker_gate` says analysis should multiply the answer score by it, but
    nothing implemented that, so the two have been sitting side by side as
    unrelated columns: an arm can top the answer score while leaking gold sources
    or blowing the search budget. This composes them.

    Vals AI's Web Search Index (vals.ai/benchmarks/web_search) independently
    arrived at dealbreaker-gating as its scoring primitive, gating on load-bearing
    *facts*; this gates on rule *compliance*. Both share the property that matters
    — a violation cannot be averaged away by good performance elsewhere.

    Composed from the existing scorers rather than reimplementing them, so the
    rules cannot drift apart.

    None-propagation is deliberate and asymmetric:
      * no gold answer -> None (nothing to score)
      * gate violated  -> 0.0  (measured, and invalidating)
      * gate not measurable -> the answer score passes through, with
        gate_applied=False recorded.

    That last case is the no-tool control arm, which has no retrieval and so
    cannot violate a retrieval rule. Dropping it to None would remove the
    parametric baseline from the gated comparison, which is the one number every
    search arm has to be subtracted from. Filter on gate_applied if you want the
    strictly-gated subset.

    Uses the deterministic matcher, not the LLM judge, so this number is
    reproducible and costs nothing to recompute. A judge-gated headline needs an
    analysis-side join instead — Braintrust scorers cannot read each other's
    outputs.
    """
    answer = qa_answer_match(input, output, expected, **kwargs)
    if answer.get("score") is None:
        return {"name": "gated_answer_match", "score": None,
                "metadata": {"applicable": False,
                             "reason": "no gold answer on this row"}}
    gate = dealbreaker_gate(input, output, expected, metadata=metadata, **kwargs)
    gate_score = gate.get("score")
    gate_applied = gate_score is not None
    violated = gate_score == 0.0
    return {
        "name": "gated_answer_match",
        "score": 0.0 if violated else answer["score"],
        "metadata": {
            "applicable": True,
            "answer_score": answer["score"],
            "match_type": answer.get("metadata", {}).get("match_type"),
            "gate_applied": gate_applied,
            "gate_violated": violated,
            "violated_rules": gate.get("metadata", {}).get("violated", []),
            "gates_checked": gate.get("metadata", {}).get("gates_checked", []),
            "decision_surface": _surface(output),
        },
    }


DETERMINISTIC_SCORERS = [
    qa_answer_match,
    gated_answer_match,
    leakage_guard,
    dealbreaker_gate,
    budget_economy,
    temporal_grounding,
    snippet_sufficiency,
    evidence_precision,
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
