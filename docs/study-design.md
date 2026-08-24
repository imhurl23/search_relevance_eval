# Study design

This document fixes the research questions, contrasts, exclusions, and reporting
rules before the full evaluation runs.

## Objective

Measure the contribution of retrieval to factual answer quality, cost, and
latency. Explain observed answer gains with evidence available from the search
trajectory.

The design answers three questions:

1. How much does retrieval improve each model over its no-search baseline?
2. Can either open model with retrieval reach a frontier model's no-search
   accuracy at lower total cost?
3. Which observable retrieval properties accompany an answer-quality change?

## Conditions

The matrix contains 14 conditions:

- Four models each run no search and two You.com harness arms: 12 conditions.
- OpenAI and Anthropic each add one native-search arm: 2 conditions.

The harness arms are `normalized` and `wide`.
The exact models, parameters, prices, and commands are in the
[README](../README.md#experimental-matrix).

## Datasets

| Dataset | Role | Useful metadata |
|---|---|---|
| LiveNewsBench | Rolling news accuracy and leakage | Event date and source domains where supplied |
| Corvus-QA | Fact changes with controlled recency | `recency_rung`, `coverage_tier`, previous answer |
| RetrievalQA | Historical retrieval pilot and control domain | Accepted answers and `answer_as_of` for dynamic rows |

Run each contrast on one pinned dataset version. Do not combine datasets into a
single headline score.

RetrievalQA dynamic rows carry an explicit historical reference date. Their
relative day/week filters resolve to historical date ranges. This preserves the
upstream label's time boundary.

## Claims and estimands

### Retrieval value

For each model, estimate:

```text
gated_answer_match(harness arm) - gated_answer_match(no search)
```

Pair rows by `task_key`. On Corvus-QA, report the effect within each
`recency_rung`. The registered directional prediction is a larger retrieval
gain for more recent fact changes.

### Capability substitution

Compare each open-model harness condition with each frontier no-search condition
on paired tasks. Report both gated accuracy and `total_cost_usd`.

The claim succeeds only for an observed model pair and search setup. Report tool
non-use and search failures with the effect because either can explain a weak
open-model result.

### Retrieval mechanism

For every reported harness accuracy contrast, report changes in these candidate
mediators:

| Metric | Observed property |
|---|---|
| `snippet_sufficiency` | A gold answer appears in a title or snippet |
| `evidence_precision` | Useful evidence density |
| `temporal_grounding` | Results marked after the event date |
| `domain_entropy` | Distribution across source domains |
| `compression_redundancy` | Repetition across snippets |
| `token_discounted_gain` | Tokens consumed before a gold answer appears |

These metrics describe associations. The design does not identify a causal
mediation effect.

Native arms support fewer mediator fields. Use `decision_surface` to determine
whether a metric is available, and compare each metric only across compatible
surfaces.

## Registered contrasts

Apply Holm correction across every primary test produced by these three contrast
definitions:

1. `harness(normalized) - none` within each model.
2. Each open model's `normalized` condition against each frontier no-search
   condition.
3. `native - harness(normalized)` within OpenAI and within Anthropic.

The following analyses are exploratory:

- `wide - normalized`;
- category and recency subgroups beyond the registered breakdowns;
- mediator analyses;
- cross-vendor model comparisons.

Do not compare OpenAI native search directly with Anthropic native search. That
comparison changes the model and the search implementation together.

## Row handling

Apply the same rules to every arm and report the affected count and rate.

| Condition | Treatment |
|---|---|
| Unpaired `task_key` | Drop from the paired contrast |
| `zero_search_row=True` | Exclude from the search-treated estimate; report separately |
| `search_fully_failed=True` | Exclude from the search-treated estimate; report separately |
| `search_degraded=True` | Keep; report a sensitivity analysis without it |
| `model_refused=True` | Exclude from answer accuracy; report separately |
| `answer_truncated=True` | Exclude as an instrumentation failure |
| `dealbreaker_gate=0` | Keep; `gated_answer_match` already assigns zero |
| Score is `None` | Exclude from that metric's denominator |
| `model_cost_confirmed=False` | Exclude from cost comparisons |

Keep an intent-to-treat estimate for search arms as a sensitivity analysis. It
includes tool non-use and search failure and therefore measures the deployed
condition rather than successful retrieval alone.

## Analysis

1. Pair conditions on `task_key`.
2. Compute the mean paired difference.
3. Bootstrap tasks for a 95% confidence interval.
4. Apply Holm correction to the registered contrast family.
5. Report win, tie, and loss counts.
6. Report results by `benchmark_category`; show any pooled mean as a summary.
7. Report row exclusions and missing-score denominators by arm.

The full matrix runs once. Estimate operational variance by repeating the same
100-question subset twice, close in time, after the full run. Expand replication
only when a reported effect is smaller than the observed repeat spread.

For publication, add a mixed-effects model with condition as a fixed effect and
task as a random intercept. The repository's bootstrap analyzer remains the
reproducible baseline.

## Cost and latency

Report these fields separately and together:

- `search_cost_usd`;
- `model_cost_usd`;
- `total_cost_usd`;
- `latency_s`;
- mean and P95 search calls;
- cached-input use and the applicable price multiplier.

Prices come from the pinned table in `agents.py`. Confirm current public prices
before a publication run and record the check date. Do not substitute search
spend for total cost.

## Harness versus native

The registered harness/native contrast targets up to 10 surfaced results per
search. You.com enforces that target as up to 5 web plus 5 news results by
applying `count: 5` independently to each section.

Native APIs cannot enforce the same split or an exact source count. OpenAI uses
`search_context_size: "high"`, its closest supported evidence-volume setting;
Anthropic runs at `max_uses` parity with dynamic filtering pinned off
(`allowed_callers: ["direct"]`) and exposes no content-volume parameter. Native
source volume is therefore observed and reported, not post-hoc truncated or
described as exact 5+5 parity. The shared target is recorded as
`result_count_target_per_search: 10`, alongside
`result_count_control: "unavailable_observed_only"` on native runs.

Three differences cannot be closed, and they bound what this contrast can claim:

- **Freshness has no native equivalent.** No native API exposes a freshness
  control, so that treatment axis is harness-only and freshness effects never
  cross the harness/native boundary.
- **Evidence volume is targeted, not enforced, on native arms.** The harness
  requests up to 5 web and 5 news results; native search content arrives inside
  `prompt_tokens` and can only be isolated by differencing the same model's
  `no_search` arm.
- **Only the harness arm can ask for an uncached result.** Freshness is the
  quantity under measurement, and `Cache-Control` has no native counterpart.

Harness-versus-native is therefore a system comparison — You.com as configured
here against vendor-native search as shipped — not a retrieval-quality
comparison, and it does not support a claim that one retrieval engine beats
another. Within-arm contrasts carry no such limit.

## Validity checks

- Use one dataset snapshot, prompt version, model ID, and serving path within a
  contrast.
- Interleave conditions in time.
- Report gold-domain availability and `exclusion_enforced` by arm.
- Report observed results per search against `result_count_target_per_search`;
  only the harness enforces the 5-web/5-news section maxima.
- Report `search_budget_enforced`; OpenAI uses `max_tool_calls`, while
  Anthropic carries the remaining `max_uses` budget across continuations.
- Treat Anthropic dates as last-modified timestamps. You.com dates are mixed:
  web results are last-modified, news results are publication timestamps. Split
  on each result's `source` before treating a date as a publication date.
- Read both You.com sections and interleave them; news is additive, not capped
  into the arm's result count. The API applies `count` per section, so surface
  size runs 1x to 2x count depending on whether the query has news intent.
  Record `n_web_results` and `n_news_results` per row and condition on them.
  The variation is per query rather than per arm, so it does not confound the
  setup contrast, but it does mean `n_results` is not a constant and must not
  be compared against a single requested count.
- News results are the only surface in the study reporting a true publication
  timestamp. Prefer them when auditing `temporal_grounding`, and split on each
  result's `source` before pooling dates.
- Do not pool runs across the change to POST + highlights + news results. That
  change enlarged and reordered the harness decision surface, and it was
  contributed by the search vendor under measurement; re-baseline instead.
- Use Corvus-QA `recency_rung` for the main freshness analysis.
- Use a cross-vendor judge jury for reported frontier comparisons.
- Inspect disagreements between deterministic answer match and semantic judges.

## Scope

The design estimates effects for the pinned models, datasets, prompts, and search
systems. It covers short factual answers. One external search API cannot establish
a general result about independent retrieval providers. Native-search evidence
is less observable than harness evidence. The native-versus-harness contrast also
changes one tool-specific prompt sentence.

OpenAI Terra and Anthropic Opus gateway and native-search adapters passed live
checks on August 14, 2026. Both Baseten model routes passed on the same date.
