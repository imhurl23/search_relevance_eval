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
- harness `provider_http_latency_s`, `rate_limit_wait_s`, and `retry_backoff_s`;
- mean and P95 search calls;
- cached-input use and the applicable price multiplier.

Prices come from the pinned table in `agents.py`. Confirm current public prices
before a publication run and record the check date. Do not substitute search
spend for total cost.

The matrix launcher requires an acknowledged maximum search bill for execution.
It computes the ceiling from selected rows, trials, five calls per row, every
condition, and all allowed condition attempts. This ceiling excludes model and
judge inference because the vendor APIs expose no common enforceable dollar
budget. `max_row_executions` bounds that remaining exposure and includes retries.

`latency_s` is deployed end-to-end latency and includes harness scheduling and
rate-limit waits. Use `provider_http_latency_s` for You.com's HTTP service time;
do not compare a You.com tool-span queue wait with a native provider's hidden
server-side scheduling as if both were retrieval latency.

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

These differences cannot be closed and bound what this contrast can claim:

- **Freshness has no native equivalent.** No native API exposes a freshness
  control, so that treatment axis is harness-only and freshness effects never
  cross the harness/native boundary.
- **Evidence volume is targeted, not enforced, on native arms.** The harness
  requests up to 5 web and 5 news results; native search content arrives inside
  `prompt_tokens` and can only be isolated by differencing the same model's
  `no_search` arm.
- **Tool capabilities differ.** OpenAI can search, open a page, and find text in
  a page. The harness exposes search-result highlights and no page-fetch tool.
  Rows record each native action type and every emitted native query.
- **Cache behavior is not controlled.** The You.com request sends
  `Cache-Control: no-cache` to intermediaries, but that header does not establish
  index or extraction freshness. Native providers control their own caches.
- **Search-call units differ.** One OpenAI tool action can carry multiple search
  queries, and `max_tool_calls` also covers page actions. Report native tool
  calls, search actions, emitted queries, and page actions separately.

Harness-versus-native is therefore a system comparison — You.com as configured
here against vendor-native search as shipped — not a retrieval-quality
comparison, and it does not support a claim that one retrieval engine beats
another. Within-arm contrasts carry no such limit.

## Validity checks

- Use one dataset snapshot, prompt version, model ID, and serving path within a
  contrast.
- Randomize condition order reproducibly with `run_matrix.py`; record
  `matrix_order_seed` and `matrix_order_index`.
- Execute from a clean tracked worktree. The checkpoint binds resume operations
  to the git commit, pinned dataset version, row count, order, concurrency, and
  error policy.
- Require the selected row count to equal `expected_rows` before every condition.
- Treat task, scorer, and classifier errors above the registered threshold as a
  failed condition. A retry writes a separate experiment; never append retry rows
  to a partial attempt.
- Report gold-domain availability and `exclusion_requested` by arm. Audit
  returned domains separately because a sent filter does not prove enforcement.
- Report observed results per search against `result_count_target_per_search`;
  only the harness enforces the 5-web/5-news section maxima.
- Report `search_budget_enforced`; OpenAI uses `max_tool_calls` across search,
  page-open, and find actions, while Anthropic carries the remaining search
  `max_uses` budget across continuations.
- Treat Anthropic `page_age` as last-updated, not publication time. The current
  You.com API reference does not define one publication-time construct across
  both result sections. `temporal_grounding` uses news results explicitly marked
  with `date_semantics: publication` and excludes other provider date fields.
- Read both You.com sections and interleave them; news is additive, not capped
  into the arm's result count. The API applies `count` per section, so surface
  size runs 1x to 2x count depending on whether the query has news intent.
  Record `n_web_results` and `n_news_results` per row and condition on them.
  The variation is per query rather than per arm, so it does not confound the
  setup contrast, but it does mean `n_results` is not a constant and must not
  be compared against a single requested count.
- Use only results marked `date_semantics: publication` when auditing
  `temporal_grounding`.
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
changes the visible tool contract and the provider's hidden search orchestration.

OpenAI Terra and the Anthropic native-search adapter passed live checks on
August 14, 2026, using Opus. Sonnet 5 replaced Opus in the registered matrix on
August 24, 2026. The five-row-per-condition pilot
`sonnet-parity-pilot-2026-08-24-v2` then completed all 14 conditions on commit
`0b35de6`: 70/70 rows, zero task/scorer/search errors, no retries, and observable
native search actions from both frontier vendors. The audit recorded five
quoted-query operator violations and two zero-search Sonnet harness rows; retain
those registered observability fields in the analysis rather than silently
rewriting or dropping the model decisions. Both Baseten model routes passed live
checks on August 14, 2026.
