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

The matrix contains 22 conditions:

- Four models each run no search and four You.com harness arms: 20 conditions.
- OpenAI and Anthropic each add one native-search arm: 2 conditions.

The harness arms are `normalized`, `native_fresh`, `fresh_week`, and `wide`.
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

Apply Holm correction across every primary test produced by these four contrast
definitions:

1. `harness(normalized) - none` within each model.
2. Each open model's best prespecified harness setup against each frontier
   no-search condition.
3. `native - harness(normalized)` within OpenAI and within Anthropic.
4. `harness(native_fresh) - harness(normalized)`, pooled across models.

Define the setup used in contrast 2 before examining test results. Development
data may select it; test data may estimate it.

The following analyses are exploratory:

- `fresh_week - native_fresh`;
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

1. Average repeated trials within each `task_key` and condition.
2. Pair conditions on `task_key`.
3. Compute the mean paired difference.
4. Bootstrap tasks for a 95% confidence interval.
5. Apply Holm correction to the registered contrast family.
6. Report win, tie, and loss counts.
7. Report results by `benchmark_category`; show any pooled mean as a summary.
8. Report row exclusions and missing-score denominators by arm.

Before interpreting an effect, compare it with the across-trial spread. Increase
the trial count when the effect is smaller than normal run variation. The current
default of three trials has no formal power analysis.

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

## Validity checks

- Use one dataset snapshot, prompt version, model ID, and serving path within a
  contrast.
- Interleave conditions in time.
- Report gold-domain availability and `exclusion_enforced` by arm.
- Report `search_budget_enforced`; OpenAI native search lacks an API-level cap.
- Treat You.com and Anthropic dates as last-modified timestamps.
- Use Corvus-QA `recency_rung` for the main freshness analysis.
- Use a cross-vendor judge jury for reported frontier comparisons.
- Inspect disagreements between deterministic answer match and semantic judges.

## Scope

The design estimates effects for the pinned models, datasets, prompts, and search
systems. It covers short factual answers. One external search API cannot establish
a general result about independent retrieval providers. Native-search evidence
is less observable than harness evidence. The native-versus-harness contrast also
changes one tool-specific prompt sentence.

OpenAI and Anthropic gateway adapters passed live checks on August 10, 2026.
Baseten gateway routing and current open-model tool use still need a live check
before the full matrix.
