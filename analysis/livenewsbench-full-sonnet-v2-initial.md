# LiveNewsBench full matrix: initial analysis

Status: preliminary, generated 2026-08-26 from the completed
`livenewsbench-full-sonnet-v2` checkpoint. The export contains 18,606 scored
dataset roots: 1,329 rows in each of 14 conditions. Repeated `task_key` values
are aggregated before pairing, leaving 1,129 unique paired tasks before
protocol exclusions.

## Condition-level results

These are task-aggregated deployed-condition means. They include zero-search
and degraded-search rows; the registered paired estimates below apply the row
handling rules from `docs/study-design.md`.

| Model | No search | Normalized | Wide | Native |
|---|---:|---:|---:|---:|
| GLM-5.2 | 0.012 | 0.359 | 0.362 | — |
| DeepSeek V4 Flash | 0.041 | 0.361 | 0.366 | — |
| GPT-5.6 Terra | 0.271 | 0.422 | 0.421 | 0.391 |
| Claude Sonnet 5 | 0.124 | 0.357 | 0.356 | 0.348 |

The corresponding single-judge means are much higher:

| Model | No search | Normalized | Wide | Native |
|---|---:|---:|---:|---:|
| GLM-5.2 | 0.027 | 0.793 | 0.814 | — |
| DeepSeek V4 Flash | 0.080 | 0.804 | 0.829 | — |
| GPT-5.6 Terra | 0.518 | 0.849 | 0.859 | 0.814 |
| Claude Sonnet 5 | 0.258 | 0.792 | 0.823 | 0.787 |

The large deterministic/judge gap is itself a major finding to audit before
choosing a headline accuracy number.

## Registered gated-answer contrasts

Search-treated estimates exclude zero-search and fully failed search rows.
Refusals and truncations are excluded per protocol. Confidence intervals use a
task bootstrap; Holm-adjusted p-values use the paired large-sample normal test
across the ten registered gated-answer contrasts.

| Contrast | Paired tasks | Effect | 95% CI | Holm p |
|---|---:|---:|---:|---:|
| GLM normalized − no search | 1,128 | +0.347 | [0.318, 0.375] | <1e-12 |
| DeepSeek normalized − no search | 982 | +0.328 | [0.300, 0.360] | <1e-12 |
| GPT normalized − no search | 1,129 | +0.151 | [0.125, 0.178] | <1e-12 |
| Claude normalized − no search | 1,084 | +0.243 | [0.215, 0.270] | <1e-12 |
| GLM normalized − GPT no search | 1,128 | +0.089 | [0.062, 0.117] | <1e-9 |
| GLM normalized − Claude no search | 1,126 | +0.236 | [0.211, 0.262] | <1e-12 |
| DeepSeek normalized − GPT no search | 1,116 | +0.093 | [0.064, 0.121] | <1e-9 |
| DeepSeek normalized − Claude no search | 1,114 | +0.240 | [0.215, 0.268] | <1e-12 |
| GPT native − normalized | 1,128 | −0.031 | [−0.051, −0.012] | 0.0042 |
| Claude native − normalized | 1,079 | −0.010 | [−0.028, 0.009] | 0.312 |

The single semantic judge points in the same direction. Retrieval gains are
larger on that score (+0.766 GLM, +0.717 DeepSeek, +0.331 GPT, +0.554 Claude).
GPT native trails its normalized harness by 0.035 [0.016, 0.057], while the
Claude native/harness difference remains indistinguishable from zero.

## Initial interpretation

1. Retrieval provides a large, consistent accuracy gain for every model. The
   gain is largest for the two open models because their no-search baselines are
   very weak on rolling news.
2. Both open models with normalized retrieval exceed both frontier no-search
   baselines on gated answer match. This satisfies the accuracy half of the
   capability-substitution claim, but not yet the required lower-total-cost
   half.
3. GPT's normalized external-search harness outperforms GPT native search by
   about three percentage points. Claude's native and normalized conditions are
   statistically tied. These remain system comparisons, not retrieval-provider
   comparisons.
4. Wide retrieval does not improve gated answer match over normalized for any
   model. Paired effects range from −0.002 to +0.004 and every interval crosses
   zero. The extra result volume therefore has no demonstrated exact-answer
   benefit in this run.
5. Sports shows the largest exploratory normalized-retrieval gain for every
   model (+0.27 GPT, +0.52 Claude, +0.61 GLM, +0.63 DeepSeek). Category effects
   are exploratory and some categories are small.

## Data-quality and interpretation checks

- DeepSeek no-search has 170 truncated answers (12.8%), reducing its registered
  paired retrieval contrast to 982 tasks. A sensitivity analysis that treats
  truncation as failure, rather than excluding it, is required.
- Claude normalized has 48 zero-search rows (3.6%). Its normalized-vs-control
  effect is +0.243 among search-treated rows and +0.233 in the deployed-condition
  sensitivity estimate.
- No condition has a fully failed search row. Degraded-search counts are 4 for
  DeepSeek normalized, 12 for DeepSeek wide, and 22 for GPT native.
- The dataset contains both `Law and crime` and `Law and Crime`; normalize this
  label before publishing category estimates.
- The current root export does not contain trace-aggregated `total_cost_usd`.
  Cost/Pareto and the cost half of capability substitution are therefore not
  yet reportable.
- The semantic result uses one judge. Per the preregistered limitations, add a
  multi-vendor jury before using judge scores for frontier claims.
- The bootstrap analysis is the reproducible baseline. Publication analysis
  still calls for a task-random-intercept mixed-effects model.

## Next analysis steps

1. Aggregate model and search costs from child spans and build the cost-quality
   frontier.
2. Export all deterministic mediator scores and analyze snippet sufficiency,
   evidence precision, temporal grounding, domain entropy, redundancy, and
   token-discounted gain.
3. Run truncation, zero-search, and degraded-search sensitivity tables.
4. Normalize category labels and produce category-balanced and category-level
   figures.
5. Audit deterministic-vs-judge disagreements on a stratified sample.
