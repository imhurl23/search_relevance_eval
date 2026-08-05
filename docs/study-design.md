# Study design

Written before the runs. The point of committing this first is that the 23
conditions then execute against stated hypotheses, instead of producing a table
that gets interpreted afterwards — which is where a leaderboard comes from.

## Objective

**Not** "which search wins." That is a ranking, and
[Vals AI's Web Search Index](https://www.vals.ai/benchmarks/web_search) already
produces one: independent API (Exa) beats native provider search by ~6 points on
finance (p < 0.001), ties on legal (p = 0.86).

This study asks **what retrieval actually buys, for whom, and through what
mechanism.** Three claims, each requiring evidence the ranking design cannot
produce.

---

## Claim A — retrieval's marginal value

> For a given model, how much of its measured accuracy comes from retrieval
> rather than from what it already knew?

**Why it needs saying:** on time-sensitive questions this is the whole quantity of
interest, and an accuracy number without it is uninterpretable. A model scoring
48% with search may be scoring 45% without it. Vals reports no parametric
baseline, so their scores cannot be decomposed this way.

**Evidence required.** Per model: `search_mode=harness` minus `search_mode=none`,
paired by `task_key`, on the same pinned dataset version. Reported per
`recency_rung` where the dataset supplies it, because marginal value should
*increase* as the fact gets more recent — and if it does not, either the dataset
is contaminated or the retrieval layer is stale. That directional prediction is
the sharpest available check on our own instrument.

**Falsifiable prediction.** Marginal value is positive and monotonically
increasing across recency buckets. A flat or inverted profile falsifies either the
freshness premise or the dataset's cutoff separation.

---

## Claim B — retrieval as capability substitution *(candidate headline)*

> Can a cheap open-weights model plus good retrieval match an expensive frontier
> model without it — and at what cost ratio?

**Why it's the headline:** it is a procurement decision, not a leaderboard
position. `openai/gpt-oss-120b` is $0.10/$0.50 per MTok; `claude-fable-5` is
$10/$50 — **100× on input.** Retrieval costs $0.005–$0.015 per search. So the
question "how much model can you trade for how much retrieval" has a concrete
answer with budget consequences, and no published benchmark asks it because none
has both an OSS arm and a parametric floor.

**Evidence required.** `oss × harness` vs `frontier × none`, paired by
`task_key`, compared on `gated_answer_match` and on `total_cost_usd`. Both sides
must be priced — which is why the OSS arm's token price was added; an unpriced
side makes this rhetorical rather than measurable.

**Falsifiable prediction.** There exists a search configuration where
`oss × harness` ≥ `frontier × none` on gated accuracy at ≥10× lower total cost.
If the OSS model cannot use the tool reliably (high `zero_search_row`), the claim
fails for a *capability* reason rather than a retrieval one, and that distinction
must be reported, not smoothed over.

---

## Claim C — mechanism decomposition

> When one search configuration beats another, *through what*?

**Why it needs saying:** Vals reports a 6-point gap and a domain-dependence
finding they have no instruments to explain. Six metrics already logged here are
mediators, not decoration:

| mediator | mechanism it isolates |
|---|---|
| `snippet_sufficiency` | was the answer visible without a click |
| `evidence_precision` | signal density on the surface |
| `temporal_grounding` | was the surfaced evidence post-event |
| `domain_entropy` | source diversity |
| `compression_redundancy` | syndication / marginal information |
| `token_discounted_gain` | how *cheaply* gold appeared |

**Evidence required.** For each accuracy contrast, report the mediator deltas
alongside it, and check whether the mediator gap has the sign and magnitude to
account for the accuracy gap.

**Scope limit, stated up front.** Mediation is **complete across the three
independent APIs** (all six mediators computable on the `full` surface) and
**partial for native-vs-harness**: Anthropic native kills the four
snippet-derived mediators, OpenAI native kills those plus `temporal_grounding`.
So lead with mechanism across APIs; carry native as a bounded comparison.

---

## Methods contribution — native search is not auditable

Independent of A/B/C: **vendor-native search cannot be audited the way an
API-backed pipeline can, and this harness measures exactly how much.** The
`decision_surface` tier per row states which metrics die per vendor.

That has an edge beyond this study: if a vendor's search cannot be inspected for
source leakage, evidence sufficiency, or freshness, then benchmark results
obtained on it are not verifiable in the way API-based results are. Nobody
publishes this, and it falls out of instrumentation already built.

---

## Design

Two treatment axes, 23 conditions, per README "The test matrix". Two domains,
because one proves nothing given the +6 → 0 swing between Vals' two domains:

| domain | dataset | why it's here |
|---|---|---|
| news freshness | `LiveNewsBench` | fast-moving, short-form, high leakage risk |
| event-sourced fact transitions | `Corvus-QA` | supplies `recency_rung` and `coverage_tier`, so Claim A's recency prediction is directly testable |

`Corvus-QA` also carries curated `answer_aliases` and per-row `coverage_tier`,
which acts as a **headroom control**: a row no pinned reference search could
answer measures the ceiling, not the provider. Report with and without
unanswerable rows.

## Primary vs exploratory contrasts

Pre-specified, because 22 pairwise contrasts against one baseline inflates
family-wise error and a reviewer stops there.

**Primary (4).** Holm-corrected across this family:

1. `harness` − `none`, within each model *(Claim A)*
2. `oss × harness` − `frontier × none` *(Claim B)*
3. `native` − `harness(exa)`, within each frontier vendor *(the native question)*
4. `harness(exa)` − `harness(parallel|youdotcom)`, pooled *(does "independent API
   beats native" depend on which API)*

**Exploratory, reported without inference.** Everything else: freshness treatment
(`normalized` vs `native_fresh`), the six mediators, per-category and
per-`recency_rung` breakdowns, cross-vendor comparisons.

**Never reported as a contrast:** `native-openai` vs `native-anthropic` — two
variables (model and search implementation).

## Analysis rules, fixed in advance

- **Pairing.** All contrasts paired by `task_key` on one pinned
  `dataset_version`. Unpaired rows are dropped, and the drop count is reported.
- **No pooling across `benchmark_category`.** The pooled mean is a summary of the
  breakdown, never the result.
- **No pooling across `date_field_semantics`.** Exa/Parallel report publication
  dates; You.com and Anthropic native report last-modified. Different constructs.
- **Exclusions**, reported as counts and rates per arm, never silently:
  - `zero_search_row` — tool available, never used. Excluded from search-arm
    accuracy; reported separately, because an OSS tool-calling failure and a
    frontier model declining to search are different phenomena wearing one flag.
  - `model_refused` — a policy decline is not a wrong answer.
  - `answer_truncated` — a truncated answer is an instrumentation failure.
  - `dealbreaker_gate == 0` — handled by gating, not exclusion.
- **`None` scores are excluded from averages, never coerced to 0.** A metric that
  is not measurable on an arm must not contribute evidence about that arm.
- **Cost** is always reported as `search_cost_usd` + `model_cost_usd` +
  `total_cost_usd`. Rows with `model_cost_confirmed=False` are excluded from cost
  comparisons rather than read as cheap.
- **Variance envelope before effects.** Compute run-to-run spread across trials
  *first*. An effect smaller than the spread is not reportable, and the spread on
  a live-web benchmark is itself a publishable number nobody reports.

## Known unresolved before running

Carried from README "Remaining gaps" — these are the ones that bound the claims
above, not the full list:

1. The native-vs-harness contrast varies the prompt by one sentence. Bounded, not
   eliminated; no prompt-only control arm is built.
2. The OpenAI native arm is not held to the 5-search budget (no `max_uses`).
3. `--trials 3` has no power analysis behind it. The variance envelope above is
   the mitigation, and it may show 3 is insufficient.
4. No adapter has run against a live API. Nothing in this document is validated
   until a smoke run resolves that.
