# Exploratory mixed-effects analysis: what matters, and what moves evidence, answers, and quality

Companion to [the variable analysis](livenewsbench-full-sonnet-v2-variable-analysis.md) and
[the context-timing analysis](livenewsbench-full-sonnet-v2-context-timing-analysis.md).
Reproduce with `python analysis/mixed_effects_analysis.py`.

This is the exploratory follow-up. The preregistered primary model lives in
[the registered mixed-effects report](livenewsbench-full-sonnet-v2-mixed-effects.md),
produced by `analysis/run_mixed_effects.py`. The two answer different questions and
are kept in separate files.

## What was fitted

18,606 rows = 1,329 dataset rows (1,129 unique task keys) x 14 model-arm conditions, one
generation per cell. Every model is a **random-intercept linear mixed model with the task key
as the grouping factor**, so the pairing across conditions is modelled rather than assumed
away. On 0/1 outcomes this is a linear probability model with partial pooling: every
coefficient reads directly in percentage points. GEE logistic regression with an exchangeable
working correlation clustered on task key is reported as a robustness check.

The design matrix is built cell-by-cell because the Native arm was only run for Claude and
GPT. Empty interaction cells are omitted, which keeps the model full rank: 14 cells =
intercept + 3 model + 3 arm + 7 estimable interactions. Reference cell is
**GPT-5.6 Terra / Normalized**.

Fixed-effect blocks:

| Block | Contents |
|---|---|
| `task_context` | category (9 dummies, small categories pooled), quantitative/composed flag, event age in 30-day units |
| `model_identity` | Claude Sonnet 5, DeepSeek V4, GLM-5.2 vs GPT-5.6 Terra |
| `retrieval_arm` | No search, Wide, Native vs Normalized |
| `model_x_arm` | the 7 model-arm cells that were actually run |

Retrieved-evidence metrics only exist in the normalized-harness arms, so evidence models are
fitted on the clean 4 x 2 (model x Normalized/Wide) subset of 10,632 rows.

## Answer 1: what is most important

Two numbers per outcome carry the ranking. `ICC0` is the share of variance attributable to
*which question it is*, from an intercept-only model. `dR2 unique` is the marginal
pseudo-R-squared a block contributes that no other block can supply.

| Outcome | ICC0 (question) | Retrieval arm | Model x arm | Task context | Model identity | Full R2m / R2c |
|---|---:|---:|---:|---:|---:|---:|
| Semantic judge pass | 0.210 | **0.0263** | 0.0248 | 0.0141 | 0.0006 | 0.382 / 0.603 |
| Gated answer match | **0.526** | 0.0058 | 0.0051 | 0.0176 | 0.0010 | 0.099 / 0.614 |
| Searches issued | 0.084 | **0.1052** | 0.0103 | 0.0021 | 0.0083 | 0.600 / 0.723 |
| Log latency | 0.054 | 0.0066 | **0.1183** | 0.0093 | 0.0037 | 0.309 / 0.370 |
| Log answer length | 0.072 | 0.0131 | 0.0822 | 0.0043 | **0.1345** | 0.642 / 0.752 |

Sequentially (the more familiar reading), retrieval availability alone takes marginal R2 on
the judge outcome from 0.046 to 0.357 — a **+0.312 jump**, an order of magnitude larger than
any other block. Every block is jointly significant on both quality outcomes
(all Wald p < 1e-4).

**The importance ordering, in one list:**

1. **Retrieval availability** dominates everything for quality and search behaviour.
2. **Question identity** is the next-largest source of variance, and on the strict gated
   outcome it is the largest single factor: 52.6% of gated variance is the question, not the
   system. The full fixed-effect model explains only 9.9% of gated variance marginally, but
   61.4% conditionally.
3. **Model x retrieval interaction** beats **model identity** in unique R2 by 41x on judge
   (0.0248 vs 0.0006) and 5x on gated (0.0051 vs 0.0010). "Which model is best" is a much
   weaker fact than "which model depends on retrieval."
4. **Model identity** is nearly irrelevant to quality once retrieval is accounted for, but it
   is the single largest driver of answer length and a large driver of latency.
5. **Result-count tier** (Wide vs Normalized) contributes almost nothing to quality
   (unique R2 0.0001) but a lot to context shape — see Answer 2.
6. **Event age** has no main effect on quality (judge +0.0003/30 days, p = 0.93; gated
   -0.0039, p = 0.39) but is a strong *moderator* — see Answer 4.

### Per-model treatment effects (percentage points, 95% CI, BH-adjusted within outcome)

Retrieval gain = Normalized minus No search:

| Model | Judge | Gated |
|---|---|---|
| GLM-5.2 | **+76.4** (74.1, 78.7) | **+34.2** (32.0, 36.3) |
| DeepSeek V4 | **+72.6** (70.3, 74.9) | **+31.5** (29.3, 33.6) |
| Claude Sonnet 5 | **+54.6** (52.3, 56.9) | **+23.4** (21.2, 25.6) |
| GPT-5.6 Terra | **+34.2** (31.9, 36.5) | **+15.7** (13.6, 17.9) |

Differences in retrieval gain are large and precise: GLM minus GPT is **+42.2 pp** judge
(38.9, 45.5) and **+18.4 pp** gated (15.4, 21.5). This reproduces the task-bootstrap result
and tightens it.

Native vs Normalized (defined only for GPT and Claude): GPT loses **-3.6 pp** judge
(-5.9, -1.3) and **-3.3 pp** gated (-5.5, -1.2) on its own toolchain relative to the common
harness; Claude is indistinguishable (-0.5 and -0.9, both n.s.). GEE logistic agrees
(GPT Native OR 0.77 judge, 0.87 gated, both p ≤ 0.001).

## Answer 2: what changes the evidence

Fitted on the 4 x 2 subset. The striking result is the ICC column:

| Evidence outcome | ICC0 (question) | Unique R2: result tier | Unique R2: model | Wide - Normalized (pooled) |
|---|---:|---:|---:|---|
| Literal gold visible in snippets | **0.818** | 0.0000 | -0.0000 | +1.2 pp (p_BH 0.038 for 3 of 4 models) |
| Evidence precision | **0.816** | 0.0015 | 0.0008 | **-1.2 to -1.8 pp** (all p < 1e-8) |
| Token-discounted gain | **0.820** | 0.0005 | 0.0000 | **+1.2 to +2.0 pp** (all p_BH ≤ 0.001) |
| Temporal grounding | 0.463 | 0.0061 | 0.0015 | **-7.5 to -9.7 pp** (all p < 1e-20) |
| Source diversity | 0.375 | 0.0217 | 0.0063 | **-4.9 to -6.2 pp** (all p < 1e-100) |
| Distinctness / low redundancy | 0.249 | 0.0508 | 0.0089 | **-6.1 to -6.9 pp** (all p < 1e-250) |

Read this as a clean separation:

- **Whether the answer is findable at all is a property of the question, not the system.**
  82% of the variance in literal-gold visibility, evidence precision, and token-discounted
  gain is between questions. Neither the model nor the result tier moves it materially
  (tier unique R2 ≤ 0.0015; model identity is not even significant for gold visibility,
  p = 0.20).
- **The result tier changes the *shape* of the context, strongly and consistently.** Wide
  retrieval lowers temporal grounding, source diversity, and distinctness by 5–10 points in
  every model, and lowers per-snippet precision. It buys a ~1 pp increase in literal-gold
  visibility and a 1–2 pp increase in token-discounted gain. That is the trade in a sentence:
  **more chances to see the gold string, in a noisier, more redundant, less temporally
  anchored, less diverse context.**
- Model x tier interactions on evidence are negligible (unique R2 ≤ 0.0010). The wide payload
  degrades context the same way for all four models.

## Answer 3: what leads to changes in answers and quality

### Behaviour

Turning retrieval on adds **+2.04 to +2.58 searches** and raises answer length by
0.39 log-words (GPT) to 2.59 log-words (GLM). Verbosity is overwhelmingly a model property
(unique R2 0.134) — the harness barely touches the ranking.

Latency is the one outcome where **model x arm is the dominant block** (unique R2 0.118,
chi2 3500 on 7 df). The arms are not a common latency tax: retrieval costs GLM +1.82 log-s
and Claude +1.06 log-s, but GPT +0.03 (n.s.). Native tooling costs GPT +0.31 log-s while
using 0.36 *fewer* searches — a different serving path, not just a different search count.

### Which evidence properties predict a correct answer

Within retrieval arms, holding model, tier, and task context fixed, per-SD associations with
each outcome (these are **post-treatment conditional associations, not causal effects**):

| Mediator (per 1 SD) | Gated answer match | Semantic judge pass |
|---|---|---|
| Literal gold visible in snippets | **+3.75 pp** (p 7e-10) | +0.51 pp (n.s.) |
| Evidence precision | **+3.35 pp** (p 2e-8) | +1.55 pp (p 0.009) |
| Searches issued | -1.59 pp (p 3e-5) | **-4.07 pp** (p 3e-24) |
| Temporal grounding | -0.65 pp (p 0.06) | +0.39 pp (n.s.) |
| Source diversity | +0.43 pp (n.s.) | +0.13 pp (n.s.) |
| Distinctness | +0.11 pp (n.s.) | -0.51 pp (n.s.) |
| Log answer length | +0.56 pp (n.s.) | +0.44 pp (n.s.) |

**The two outcomes are listening to different things.** The gated deterministic outcome
tracks retrieved evidence: gold visibility and precision are its top two predictors. The
semantic judge is essentially deaf to evidence quality and instead tracks search effort
negatively — extra searches are a struggle signal, not a quality signal. This is a second,
independent reason to treat the two scorers as measuring different constructs rather than one
being a noisy version of the other.

### Decomposing the Wide effect on the judge

Total Wide effect on judge (pooled over models, 4 x 2 subset): **+2.66 pp**. Adjusting for all
seven mediators leaves a **+1.85 pp** direct effect, implying roughly **30%** of the tier
effect runs through measured evidence and behaviour changes. Per model and BH-adjusted, the
judge gain survives for Claude (**+3.2 pp**, p_BH 0.012) and DeepSeek (**+2.4 pp**, p_BH 0.050),
is borderline for GLM (+2.2, p_BH 0.074) and absent for GPT (+1.2, p_BH 0.320). On gated the
Wide effect is zero for every model (|estimate| ≤ 0.5 pp, all p_BH ≥ 0.96), so the mediation
decomposition for gated is uninterpretable and is not reported.

So the earlier conclusion holds with a small amendment: **wide retrieval is mainly a
context-quality trade, but it is not literally free — it produces a small, real semantic-judge
gain for the two models that were most retrieval-dependent, and no gain on strict match.**

## Answer 4: the retrieval gain is not a constant

Two results say the headline number is a property of the *slice*, not of the system.

**Event age moderates it.** On the No-search vs Normalized subset, adding interactions:

| Term | Judge | Gated |
|---|---|---|
| Retrieval (at mean age, non-quantitative) | +36.7 pp (34.0, 39.5) | +10.7 pp (8.1, 13.2) |
| Retrieval x event age, per 30 days | **-2.64 pp** (-3.18, -2.10), p < 1e-16 | **-1.26 pp** (-1.76, -0.77), p < 1e-6 |
| Retrieval x quantitative/composed | -4.92 pp (-7.36, -2.48) | +9.39 pp (7.13, 11.65) |

The mechanism is visible in per-arm age slopes. The **No-search** arm *improves* with event
age (**+1.96 pp per 30 days** on judge, 1.37 to 2.54, p < 1e-4; +0.75 pp on gated,
p = 0.004) while every retrieval arm is flat or slightly negative (-0.6 to -0.9 pp,
p ≥ 0.05). That is the signature of parametric leakage: older events are more likely to be in
training data, so the no-search baseline rises and the measured retrieval gain shrinks. Across
the 236–482 day age span in this dataset the judge retrieval gain falls from roughly **+45 pp
at the recent end to +24 pp at the old end**. Any published "retrieval is worth X points"
number from this benchmark is dated the moment it is written.

**Event age is confounded with split membership.** LiveNewsBench builds its
splits by event age, and the matrix spans all four splits of
`jan_2026_release_2`: `test` and `human_verified_test` run 236-298 days, `val`
298-329, and `train` 328-482. Age and split are therefore close to collinear, and
the decay above is equally describable as a split effect. The no-search judge
rate is 0.158 on `test` and 0.252 on `train`, with retrieval gains of 0.671 and
0.547. The parametric-leakage reading survives either description, because the
splits are defined by the same variable, but no analysis here separates the two.

**Questions differ more than models do.** Adding a random slope for retrieval by task key:
the SD of the per-question retrieval effect is **0.276** (judge) and **0.327** (gated) —
*larger* than the SD of the question intercept (0.181 and 0.160). The random-slope model beats
random-intercept decisively (LR chi2 1053 and 2888 on 2 df). Retrieval is near-useless on some
questions and near-decisive on others, and that spread exceeds the entire between-model spread.

## Caveats that constrain these numbers

- **Linear probability model.** Coefficients are percentage points and can in principle leave
  [0,1]. GEE logistic agrees on the sign and significance of every model, arm, and interaction
  term; the only terms that are non-significant in one and not the other are the Wide effects,
  which are non-significant in both on gated.
- **One generation per cell.** There is no within-cell replication, so run-to-run variance is
  bundled into the residual and cannot be separated from item-level noise. The ICCs are
  therefore a *lower* bound on the question's share and an upper bound on what looks like
  system-driven residual variance.
- **Mediators are post-treatment.** The b-paths in Answer 3 and the 30% indirect share are
  descriptive. Nothing here identifies the causal effect of, say, highlighting — highlights
  were on in every harness arm.
- **Differential missingness.** Temporal grounding is missing for 10.0% of Normalized rows and
  11.4% of Wide rows. The evidence models drop those rows, which is conditioning on a
  post-treatment variable. The imbalance is small but the -7.5 to -9.7 pp Wide effect on
  grounding should be read as "among rows where grounding was computable."
- **Native is not a factorial.** Native was only run for GPT and Claude, so its contrast is
  within-model only and does not enter the model x arm comparison for DeepSeek or GLM.
- **Multiplicity.** Per-model arm contrasts are BH-adjusted within outcome. The retrieval and
  interaction effects are orders of magnitude beyond any multiplicity concern; the Wide-on-judge
  effects are the only findings whose survival depends on the correction.
- **Category is not randomised.** Task-context effects (quantitative questions score -10.2 pp
  on judge but +4.6 pp on gated; Sports +18.4 pp on gated; Disasters -6.0/-8.9 pp) are
  descriptive slice differences, not treatment effects.

## Outputs

| File | Contents |
|---|---|
| `mixed_importance.csv` | ICC, marginal/conditional R2, sequential and unique dR2, block Wald tests |
| `mixed_fixed_effects.csv` | every fixed-effect coefficient with CI, p, and standardised effect |
| `mixed_contrasts.csv` | per-model arm contrasts and model x retrieval differences, BH-adjusted |
| `mixed_mediation.csv` | a-paths, per-SD b-paths, total/direct Wide effect |
| `mixed_moderation.csv` | retrieval x age and retrieval x question-type interactions |
| `mixed_heterogeneity.csv` | random-slope SDs and LR tests for per-question retrieval effects |
| `mixed_gee_logistic.csv` | GEE logistic odds ratios, exchangeable, clustered on task key |
| `mixed_effects_run.log` | full console transcript of the run |

---

## Attribution

Questions and reference answers come from LiveNewsBench, release
`jan_2026_release_2`, pinned at commit `8a6b96e`, MIT licensed. The benchmark is
the work of Yunfan Zhang, Kathleen McKeown, and Smaranda Muresan —
[arXiv:2602.13543](https://arxiv.org/abs/2602.13543). Cite it alongside any
result taken from this report; the BibTeX entry is in the
[repository README](../README.md#citation).

The rows carry the upstream canary asking that the benchmark never enter a
training corpus. These runs use five searches and zero page visits, which
differs from the benchmark's default allowance of five each.
