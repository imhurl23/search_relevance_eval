# Highlights and failure analysis

Status: exploratory follow-up to the completed
`livenewsbench-full-sonnet-v2` matrix.

## What this experiment can say about highlights

It cannot identify a causal effect of You.com highlights. Every harness
condition requested `extraction_mode: highlights`; there is no otherwise
identical snippet-only or no-extraction control. The normalized-versus-wide
comparison changes result count, not extraction mode.

`snippet_sufficiency` is still useful as a mediator. It asks whether a literal
gold alias appeared anywhere in the title/snippet layer delivered to the model.
This is a conservative floor, not a semantic evidence oracle. It also cannot
prove that text came from `contents.highlights`: when highlights are absent,
the adapter falls back to snippets or descriptions, and that provenance was not
recorded per result.

## Highlight-layer association

Search-treated, row-level normalized-arm results:

| Model | Literal gold visible | Gated accuracy when visible | Gated accuracy when not detected | Judge accuracy when visible | Judge accuracy when not detected |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 | 12.1% | 82.6% | 29.1% | 91.3% | 77.5% |
| DeepSeek V4 Flash | 12.1% | 80.6% | 29.7% | 88.8% | 79.6% |
| GPT-5.6 Terra | 11.6% | 86.4% | 36.1% | 94.2% | 83.7% |
| Claude Sonnet 5 | 11.5% | 75.5% | 29.5% | 87.1% | 78.0% |

Literal gold visibility is therefore strongly associated with deterministic
correctness, but the low 11–12% sufficiency rate is not credible as the full
semantic evidence-coverage rate. The judge still accepts 77–84% of answers when
the string-based snippet scorer detects no gold. This points to missing aliases,
semantic paraphrases, useful indirect evidence, and possible judge
over-acceptance.

The strongest utilization failures are rows where literal gold was visible but
the gated answer was wrong. Counts in the normalized arms are 28 GLM, 31
DeepSeek, 21 GPT, and 36 Claude. One clear example is Claude returning “I could
not find this” on the New Year's drone-count question despite a literal gold
alias appearing in the surfaced evidence.

## What wide retrieval changed

Wide retrieval slightly increased literal-gold coverage but diluted evidence:

| Model | Sufficiency: normalized → wide | Evidence precision: normalized → wide | Gated wide − normalized |
|---|---:|---:|---:|
| GLM-5.2 | 12.1% → 13.3% | 3.47% → 2.22% | +0.003 |
| DeepSeek V4 Flash | 12.1% → 13.3% | 3.57% → 2.30% | +0.004 |
| GPT-5.6 Terra | 11.6% → 12.1% | 4.22% → 2.45% | −0.002 |
| Claude Sonnet 5 | 11.5% → 12.8% | 3.79% → 2.23% | +0.000 |

This is a consistent pattern: more results marginally increase the chance that
a literal answer appears, but reduce the density of answer-bearing results by
roughly one-third to two-fifths. The net accuracy effect is zero. More
highlighted content is not automatically more useful context.

## What the systems are getting wrong

### 1. Multi-source arithmetic and temporal composition

The clearest genuine weakness is combining several retrieved facts correctly.
A heuristic quantitative/composed subset has lower judge accuracy than other
factual questions for every normalized model:

| Model | Quantitative/composed | Other factual |
|---|---:|---:|
| GLM-5.2 | 72.5% | 86.4% |
| DeepSeek V4 Flash | 73.6% | 88.4% |
| GPT-5.6 Terra | 79.5% | 90.9% |
| Claude Sonnet 5 | 72.0% | 87.1% |

Observed errors include subtracting the wrong intermediate count, selecting the
wrong pair of dates, and confusing “submitted” with “considered.” Examples:

- Booker Prize: models use 153 as both submitted and considered, answering 0
  instead of the labeled difference of 18.
- Japan budget: models compare 10.8% with 9.4% and answer 1.4 points instead of
  the labeled 7 points.
- Seychelles assembly: Claude sums the coalition seats to all 34 instead of the
  labeled one-seat shortfall.

### 2. Entity disambiguation across similar reports

The systems often retrieve the right story family but bind the wrong entity or
attribute: Slovenia instead of Norway or the Netherlands, the wrong tanker in
the Black Sea incident, and the wrong sport omitted from a later summary.

### 3. Exact-match scoring misses semantically correct answers

The normalized arms contain 573–601 rows per model where gated string match is
zero but the semantic judge says correct. Examples include:

- `Francis “Chiz” Escudero` versus gold `Francis Escudero`;
- `2 additional years` versus gold `Two years.`;
- `during the renovation's design and execution phase` versus the corresponding
  gold phrase.

Hard-rule gating is not the source of this gap: only 1–3 normalized rows per
model fail the dealbreaker gate. The divergence is overwhelmingly raw matcher
versus judge behavior. A stratified human audit is necessary because the judge
may also be over-permissive.

### 4. Likely label or question-construction audit candidates

On 98 of 1,129 unique tasks, all four normalized models are rejected by the
semantic judge. Several show cross-model agreement on the same alternative
answer, making them high-priority label audits rather than automatic model
failures:

- Chile Senate arithmetic: all models derive 20 − 6 = 14; the gold is 13.
- Dutch FvD seats: all models report an increase from 3 to 7 seats, or 4; the
  gold is 3.
- Merz/Ukraine package dates: models report May 6 to May 28, or 22 days; the
  gold is 26.
- Taiwan arms-package components: three models total $91.4M + $375M + $353M as
  about $819M; the gold is $910M. GLM reaches the gold only by apparently
  counting the $91.4M component twice.

These examples are audit candidates, not proven bad labels: shared retrieval
sources can make all models converge on the same wrong interpretation.

## Recommended next experiment

To estimate the highlights feature itself, run a paired harness ablation with
the same model, prompt, query, count, freshness, search budget, and temporal
block, changing only the extraction surface:

1. highlights;
2. standard snippets/no extraction;
3. optionally full-page extraction as a separately priced arm.

Record per-result extraction provenance so fallback descriptions cannot be
silently pooled with actual highlights. Primary outcomes should include answer
accuracy, semantic evidence sufficiency, latency, context tokens, and total
cost. The current `snippet_sufficiency` string matcher should be supplemented
with a blinded per-result semantic oracle.

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
