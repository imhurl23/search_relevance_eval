# What to borrow from search benchmarks—and what this study uniquely answers

This memo compares the current LiveNewsBench analysis with the public search
benchmark pages from [Artificial Analysis](https://artificialanalysis.ai/agents/search-api)
and [Vals AI](https://www.vals.ai/benchmarks/web_search). It is an editorial and
analysis roadmap, not a claim that the studies are interchangeable.

## The positioning in one sentence

Artificial Analysis is strongest at **ranking search APIs**. Vals is strongest
at showing **which search implementation wins in a domain and at what cost**.
This study is strongest at explaining **when retrieval is needed, which models
benefit, what the retrieved context changes, and why a search-enabled answer
still fails**.

It also has a stronger reproducibility story than the analysis currently
foregrounds. The questions and expected answers come from the public
[LiveNewsBench dataset on Hugging Face](https://huggingface.co/datasets/YunfanZhang42/LiveNewsBench).
The [upstream benchmark code](https://github.com/YunfanZhang42/LiveNewsBench) is
public, and the separate search eval library contains the 14-condition matrix,
provider adapters, You.com normalization layer, native search integrations,
scorers, checkpointing, and analysis code built for this comparison. Add the
library's public URL before publication; the repository's currently configured
GitHub URL is not publicly accessible.

The headline should therefore not be another generic search leaderboard. A
better frame is:

> Search is not a fixed model capability or a universally beneficial add-on.
> Its value depends on model knowledge, event age, question composition, and
> the quality of the evidence surface; more retrieved content can add coverage
> without adding accuracy.

## What the two benchmark pages do well

### Artificial Analysis

- Gives readers a compact quality, cost, and latency leaderboard.
- Holds the answer model and agent harness fixed so the search provider is the
  main variable.
- Includes a no-search baseline and reports baseline lift.
- Separates search cost from model cost and search time from derived model
  time.
- Shows quality–cost and quality–latency Pareto frontiers.
- Makes provider tiers/configurations separate rows instead of hiding them in a
  provider average.
- Explains a useful second-order effect: better search results can reduce
  downstream model tokens, searches, and total cost.

### Vals AI

- Starts with three plain-English findings rather than a table dump.
- Uses task-level mixed-effects models to control for task difficulty.
- Breaks the result down by domain and task category, exposing where retrieval
  matters and where computation remains the bottleneck.
- Shows means and distributions, including the long tail of runaway search
  loops.
- Reports search calls, answer length, sources cited, and top cited domains.
- Uses a stacked model-cost/search-cost view and a Pareto frontier.
- Calls out a pathological individual run rather than allowing an average to
  conceal it.

## Add now using the existing exports

These additions would materially improve the current analysis without a new
evaluation run.

### 0. State what was built and what is public

Before presenting results, distinguish the three layers of the work:

1. **Public evaluation data:** LiveNewsBench provides the current-events
   questions and expected answers on Hugging Face.
2. **Upstream benchmark implementation:** the LiveNewsBench authors publish the
   code used to construct and evaluate the original benchmark.
3. **This evaluation library:** the comparison built a separate controlled
   structure for no search, normalized You.com, wide You.com, and provider-native
   search across four models.

The third layer is a contribution, not incidental plumbing. It makes the
provider comparison paired and reproducible, pins the dataset and condition
order, normalizes the external search surface, records result-level mediators,
and resumes interrupted runs without silently changing the study.

Link the Hugging Face dataset and both code repositories near the first methods
description and again in the final reproducibility section. This lets readers
reuse the questions, audit the upstream benchmark, or rerun and extend the
You.com-versus-native comparison.

### 1. A three-finding opening card

Put this before the longer causal discussion:

1. **Retrieval is a capability equalizer.** It adds 33–35 strict-match points
   for GLM and DeepSeek, 24 for Claude, and 15 for GPT, compressing a 25-point
   no-search spread to a roughly 7-point search-enabled spread.
2. **More results are not better results.** Wide retrieval slightly increases
   literal answer coverage, lowers evidence precision and distinctness, and
   produces essentially zero strict-match gain.
3. **Recency determines whether search is needed; composition determines
   whether search is enough.** Search removes most of the event-age penalty,
   but composed questions remain about 13 points harder after adjustment.

This is the most legible synthesis of the existing variable and timing reports.

### 2. One decision table instead of a single overall winner

Add a compact table with rows for common product questions:

| Product question | Evidence from this study | Decision implication |
|---|---|---|
| Is search worth invoking? | Retrieval lift by model and event age | Route on freshness and model |
| Should we return more results? | Wide vs. normalized plus mediator deltas | Optimize evidence per token, not count |
| Is native search automatically best? | Paired native vs. normalized effects | No; compare the deployed stack |
| Why did a searched answer fail? | Visibility, precision, composition, and failure cases | Change retrieval or reasoning based on failure class |
| When is the agent stuck? | Accuracy by latency quartile and search count | Escalate strategy after repeated unsuccessful searches |

### 3. Distribution and tail-risk views

Vals is right to show that means hide operational failures. Add:

- search calls per task by condition on a log-capable axis;
- latency median, P95, and distribution, not only the mean;
- the share of runs reaching the five-search cap;
- accuracy conditional on 0, 1, 2, 3, 4, and 5 searches, explicitly labeled
  as diagnostic rather than causal;
- truncation, refusal, fully failed search, and degraded-search rates beside
  answer accuracy.

Most of this is already present in the row-level export or analysis tables.
The presentation should distinguish deployed-system rates from eligible-answer
rates.

### 4. A failure funnel

Turn the existing qualitative failure work into a quantitative funnel:

```text
search invoked
  -> usable evidence returned
    -> answer-bearing evidence visible
      -> evidence used correctly
        -> composition/calculation correct
          -> answer accepted by deterministic and semantic scorers
```

Report a count at every observable stage and mark genuinely unobserved stages
as such. This would connect the mediator analysis, utilization failures,
composed-question gap, scorer disagreement, and label audits in one view.

### 5. A task-level win/tie/loss view

The paired effects already contain win, tie, and loss counts. Surface them.
Average uplift does not tell a product owner whether search helps many tasks a
little, rescues a smaller subset, or harms a meaningful minority. The GPT
normalized-vs-no-search contrast, for example, includes both wins and losses;
that is important evidence for a router.

### 6. Source behavior, with a careful scope

Borrow Vals's source analysis where the export permits it:

- unique domains per answer/search trajectory;
- concentration in the top one and top five domains;
- official/primary-source share;
- repeated-domain and near-duplicate-result share;
- source mix by category and by success/failure.

Do not equate more cited domains with better evidence. Tie the source metrics to
answer correctness, evidence precision, temporal grounding, and distinctness.
That mechanism link is the differentiator.

### 7. A limitations comparison box

Readers should see the estimand boundary immediately:

- Artificial Analysis isolates a search-provider effect for one fixed model,
  but allows page fetch and many turns.
- Vals estimates native-versus-Exa effects in two professional domains with
  multiple models, after removing domain-specific retrieval tools.
- This study estimates no-search, shared-harness, result-volume, and native
  system contrasts across four models on the same short-form live-news tasks,
  with a five-search cap and no arbitrary page fetch.

The studies answer related but different questions; raw scores should not be
compared across them.

## Add after cost fields are trace-complete

The current report correctly avoids a quality-per-dollar claim because the
root-span export lacks a defensible complete cost field. Once fixed, add:

1. stacked search and model cost per task;
2. mean, median, P95, and worst-case cost;
3. quality–cost and quality–latency Pareto frontiers;
4. incremental cost per additional correct answer versus no search;
5. cost of unsuccessful searches and cost of five-call-cap failures;
6. open-model-plus-search versus frontier-model-without-search, the registered
   capability-substitution question;
7. a budgeted router simulation: expected accuracy, latency, and cost when
   search is invoked only for newer and/or composed tasks.

The last two are more distinctive than a generic cheapest-provider chart.

## What this study answers that the public pages do not

Use “does not identify in its current design,” not “never can.” The distinction
matters.

| Question | This study | Why the cited public analysis does not identify it |
|---|---|---|
| Does the same retrieval system benefit weaker and stronger models differently? | Yes: model-by-retrieval difference-in-differences across four models | Artificial Analysis fixes one answer model; Vals varies models but its public headline is tool/domain effects rather than a no-search retrieval interaction |
| Does retrieval close the open/frontier capability gap? | Yes, on these paired news tasks | Artificial Analysis has no model variation; Vals has no public no-search arm for this comparison |
| How does the value of retrieval change with event age? | Yes: event-date slopes for no-search accuracy and retrieval gain | Neither public page reports a within-task recency/value curve |
| Is more retrieved context causally useful? | Partly: normalized vs. wide is an assigned result-volume contrast | Neither page publishes a within-provider result-count ablation tied to evidence quality |
| Why can more context fail to improve accuracy? | Evidence coverage rises slightly while precision, diversity/distinctness measures deteriorate | The public pages report outcomes and operational behavior, not this retrieval-surface mechanism |
| Is the remaining bottleneck retrieval or composition? | The adjusted composed-question penalty remains after retrieval | Vals shows category convergence on computation-heavy tasks, but this study directly links event recency, retrieval gain, and composition in one task-level model |
| When is an in-progress search trajectory likely stuck? | Search-count and latency escalation predict sharply lower success | Both pages report calls/time; this study converts them into a strategy-switch diagnostic under a fixed five-call budget |
| How sensitive is the conclusion to the outcome definition? | Deterministic match and semantic judge disagree massively and asymmetrically | Neither page foregrounds scorer disagreement as a first-order dependent-variable result |
| Are benchmark labels contributing to the apparent ceiling? | All-model failures and cross-model alternative-answer agreement create a targeted adjudication queue | Neither public page presents this failure/label-audit decomposition |
| Does highlighting itself cause improvement? | Not yet—and the report says why | This study records the missing counterfactual and specifies the exact ablation needed; the other pages do not answer it either |

## The strongest defensible “only we answer this” narrative

The unique story is a chain of decisions, not one score:

1. **Need:** newer events make internal knowledge unreliable, especially for
   some models.
2. **Lift:** normalized retrieval improves every model, but by very different
   amounts.
3. **Surface:** widening the payload increases coverage while diluting signal,
   with no strict-match benefit.
4. **Use:** even visible answer-bearing evidence is sometimes ignored or used
   incorrectly.
5. **Reason:** composed questions remain hard after retrieval because search
   solves knowledge access, not arithmetic or scope reconciliation.
6. **Control:** repeated searches and rising latency are an observable signal to
   switch strategy rather than continue the same loop.
7. **Measurement:** the apparent absolute quality depends strongly on scorer
   definition and label quality.

Artificial Analysis can tell a buyer which API leads its fixed-model
leaderboard. Vals can tell a buyer whether native or Exa is more efficient for
finance or legal work. This study can tell an agent designer **whether to
search, how much evidence to surface, when to change strategy, and which part of
the answer pipeline failed**.

## Recommended publication sequence

1. Three key findings.
2. Retrieval lift by model, with paired intervals and win/tie/loss counts.
3. Retrieval gain by event age.
4. Normalized vs. wide: accuracy beside evidence-quality deltas.
5. Composition penalty after retrieval.
6. Failure funnel and concrete examples.
7. Search-escalation tail risk.
8. Native vs. normalized system comparison.
9. Scorer disagreement and label-audit ceiling.
10. Cost/Pareto section once trace-complete costs are available.

This order moves from the buyer's first question (“does search help?”) to the
agent designer's harder questions (“when, why, and what should the system do
next?”).

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
