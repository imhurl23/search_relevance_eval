# Starter framework for web-search API × LLM freshness experiments

This repository is a **starter framework for designing controlled experiments
that measure how a web-search API affects an LLM's ability to answer fresh,
time-sensitive questions**. It is not a finished benchmark, a provider ranking,
or a claim that the current defaults are the right experimental design.

The included implementation gives you a concrete starting point: one frozen
web-research agent answers news questions while the search provider and search
configuration are varied. The model snapshot, prompt, tool schemas,
search/click budget, result normalization, and page fetcher are held constant so
differences can be attributed as closely as possible to the search layer.

Use the framework to define and test your own experiment:

1. Choose the freshness question you want to answer—for example, whether live
   crawling improves accuracy on events from the last week.
2. Select versioned datasets and pin the exact snapshot used by every
   experimental condition.
3. Define comparable provider arms, including retrieval tier, date filters,
   snippet budget, result count, and cost.
4. Freeze the agent model, prompt, tool contract, and interaction budget.
5. Run provider conditions close together in time, with repeated trials, so the
   changing web does not become an uncontrolled variable.
6. Compare answer quality alongside leakage, temporal grounding, retrieval
   diversity, latency, and cost—not accuracy alone.

The current example compares LiveNewsBench and RetrievalQA across Exa,
Parallel, and You.com. The **Remaining gaps** and **Open decisions** sections
below identify choices that must be resolved before treating results as
publication-quality.

## Included example configuration

- Agent: `gpt-4o-2024-11-20` (override with `--agent-model`), temp 0, seed 42,
  5 searches + 5 clicks
- Providers: `exa`, `parallel`, `youdotcom`
- Arms: `normalized` (uniform snippet budget), `native_fresh` (each vendor's own
  freshness knobs), `no_search` (control — no tools, parametric memory only)
- Datasets: `LiveNewsBench`, `RetrievalQA` — Braintrust, versioned with named
  snapshots
- Scorers: 10 deterministic plus one judge score per run, single or jury. See
  [scorers.py](scorers.py)

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=... EXA_API_KEY=... PARALLEL_API_KEY=... YDC_API_KEY=...
```

`.env` supplies `BRAINTRUST_API_KEY` and `BRAINTRUST_PROJECT_ID` and always
overrides any ambient credential, per [AGENTS.md](AGENTS.md).

## Running

The importers push datasets, not the eval:

```bash
python import_livenewsbench.py <datasets_root> --source-commit <sha>
python import_retrievalqa.py <source_file> --source-revision <rev> --source-sha256 <sha>
```

Use one `study-id`, pinned dataset version, and agent model across every
condition in a comparison. Interleave conditions in time—running them days
apart makes the web itself a variable:

```bash
python run_eval.py run --provider exa --arm native_fresh \
    --study-id july-freshness --dataset-version <snapshot-xact-id> --trials 3
```

Run the control arm once per dataset version. Provider scores are not
interpretable without it — it tells you how many rows the model can already
answer with no search at all:

```bash
python run_eval.py run --arm no_search \
    --study-id july-freshness --dataset-version <snapshot-xact-id>
```

Repeat the full matrix with `--agent-model <pinned-model>` to test whether a
search-provider effect generalizes beyond one agent model. Unpinned datasets
are rejected unless `--allow-latest` is explicitly supplied for an exploratory
run.

To convene a cross-vendor judge jury instead of a single judge, repeat
`--judge`. Use `model@base_url` with `JUDGE_API_KEY` for non-OpenAI routes:

```bash
python run_eval.py run --provider exa --arm native_fresh \
    --judge gpt-4.1 --judge claude-sonnet-5@https://... --judge gemini-3-pro@https://...
```

### Dataset contract

Set by the importers, read by [scorers.py](scorers.py). Leakage and freshness
ground truth live in `metadata`, not `expected`:

```
input    = {"question": str}
expected = answer string, or a list of acceptable answers (RetrievalQA)
metadata = upstream row fields (link, articles[], event_date, ...)
           + livenewsbench_release / livenewsbench_split / source_commit
```

## Remaining gaps

1. **No native model-search baseline.** `no_search` isolates retrieval uplift,
   but does not answer whether an independent API beats the model provider's
   own search tool.
2. **No automated time blocking.** Conditions must still be manually
   interleaved. Publication runs should randomize or round-robin conditions at
   task level to reduce web-time confounding.
3. **No expert multi-step task domain.** Results do not generalize to legal,
   finance, or other professional research workflows.
4. **Bootstrap, not mixed effects.** The included analysis is dependency-free
   and task-paired; final reporting should also fit condition fixed effects
   with task random intercepts.
5. **Total cost requires trace aggregation.** Search fees and agent tokens are
   logged, but child LLM-span cost must be aggregated into `total_cost_usd`
   before a cost frontier is valid.

## Open decisions

The code picks a default for each of these. The defaults are assumptions, not
conclusions. Settle them before publishing.

### Blocking

**1. `native_fresh` means three different things.**

| Provider | What `native_fresh` changes | Freshness? |
|---|---|---|
| Exa | `livecrawl: "always"`, `maxAgeHours: 24` | Yes |
| You.com | `freshness: "week"` | Yes, but 7× wider than Exa |
| Parallel | `processor: "base"` → `"pro"` | No — a quality and cost tier |

The Parallel column measures pro against base, not fresh against cached, and its
`$0.004 → $0.009` cost difference comes along with it, which then affects any
cost/quality comparison. Options:

- Use `source_policy.after_date` for Parallel, computed from *now*, never from
  the row's `event_date`. Keep `processor: "base"` in both arms.
- Keep the processor switch and rename the arm `native_best`. Honest, but no
  freshness result for Parallel.
- Split into `native_fresh` and `native_tier`. Doubles run cost.

Even after that, Exa's 24 hours and You.com's week are different treatments.
Aligning them means `freshness: "day"` or `maxAgeHours: 168`.

**2. The providers search different corpora.** Exa filters to
`category: "news"`. Parallel has no category filter, so it searches the general
web. You.com reads `results.web[]` because `results.news[]` returns no snippets
at all. A provider can win or lose on corpus scope rather than retrieval quality.
There is no clean fix, so choose which asymmetry to accept: drop Exa's news
filter, keep it and document it, or limit the eval to what all three do the same
way.

### Important

**3. Snippet caps differ by arm.** `SNIPPET_CHARS = 400` applies to Exa and
You.com in both arms, but to Parallel only in `normalized`. Parallel's
`native_fresh` snippets can reach 1500 characters, which is a decision-surface
advantage unrelated to retrieval. It is also the reason the `normalized` arm
exists, since truncating Parallel's excerpts removes its main feature. Decide
whether `native_fresh` caps all three or none.

**4. Exa's search tier is not frozen.** `type` defaults to `"auto"`, which routes
per query. Now pinned as `EXA_SEARCH_TYPE = "auto"` — same behavior, but visible.
`"fast"` or `"deep"` would actually freeze it, at the cost of changing retrieval
quality.

**5. The judge and the agent share a vendor.** The agent is `gpt-4o` and
`--judge` defaults to `gpt-4.1`, which cannot rule out self-preference. Passing
`--judge` several times convenes a majority-vote jury across vendors instead;
every ballot and a `unanimous` flag land in row metadata. The default is still a
single judge, because that is what keeps parity with LiveNewsBench's published
numbers. Decide which one is authoritative for your headline — and note the
deployed QA Answer Correctness scorer uses `openai/gpt-oss-120b`, a third answer
to the same question. `qa_answer_match` needs no model at all, so its
disagreement with the judge is worth reporting rather than smoothing over.

**6. Cost numbers are unverified.** `SEARCH_COST` in [run_eval.py](run_eval.py)
is hand-entered. Check all six `(provider, arm)` prices against current pricing
pages. Exa's `native_fresh` now sends `livecrawl: "always"`, so confirm whether
that arm costs more per call. Parallel pro against base needs its own check, and
that cost difference is currently tangled up with the freshness treatment (see
1). `_tok()` estimates tokens as `chars / 4`; use tiktoken before publishing
token-normalized numbers, including `token_discounted_gain`.

**7. Domain exclusion works differently per vendor.** Exa takes an
`excludeDomains` array. Parallel's `source_policy.exclude_domains` covers
subdomains automatically and caps at 200 combined. You.com takes a
comma-separated string, mutually exclusive with `include_domains`. Apex-to-
subdomain coverage is not uniform, so `leakage_guard` can fire for API reasons
rather than retrieval behavior. Test it directly: one row per provider with the
gold domain excluded, confirm zero leaks. The exclude list is ordered source
domains first, so truncation drops archive mirrors rather than gold sources.

**8. Some rows have no leakage ground truth.** Newer LiveNewsBench rows omit
source URLs upstream. `leakage_guard` then checks archive domains only and
reports `source_domain_available: false`. Either restrict to rows that have
source URLs, or report the two subsets separately.

**9. Item pairing needs a pinned dataset.** The runner now requires
`--dataset-version` unless `--allow-latest` is explicitly supplied. Use the same
snapshot and `--study-id` for every condition in a comparison.

### Minor

**10. `trial_count = 3` is thin.** Enough to show retrieval is nondeterministic,
not enough for a good variance estimate. If run-to-run spread is part of the
argument, raise it. Uncertainty itself is computed in the analysis layer, not by
a scorer — see below.

**11. RetrievalQA measures a different construct.** It runs through the same
generic factual-question prompt and its list-valued answers are supported. It
lacks LiveNewsBench event dates and leakage rules, so do not pool its rows into
one "freshness" headline score. Report it as a separate domain.

## Scoring and analysis are separate layers

Scorers emit **one score per row** and nothing else. Standard errors, confidence
intervals, significance tests, macro-averages, and jury agreement rates are
properties of a *set* of rows, so they belong to analysis, downstream of the
experiment. Keeping them out of the scorers is what lets a scorer stay a pure
function of a single row — reproducible, cheap to recompute, and safe to deploy
to Braintrust.

The dependency-free [analyze_results.py](analyze_results.py) implements the
paired summary layer for JSONL exports:

```bash
python analyze_results.py experiments.jsonl \
    --score qa_answer_match \
    --study-id july-freshness \
    --baseline 'gpt-4o-2024-11-20::no_search'
```

It averages repeated trials within task and condition, then reports paired
effects, a 95% task-bootstrap interval, category-balanced effects, win/tie/loss
counts, mean and P95 search calls, answer length, and a cost Pareto frontier
when trace-level `total_cost_usd` is present.

The following analysis responsibilities still require care:

- **Subtract the control.** Report provider scores against the `no_search` arm,
  not in isolation. The difference is retrieval's marginal value.
- **Gate the headline.** Multiply the answer score by `dealbreaker_gate` so a row
  that leaked a gold source or blew the budget cannot count as a win.
- **Uncertainty.** The included confidence interval resamples paired tasks, not
  individual trials. For publication, also fit a mixed-effects model with
  condition as a fixed effect and task as a random intercept.
- **Macro-average over strata.** The analyzer category-balances the effect using
  `benchmark_category`; for LiveNewsBench, also consider a purpose-built
  freshness bucket derived without leaking event dates into retrieval.
- **Decompose cost.** `search_cost_usd` is search spend only; model inference
  cost comes from wrapped LLM spans. Aggregate both into `total_cost_usd`.
  The analyzer deliberately refuses to treat search fees alone as total cost.
- **Jury reliability.** Agreement rate and per-judge bias, from the `ballots`
  metadata.
- **Runaway pressure.** `refused_searches` / `refused_clicks` count the tool
  calls the 5/5 cap turned away. A provider that constantly pushes past the
  budget would cost far more uncapped — a real failure mode that a hard cap
  otherwise hides completely.

## Deliberate choices

Not open questions. Listed so they do not get changed by accident.

- The page fetcher is provider-neutral. Using Exa `/contents` or You.com
  `livecrawl` in `run_fetch` would put the provider back into the fetch step.
  This limits the eval to search-layer quality.
- You.com `livecrawl` stays off in `native_fresh`. It only fills
  `result.contents`, which the agent never sees, so it would add latency and cost
  without changing what is measured.
- No adapter derives dates from `event_date`, and queries avoid temporal words.
  You.com uses the broader of query-implied and parameter freshness, so a dated
  query would un-blind the arms.
- Raw pre-normalization payloads go to each tool span's metadata verbatim. The
  decision-surface analysis needs them in the traces.
- `notrace_io=True` on both tool spans. Otherwise `@traced` logs the return tuple
  over the explicit `output=`.
- A single gold answer, not a multi-item rubric. Rubric grading suits open-ended
  professional tasks; LiveNewsBench and RetrievalQA are short-form factoid QA
  where the gold answer is one fact, so the SimpleQA three-way grade fits and a
  rubric would invent structure the dataset does not have. The rubric idea worth
  borrowing is *gating*, which is what `dealbreaker_gate` does.

> **Relationship to Vals:** This framework uses the same basic experimental
> principle as [Vals AI's Web Search Index](https://www.vals.ai/benchmarks/web_search):
> swap the search tool while holding the agent setup fixed. It adapts that idea
> to freshness-focused, short-form QA and adds controls for public-dataset
> contamination and source leakage.
