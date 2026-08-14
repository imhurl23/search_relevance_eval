# Search relevance evaluation

This repository measures how web retrieval changes short-form factual answers.
It compares four models under no search, a shared You.com tool, and each frontier
vendor's native search.

The study has 14 conditions. Each condition uses the same pinned dataset rows,
answer format, five-search allowance, and scoring code. The preregistered
contrasts and reporting rules live in [docs/study-design.md](docs/study-design.md).

## What is included

- Four model configurations: two Baseten-hosted open models, OpenAI, and
  Anthropic.
- Two You.com retrieval setups that vary result count.
- Native search for OpenAI and Anthropic.
- A no-search control for every model.
- Importers for LiveNewsBench and RetrievalQA.
- A local curation and publication pipeline for Corvus-QA.
- Eleven deterministic scorers and one or more semantic judges.
- Row-level latency, token use, search spend, model spend, and integrity flags.

The harness supports one external search API: You.com. Results therefore support
claims about You.com, not independent search APIs as a class.

## Repository map

| Path | Purpose |
|---|---|
| `run_eval.py` | Run one experiment condition or a gateway check |
| `run_matrix.py` | Preview or launch the finalized matrix |
| `agents.py` | Provider clients, search adapters, prompts, and pricing |
| `scorers.py` | Deterministic and judge-based row scorers |
| `analyze_results.py` | Paired summaries for exported JSONL results |
| `import_livenewsbench.py` | Import a pinned LiveNewsBench revision |
| `import_retrievalqa.py` | Import a pinned RetrievalQA file |
| `corvus/` | Build, validate, and import Corvus-QA |
| `config/corvus/` | Source policies and approval templates |
| `docs/corvus-compliance.md` | Source and publication requirements |
| `FORBIDDEN.md` | Prose patterns rejected during review |
| `tests/` | Offline tests and provider-response fixtures |

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Put credentials in `.env`. The loader always overrides or clears matching
ambient variables, so the file is the source of truth for a run.

A harness run needs `YDC_API_KEY`, the selected model vendor's key, and
Braintrust credentials. A native run does not need `YDC_API_KEY`.

Run the offline suite:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## Run an evaluation

Use a pinned Braintrust dataset version for every recorded comparison:

```bash
DATASET_VERSION=replace-with-snapshot-xact-id

.venv/bin/python run_eval.py run \
  --dataset-name RetrievalQA \
  --dataset-version "$DATASET_VERSION" \
  --study-id retrievalqa-pilot \
  --model-vendor openai \
  --search-mode harness \
  --arm normalized \
  --trials 1 \
  --limit 5
```

Remove `--limit` only after the pilot passes. `--allow-latest` exists for local
exploration; it should not appear in a reported study.

Control, harness, and native examples:

```bash
.venv/bin/python run_eval.py run \
  --dataset-version "$DATASET_VERSION" --study-id retrieval-study \
  --model-vendor openai --search-mode none

.venv/bin/python run_eval.py run \
  --dataset-version "$DATASET_VERSION" --study-id retrieval-study \
  --model-vendor openai --search-mode harness --arm normalized

.venv/bin/python run_eval.py run \
  --dataset-version "$DATASET_VERSION" --study-id retrieval-study \
  --model-vendor openai --search-mode native
```

Repeat `--judge` to use a judge jury. A non-OpenAI route uses
`model@base_url` and `JUDGE_API_KEY`. The default judge is `gpt-5.6-luna`.
Pass `--judge gpt-4.1` for direct parity with LiveNewsBench's published judge.

## Experimental matrix

### Models

| Class | Vendor | Pinned model | Input/output $ per MTok | Native search |
|---|---|---|---:|---|
| Open | Baseten | `deepseek-ai/DeepSeek-V4-Flash-0731` | 0.13 / 0.26 | No |
| Open | Baseten | `zai-org/GLM-5.2` | 1.40 / 4.40 | No |
| Frontier | OpenAI | `gpt-5.6-terra` | 2.00 / 12.00 | Yes |
| Frontier | Anthropic | `claude-opus-5` | 5.00 / 25.00 | Yes |

Prices are pinned list prices in `agents.MODEL_USD_PER_MTOK`. Recheck them
before publication. Promotional and negotiated rates are excluded.

### Search configurations

| Mode | Arm | Retrieval parameters | Search price |
|---|---|---|---:|
| `none` | — | No search tool | $0 |
| `harness` | `normalized` | You.com, 8 results, no freshness filter | $0.005/call |
| `harness` | `wide` | You.com, 20 results, no freshness filter | $0.005/call |
| `native` | — | Vendor-hosted search | $0.010/search |

Harness results expose at most 400 snippet characters. The agent may make five
searches and cannot fetch arbitrary pages. You.com charges per call, so the
`wide` arm costs the same per search as the 8-result arms.

The two Baseten models each run three conditions. The frontier models each run
four. The total is 14:

| Model | None | Normalized | Wide | Native |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | Yes | Yes | Yes | — |
| GLM-5.2 | Yes | Yes | Yes | — |
| gpt-5.6-terra | Yes | Yes | Yes | Yes |
| claude-opus-5 | Yes | Yes | Yes | Yes |

The launcher interleaves models by treatment and runs one condition at a time to
avoid provider overload. Run the matrix without long pauses because the live web
changes during a study.

Preview the full launch without spending money:

```bash
.venv/bin/python run_matrix.py \
  --dataset-name LiveNewsBench \
  --dataset-version 1000197598003916222 \
  --study-id livenewsbench-full-v1
```

After checking all 14 commands, repeat with `--execute`. The launcher defaults
to one trial. Add `--limit 5` for an end-to-end pilot.

## Gateway routing

Model calls go directly to vendors unless `BRAINTRUST_GATEWAY_URL` is set in
`.env`. Gateway routing moves agents and judges together. Each row records the
serving path and effective base URL.

```dotenv
BRAINTRUST_GATEWAY_URL=https://gateway.braintrust.dev
BRAINTRUST_GATEWAY_PROJECT=automations-spend-control
BRAINTRUST_GATEWAY_ORG=
BRAINTRUST_GATEWAY_API_KEY=
```

Verify each vendor/model pair after changing gateway or provider settings. A
gateway check makes a billed native-search call where native search exists.

```bash
.venv/bin/python run_eval.py gateway-check --model-vendor openai
.venv/bin/python run_eval.py gateway-check --model-vendor anthropic
.venv/bin/python run_eval.py gateway-check --model-vendor baseten \
  --agent-model zai-org/GLM-5.2
```

OpenAI and Anthropic gateway checks passed on August 10, 2026, including the
native-search response blocks parsed by the adapters. The current Baseten model
paths still need a live gateway check.

## Datasets

### Upstream imports

Import upstream data from immutable revisions:

```bash
DATASETS_ROOT=/path/to/LiveNewsBench/datasets
SOURCE_FILE=/path/to/retrievalqa.json

.venv/bin/python import_livenewsbench.py "$DATASETS_ROOT" \
  --source-commit replace-with-git-sha

.venv/bin/python import_retrievalqa.py "$SOURCE_FILE" \
  --source-revision replace-with-revision \
  --source-sha256 replace-with-sha256
```

The importers publish Braintrust datasets. `run_eval.py` reads an existing
dataset and requires its version unless `--allow-latest` is present.

Every row follows this contract:

```text
input    = {"question": string}
expected = string or list of accepted strings
metadata = upstream fields plus importer provenance
```

LiveNewsBench provides rolling news questions and source metadata. RetrievalQA
contains both static questions and historical time-sensitive questions. Corvus-QA
contains curated fact transitions with recency and coverage labels.

### RetrievalQA reference dates

FreshQA and RealTimeQA labels describe facts at an earlier date. The runner adds
that date to the effective question and tells the model to include it in search
queries.

- FreshQA uses its frozen evidence date: February 1, 2024.
- RealTimeQA derives the date from the upstream question ID.
- You.com day and week filters become absolute historical ranges ending on the
  row's reference date.
- Static RetrievalQA rows remain unchanged.

Rows record `answer_as_of`, its derivation, the effective question, and the
resolved search parameters. This prevents a present-day answer from being scored
against a historical label.

### Corvus-QA

Corvus-QA turns corroborated fact changes into benchmark rows. Its pipeline has
three artifacts:

| Stage | Input | Output |
|---|---|---|
| Collect and curate | Bounded source records | One `FactEvent` per observation |
| Freeze | Curated `FactEvent` JSONL | Rows, rejection ledger, and SHA-256 manifest |
| Publish | Approved frozen split | Versioned `Corvus-QA-dev` or `Corvus-QA-test` snapshot |

Use a separate directory for each freeze:

```text
curation/<freeze-id>/
  01-sources/
  02-events/
  03-freeze/
```

Build and import a split:

```bash
FREEZE_ID=replace-with-freeze-id
AS_OF=2026-08-11T17:00:00-07:00

.venv/bin/python -m corvus.cli.build_dataset \
  "curation/$FREEZE_ID/02-events/events-dev.jsonl" \
  "curation/$FREEZE_ID/03-freeze/corvus-dev.jsonl" \
  --split dev --freeze-id "$FREEZE_ID" --as-of "$AS_OF"

.venv/bin/python -m corvus.cli.import_dataset \
  "curation/$FREEZE_ID/03-freeze/corvus-dev.jsonl" \
  --manifest "curation/$FREEZE_ID/03-freeze/corvus-dev.jsonl.manifest.json" \
  --split dev --compliance-approval freeze-approval.json
```

The builder accepts `FactEvent` JSONL only. A row qualifies by default when two
resolvers and two attesters agree on the entity, attribute, new value, effective
time, and previous value. `resolver_id`, `attester_id`, and `authority_family`
remain separate because they describe different forms of independence.

Human approval binds the frozen artifact and source-policy hashes. Start from
`config/corvus/compliance_approval.example.json`. The code does not create the
approval.

Run the compliance checks before collection or publication:

```bash
.venv/bin/python -m corvus.cli.smoke_test
.venv/bin/python -m corvus.cli.check_compliance
```

[docs/corvus-compliance.md](docs/corvus-compliance.md) contains the allowed
sources, attestations, rate limits, retention rules, Section 16 workflow, and
news/sports requirements.

## Measurement

`scorers.py` emits one score per row. `analyze_results.py` computes effects and
uncertainty across rows.

The deterministic scorers cover:

- exact or contained answer match;
- answer match after hard-rule gating;
- source leakage and search-budget compliance;
- temporal grounding;
- answer presence and precision in snippets;
- token cost to surface the answer;
- snippet redundancy and domain diversity.

The semantic judge grades answer correctness. Report deterministic and judge
scores together because paraphrases can disagree with string matching.

Search modes expose different evidence:

| Surface | Available fields | Conditions | Unavailable scores |
|---|---|---|---|
| `full` | Rank, URL, title, snippet, date | Harness | None |
| `no_snippet` | Rank, URL, title, date | Anthropic native | Snippet-derived scores |
| `urls_only` | Rank, URL, title | OpenAI native | Snippet-derived scores and temporal grounding |
| `none` | No search evidence | No search | All trajectory scores |

An unavailable score is `None` and stays out of averages. Converting it to zero
would punish an unobservable field; converting it to one would reward missing
evidence.

Each row also records conditions that can invalidate or qualify an estimate:

- `zero_search_row`: a search tool was available but unused.
- `search_degraded` and `search_fully_failed`: some or all searches failed.
- `model_refused` and `answer_truncated`: the model did not produce a usable
  answer.
- `search_budget_enforced`: the API itself enforces the five-search cap.
- `exclusion_enforced`: gold-source domains were blocked.
- `sampling_pinned` and `reasoning_effort_pinned`: the provider accepted the
  requested controls.
- `prompt_version`, `serving_path`, and `decision_surface`: the execution path
  needed to interpret the row.

Analyze an exported JSONL file with paired task bootstrapping:

```bash
.venv/bin/python analyze_results.py experiments.jsonl \
  --score qa_answer_match \
  --study-id retrieval-study \
  --baseline 'gpt-5.6-terra::no_search'
```

The analyzer reports paired effects, a 95% task-bootstrap interval,
category-balanced effects, win/tie/loss counts, search counts, answer length,
and a cost frontier when `total_cost_usd` is present.

## Limits that affect interpretation

- Native and harness prompts differ by one tool-specific sentence. The runner
  records the prompt version; no prompt-only control arm exists.
- OpenAI native search exposes no `max_uses`, so its five-search budget is
  observed but not API-enforced.
- You.com and Anthropic expose last-modified dates. OpenAI native exposes no
  result dates. Freshness claims should use Corvus-QA `recency_rung`; treat
  `temporal_grounding` as a last-modified measure.
- Native search exposes less evidence than the harness. Compare trajectory
  metrics only where the declared decision surface supports them.
- A default OpenAI judge shares a vendor with the OpenAI agent. Use multiple
  vendors for reported frontier comparisons.
- Three trials have no power justification. Measure run-to-run variation before
  interpreting a small effect.
- The runner flags unused and failed search but does not impose an automatic
  exclusion rule. Apply the rules in the study design and report counts by arm.
- Model cost uses list prices and token accounting, not invoices. Native-search
  context can include an undisclosed number of billable tokens.
- The design covers short factual QA. It does not support conclusions about
  legal, financial, or other long-form research workflows.
- The web changes during the matrix. Manual interleaving remains required.

Keep documentation edits within the rules in [FORBIDDEN.md](FORBIDDEN.md).
