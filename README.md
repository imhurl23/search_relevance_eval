# Starter framework for web-search API × LLM freshness experiments

This repository is a **starter framework for designing controlled experiments
that measure how a web-search API affects an LLM's ability to answer fresh,
time-sensitive questions**. It is not a finished benchmark, a provider ranking,
or a claim that the current defaults are the right experimental design.

The included implementation gives you a concrete starting point: a web-research
agent answers news questions across two treatment axes — the **model class**
(open-weights vs. frontier) and the **search mode** (no search, a harness-owned
search tool over a search API, or the model vendor's own server-side search). The
prompt, tool contract, search budget, and result normalization are held constant
within a condition so differences can be attributed as closely as possible to the
axis under test. See [The test matrix](#the-test-matrix) for the 22 runs and for which
comparisons the instrumentation actually supports.

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

The current example runs LiveNewsBench and Corvus-QA against four You.com
setups, two frontier native-search arms, and a no-search parametric floor. The **Remaining gaps** and **Open decisions**
sections below identify choices that must be resolved before treating results
as publication-quality.

## Repository layout

- `agents.py` — per-vendor agent clients, native-search adapters, and the
  decision-surface tiers the scorers gate on
- `corvus/` — Corvus-QA schemas, source adapters, builders, and CLI modules
- `config/corvus/` — machine-readable source policies and approval templates
- `docs/` — compliance and operational documentation
- `tests/` — offline unit and smoke-test fixtures
- root scripts — the original cross-provider evaluation and import workflow

## Included example configuration

- Agent: per-vendor pinned model (override with `--agent-model`), 5 searches and
  no arbitrary webpage fetching
- Search API (`--search-mode harness`): `youdotcom` only
- You.com setups (`--arm`, harness only): `normalized`, `native_fresh` (day),
  `fresh_week`, `wide` (20 results)
- Datasets: `LiveNewsBench`, `RetrievalQA` — Braintrust, versioned with named
  snapshots; `Corvus-QA-dev` and `Corvus-QA-test` can be built and imported
  separately
- Scorers: 11 deterministic plus one judge score per run, single or jury. See
  [scorers.py](scorers.py). `gated_answer_match` is the composed headline number:
  answer correctness with hard-rule violations zeroed
- Cost: `search_cost_usd`, `model_cost_usd`, and `total_cost_usd` per row, kept
  decomposable because inference dominates the bill (see Design premises).
  Cached input is priced separately at each vendor's published multiplier —
  OpenAI caches automatically on every turn after the first, so charging its
  cached tokens at the base rate overstated the harness arm's cost by ~9x on a
  cached call while barely touching the single-call native arm. `agent_cache_hit_rate`
  is logged per row so that correction stays checkable

## What this study claims

The design is committed in [docs/study-design.md](docs/study-design.md) before the
runs, so the 23 conditions test stated hypotheses rather than producing a table
that gets interpreted afterwards. Three claims, each requiring evidence a
search-provider ranking cannot produce:

| claim | question | needs |
|---|---|---|
| **A** marginal value | how much of accuracy is retrieval vs. what the model knew | `none` floor on every model |
| **B** substitution *(candidate headline)* | can OSS + retrieval match frontier without it, at what cost ratio | OSS arm + both sides priced |
| **C** mechanism | when one config wins, *through what* | the six mediator metrics |

Claim B is the sharpest: `openai/gpt-oss-120b` is **100× cheaper per input token**
than `claude-fable-5` ($0.10 vs $10.00), and retrieval costs $0.005–$0.015 per
search — so "how much model can you trade for how much retrieval" has a concrete
answer with budget consequences.

Plus a methods contribution independent of all three: **vendor-native search is
not auditable the way an API-backed pipeline is**, and `decision_surface` measures
exactly how much. If a vendor's search cannot be inspected for source leakage,
evidence sufficiency, or freshness, results obtained on it are not verifiable in
the way API-based results are.

## Design premises

Four premises the instrumentation is built on. Each is stated here because it
determined a logging or reporting decision, and each is checkable on this data
rather than taken on faith.

1. **Search spend is the small half of the bill.** Five native searches cost
   $0.05; one search-heavy turn pulling 60k input tokens through a $10/MTok model
   costs about $0.60. So a search-fee-only comparison measures the wrong quantity.
   `model_cost_usd`, `total_cost_usd`, and `search_share_of_cost` are logged per
   row, and that last field is what makes this premise falsifiable here. The
   native arms are worst affected: their search-result tokens are billed as input
   tokens on the model call, not as a search fee, so without this the arm with the
   highest hidden cost reports the lowest.
2. **Latency is not derivable from search count.** A layer issuing more but
   faster calls can finish ahead of one issuing fewer slow ones, so a row-level
   `latency_s` is logged alongside the per-search tool-span latency.
3. **A retrieval effect measured in one domain is a single-domain result.** It can
   be substantial in one and absent in another, so results are reported per
   `benchmark_category` and pooling across categories is treated as a reporting
   error (see below). Two domains are run for the same reason.
4. **A rule violation cannot be averaged away.** A row that leaked a gold source
   is not 90% valid, which is why `gated_answer_match` zeroes it rather than
   discounting it.

## The test matrix

Two axes: the **model** and the **search config**. You.com is the only search
API, which makes the *setup* the treatment rather than the provider.

**Models** (`--model-vendor`, `--agent-model`)

| class | vendor | model | $/MTok in-out | native search |
|---|---|---|---|---|
| oss | `baseten` | `deepseek-ai/DeepSeek-V4-Flash-0731` | $0.13 / $0.26 | ⛔ |
| oss | `baseten` | `zai-org/GLM-5.2` | $1.40 / $4.40 | ⛔ |
| frontier | `openai` | `gpt-5.6-sol` | $5 / $30 | ✅ Responses `web_search` |
| frontier | `anthropic` | `claude-fable-5` | $10 / $50 | ✅ `web_search_20250305` |

Two OSS rows, not one: a single open model cannot distinguish "retrieval
substitutes for capability" from "this one model happens to be good at tool
calling." They span the cost range on purpose — the Flash row is the extreme
substitution point (38× below the cheapest frontier row), GLM-5.2 is the
strong-but-cheaper point (3.6×). Baseten runs no search of its own, so its
harness arms call You.com directly; the model only has to emit the tool call.

**Search configs** (`--search-mode`, `--arm`)

| # | mode | arm | You.com params | $/call |
|---|---|---|---|---|
| 1 | `none` | — | no tools, parametric memory | $0 |
| 2 | `harness` | `normalized` | count=8, no freshness filter | $0.005 |
| 3 | `harness` | `native_fresh` | count=8, `freshness=day` | $0.005 |
| 4 | `harness` | `fresh_week` | count=8, `freshness=week` | $0.005 |
| 5 | `harness` | `wide` | count=20, no freshness filter | $0.005 |
| 6 | `native` | — | the model vendor's own server-side search | $0.010 |

Snippet budget (400 chars), search budget (5), and the gold-source exclusion list
are held constant across every setup, so a difference between setups is
attributable to the parameters that differ.

`wide` is free: You.com bills per call **independent of `count`**, so 20 results
cost the same as 8. If it wins, that is recall the 8-result default was leaving
on the table at no extra cost. All four harness setups cost identically, which
isolates the cost comparison to the native arms and the model tokens.

**22 runs**

| | 1 none | 2 normalized | 3 fresh_day | 4 fresh_week | 5 wide | 6 native |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔ |
| GLM-5.2 | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔ |
| gpt-5.6-sol | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| claude-fable-5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

```bash
V=<xact-id>; S=matrix-v1; T=3
run() { python run_eval.py run --dataset-version $V --study-id $S --trials $T "$@"; }

for M in deepseek-ai/DeepSeek-V4-Flash-0731 zai-org/GLM-5.2; do
  run --model-vendor baseten --agent-model $M --search-mode none
  for ARM in normalized native_fresh fresh_week wide; do
    run --model-vendor baseten --agent-model $M --search-mode harness --arm $ARM
  done
done
for MV in openai anthropic; do
  run --model-vendor $MV --search-mode none
  for ARM in normalized native_fresh fresh_week wide; do
    run --model-vendor $MV --search-mode harness --arm $ARM
  done
  run --model-vendor $MV --search-mode native
done
```

Model IDs are pinned snapshots, not aliases: `gpt-5.6` is an alias for
`gpt-5.6-sol` and will move.

**The two frontier models are not claimed to be equivalent.** There is no
vendor-neutral capability tier. Matching on price pairs `gpt-5.6-sol` with
`claude-opus-5` ($5/$30 vs $5/$25); matching on within-lineup position pairs sol
with `claude-fable-5`, Anthropic's flagship ($10/$50). The two framings disagree,
so the pairing is a declared choice, not a measurement. `claude-fable-5` is the
default because flagship-vs-flagship is at least a definition both vendors
publish, whereas a price match is an artifact of their margin decisions. It
requires the org not be on zero data retention (every request 400s otherwise) and
declines more often, which `model_refused` records.

No primary contrast depends on that pairing — `native` vs `harness` and search vs
`none` both hold the model fixed within a vendor, and cross-vendor native
comparisons are ruled out. What it bounds is how far an oss-vs-frontier result
generalizes: read that contrast as "vs *these* frontier models", not "vs frontier
models".

### One API means one narrower claim

With Exa and Parallel removed, nothing here can separate **"You.com beats native
search"** from **"independent APIs beat native search."** The finding is about
You.com specifically. Do not let it be written up as the general claim.

### What is not comparable across arms, and why

Server-side search does not expose the decision surface the harness tool does.
Each row carries a `decision_surface` tier, and the scorers gate on it:

| tier | fields present | arms | scorers that go N/A |
|---|---|---|---|
| `full` | rank, url, title, snippet, published_date | harness | — |
| `no_snippet` | rank, url, title, published_date | Anthropic native | the 4 snippet-derived scorers |
| `urls_only` | rank, url, title (cited results only) | OpenAI native | snippet scorers **and** `temporal_grounding` |
| `none` | nothing | `--search-mode none` | all trajectory scorers, including `leakage_guard` |

A gated scorer returns `None` (excluded from averages), never a score. This
matters because the natural failure is silent, not loud: with an empty trajectory
`leakage_guard` returns 1.0 because it saw no URLs to leak, `budget_economy`
returns 1.0 because it saw no searches, `dealbreaker_gate` passes with zero rules
evaluated, and `search_cost_usd` is $0.00. A native arm would top compliance and
cost by construction. Cross-arm comparisons are therefore valid on the judge
score, `qa_answer_match`, cost, and latency — and on the decision-surface metrics
only within a tier.

Two things do hold across all search arms: gold-source exclusion is enforced
everywhere (harness `excludeDomains`, Anthropic `blocked_domains`, OpenAI
`filters.blocked_domains`, recorded as `exclusion_enforced`), and both native
arms bill at the same published $10/1k searches.

Two things that look comparable but are not: the *token* cost of native search
differs sharply from the harness arms even though the per-search price matches
(Anthropic basic web search loads every result into context; OpenAI's
`search_context_size=medium` loads an undisclosed amount; the harness arms load
exactly 8 × 400 chars), and the *quantity of evidence* per search is therefore
not held constant between native and harness. Cost comparisons should be read as
search-call spend plus separately-reported token spend, never as one number.

### Confounds the runner records rather than hides

- **`zero_search_row`** — the tool was available and the model never used it, so
  the row is a no-search row wearing a search arm's label. Weak OSS tool-calling
  and a native model that declines to search both land here. Filter on it.
- **`search_degraded` / `search_fully_failed`** — the model did search, but the
  search layer errored. A search API failure degrades the row rather than killing
  it, because rows do not fail at random: a provider fails hardest on the queries
  it handles worst, so dropping failed rows would score the run over a favorable
  subset of the questions actually asked. The row still gets answered and judged,
  so without these flags a provider outage reads as the model getting worse.
  `search_fully_failed` is the strong case — every attempted search failed, so the
  answer is parametric and the row belongs with `zero_search_row`. Per-error
  detail is in `search_errors`; the count is `n_search_errors`.
- **`sampling_params` / `sampling_pinned`** — temp 0 + seed 42 is *not holdable on
  either frontier vendor*. Claude Opus 5 rejects `temperature`/`top_p`/`top_k`
  with a 400, and gpt-5-family models reject `temperature` ("only the default (1)
  is supported") and offer no `seed`. Only the OSS arm pins sampling. The run
  records what each arm actually received instead of implying parity.
- **`reasoning_effort` / `reasoning_effort_pinned`** — pinned to `high` on
  Anthropic (effort and tools coexist on the Messages API), left at the vendor
  default on OpenAI (reasoning models reject `reasoning_effort` alongside function
  tools on chat completions, so pinning it on the native arm alone would make
  effort differ between OpenAI's own native and harness arms — the exact contrast
  under test).
- **`search_budget_enforced`** — the 5-search cap is API-enforced everywhere
  except the **OpenAI native arm**: its hosted `web_search` publishes no
  `max_uses`, so that one arm can exceed the budget every other arm is held to.
  `budget_economy` scores the observed count, which is what surfaces a violation,
  but the arm is not actually constrained. This is a live limitation on the
  native-vs-harness contrast within OpenAI.
- **`date_field_semantics`** — `temporal_grounding` reads `published_date`, but no
  search layer here reports a true publication date: You.com's `page_age` and
  Anthropic native's `page_age` are both last-modified, and OpenAI native has no
  date field. Semantics are uniform, so nothing gets pooled wrongly — but the
  construct measures "last touched," so prefer Corvus-QA's `recency_rung` for any
  freshness claim.
- **`prompt_version`** — the native arm's system prompt cannot describe a tool
  schema it does not own. Budget and answer format are held identical; the tool
  sentence differs, and that difference is declared.
- **`model_refused`, `answer_truncated`, `search_errors`, `pause_turns`** — a
  policy refusal or a truncated answer would otherwise score as a wrong answer
  and be misattributed to retrieval.

## Setup

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env`; every supported credential is loaded from that file and overrides
or clears the corresponding ambient credential. Each arm needs only its own
vendor's key — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `BASETEN_API_KEY` —
plus a search-API key when `--search-mode harness`.

### Serving path: direct, or through the Braintrust gateway

Model calls go direct to each vendor by default. Setting
`BRAINTRUST_GATEWAY_URL` routes them through the Braintrust gateway instead:

```bash
BRAINTRUST_GATEWAY_URL=https://gateway.braintrust.dev
BRAINTRUST_GATEWAY_PROJECT=automations-spend-control   # x-bt-project-name
BRAINTRUST_GATEWAY_ORG=                                # x-bt-org-name, optional
BRAINTRUST_GATEWAY_API_KEY=                            # defaults to BRAINTRUST_API_KEY
```

Under gateway routing the three vendor keys go unused: the vendor credentials
come from that Braintrust org's **Settings → AI Providers**, and a model with no
provider configured there returns 404 at the gateway without ever reaching a
vendor. The switch is all-or-nothing on purpose. Agents *and* judges move
together, and there is no per-vendor override, because a matrix whose arms sit
behind different serving stacks has a second variable moving inside every
contrast it was built to measure. Each run records `serving_path` and the
effective `agent_base_url`, so a study that mixes the two is at least detectable
afterward — but the fix is to re-run it, not to join across the boundary.

Two things to verify before trusting gateway results:

- **Model names must resolve in the org's provider registry.** The OSS rows are
  the likely casualty: their names are Baseten catalog paths, and the gateway
  resolves a name to a provider rather than passing it through.
- **Hosted search tools must survive the proxy.** Both native arms depend on a
  server-side tool (OpenAI's Responses `web_search`, Anthropic's
  `web_search_20250305`) whose result blocks the adapters in `agents.py` parse.
  If the proxy drops them, the native arm records an empty trajectory. Decision-
  surface gating reports that as unobservable rather than as a passing score, so
  it will not silently inflate a leaderboard — but it does waste the run.

`gateway-check` tests both, per vendor/model pair. It spends one real billed
search on the native half, so run it when the gateway config changes, not per
experiment:

```bash
.venv/bin/python run_eval.py gateway-check --model-vendor openai
.venv/bin/python run_eval.py gateway-check --model-vendor anthropic
.venv/bin/python run_eval.py gateway-check --model-vendor baseten \
  --agent-model zai-org/GLM-5.2
```

The suite is stdlib `unittest` and needs no test runner beyond the
dependencies above. It makes no network requests:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## Dataset curation pipeline

There are two dataset families in this repository, and they use different
curation paths:

- **Upstream benchmarks** (`LiveNewsBench`, `RetrievalQA`) are imported from a
  pinned upstream revision. Their importers push datasets; `run_eval.py` does
  not.
- **Corvus-QA** is curated locally from bounded source observations,
  deterministic eligibility rules, and an artifact-bound publication approval.

Import the upstream benchmarks with:

```bash
python import_livenewsbench.py <datasets_root> --source-commit <sha>
python import_retrievalqa.py <source_file> --source-revision <rev> --source-sha256 <sha>
```

### Canonical Corvus-QA flow

Corvus-QA has three stages. Keep their artifacts separate; source-specific
candidates and normalized `FactEvent` observations are not interchangeable.

| Stage | Command or owner | Input | Output |
|---|---|---|---|
| 1. Collect and curate | `corvus.cli.collect_sources` plus source-specific curation | bounded source query | normalized `FactEvent` JSONL, one observation per attester/resolver |
| 2. Freeze | `corvus.cli.build_dataset` | curated `FactEvent` JSONL | final `CorvusRow` JSONL, rejection ledger, and manifest |
| 3. Publish | approval + `corvus.cli.import_dataset` | one frozen split and manifest | versioned `Corvus-QA-dev` or `Corvus-QA-test` snapshot |

Some collectors already emit `FactEvent` observations. Others emit candidates
because the source does not contain enough structured information to infer an
answer safely. Curating those candidates into `FactEvent` JSONL is explicit
source-specific work: resolve the canonical entity and value, set the effective
time, identify the resolver/attester/authority, and retain provenance. Curation
happens directly in the normalized event artifact.

Use this directory pattern for a freeze so provenance remains auditable:

```text
curation/<freeze-id>/
  01-sources/          # immutable candidates, observations, collection manifests
  02-events/           # curated, normalized FactEvent JSONL
  03-freeze/           # CorvusRow JSONL, rejections, manifest, approval
```

The abbreviated command flow is:

```bash
# 1. Collect a bounded source slice. This Wikidata command emits FactEvents.
python -m corvus.cli.collect_sources wikidata-latest \
    --qid Q123 --property P169 --attribute ceo_of \
    --canonical-entity-id CIK0000123456 --entity-type company \
    --effective-ts 2026-07-28T09:00:00Z \
    --output curation/<freeze-id>/01-sources/wikidata-events.jsonl

# Curate any candidate-only sources into 02-events/events-dev.jsonl, preserving
# one normalized observation per resolver and attester.

# 2. Apply eligibility rules and freeze one split.
python -m corvus.cli.build_dataset \
    curation/<freeze-id>/02-events/events-dev.jsonl \
    curation/<freeze-id>/03-freeze/corvus-dev.jsonl \
    --split dev --freeze-id <freeze-id> --as-of <timestamp-with-timezone>

# 3. After inspecting all three build outputs and approving the artifact,
#    publish the split and create its immutable Braintrust snapshot.
python -m corvus.cli.import_dataset \
    curation/<freeze-id>/03-freeze/corvus-dev.jsonl \
    --manifest curation/<freeze-id>/03-freeze/corvus-dev.jsonl.manifest.json \
    --split dev --compliance-approval <freeze-approval.json>
```

Before publication, start from
[`config/corvus/compliance_approval.example.json`](config/corvus/compliance_approval.example.json).
The approval must bind the frozen artifact and source-policy hashes. The
repository never generates this human approval automatically.

### Corvus-QA artifact contract

Corvus-QA sources fact transitions rather than authored questions. Each source
adapter or source-specific curator must emit one JSONL `FactEvent` observation
with this shape:

```json
{
  "entity_id": "CIK0000123456",
  "entity_name": "Example Corp",
  "entity_type": "company",
  "attribute": "ceo_of",
  "old_value": "Former CEO",
  "new_value": "Current CEO",
  "effective_ts": "2026-07-28T09:00:00Z",
  "observed_ts": "2026-07-28T12:00:00Z",
  "source_url": "https://authority.example/event",
  "source_type": "filing",
  "resolver_id": "edgar-8k",
  "authority_family": "sec",
  "attester_id": "CIK0000123456",
  "attester_role": "issuer",
  "compliance_source_id": "sec_edgar",
  "distribution_rights_confirmed": true,
  "aliases": ["C. CEO"],
  "provenance": {}
}
```

`FactEvent` is the only curation artifact accepted by `build_dataset`. A source
candidate lacks a curated assertion, while a `CorvusRow` is already a frozen
benchmark example. Passing either artifact type to the builder is a pipeline
error.

#### Three axes of independence

Corroboration is only meaningful against a specific failure mode, so the three
things that "independent source" can mean are recorded separately:

| Field | Answers | Role |
|---|---|---|
| `resolver_id` | which detector produced this? | two detectors reading one document are not two observations |
| `attester_id` | who asserted it and is accountable? | **the eligibility gate** |
| `authority_family` | who published it? | provenance; an optional extra gate |

A row is eligible by default when **two resolvers and two attesters** agree on
the entity, attribute, new value, effective time, and previous answer.

The gate is attester, not publisher, because the errors that corrupt a
benchmark row — a wrong name, a wrong effective date — are caught by two
parties who prepared their accounts separately and are separately liable for
them. An issuer's Form 8-K and the incoming officer's Section 16 Form 3 are
exactly that, and both reach the public through EDGAR. Requiring two publishers
would reject them while admitting any pair of aggregators that both scraped the
same press release.

`attester_id` defaults to `authority_family` when an adapter does not set it,
so an adapter that cannot distinguish attesters stays exactly as strict as
publisher-level agreement. Publisher independence is still available as
`--min-authorities 2` for a study that needs it.

Build the dry-run dev set and publication test set as separate freezes:

```bash
python -m corvus.cli.build_dataset events-dev.jsonl corvus-dev.jsonl \
    --split dev --freeze-id 2026-07-dry-run \
    --as-of 2026-07-30T17:00:00Z
python -m corvus.cli.build_dataset events-test.jsonl corvus-test.jsonl \
    --split test --freeze-id 2026-08-preregistered \
    --as-of 2026-08-30T17:00:00Z
```

Each command writes the eligible rows, a rejection ledger, and a SHA-256
manifest. Inspect those artifacts before importing. The importer verifies the
hash and refuses empty input; it then replaces the selected dataset head and
creates a named snapshot:

```bash
python -m corvus.cli.import_dataset corvus-dev.jsonl \
    --manifest corvus-dev.jsonl.manifest.json --split dev \
    --compliance-approval compliance-approval.json
python -m corvus.cli.import_dataset corvus-test.jsonl \
    --manifest corvus-test.jsonl.manifest.json --split test \
    --compliance-approval compliance-approval.json
```

The split is encoded in physically separate Braintrust datasets
(`Corvus-QA-dev` and `Corvus-QA-test`) so harness tuning cannot silently consume
test rows. Run either with `--dataset-name` and its pinned snapshot version.
Corvus metadata maps `effective_ts` to `event_date` and source URLs to
`articles[*].url`, so the existing temporal-grounding and leakage scorers work
without a dataset-specific branch. `previous_answer` is retained for
stale-answer analysis.

EDGAR/Wikidata adapters live in [corvus/sources.py](corvus/sources.py);
news/sports adapters live in
[corvus/news_sports_sources.py](corvus/news_sports_sources.py). They require a
monitored organizational `CORVUS_CONTACT_EMAIL`, and SEC additionally requires
the single-deployment confirmation in `.env`. Requests are serial, locally
rate-limited, retried with bounded backoff, and restricted to approved hosts.
EDGAR Item 5.02 candidates require a curated `OfficerTransition`; only the
supporting excerpt's hash enters provenance. Wikidata changes use the cited
reference's domain as the authority family, preventing an SEC-cited Wikidata
claim from falsely counting as independent of EDGAR.

Use the compliance-gated collector rather than constructing generic HTTP
clients:

```bash
python -m corvus.cli.smoke_test
python -m corvus.cli.check_compliance
python -m corvus.cli.collect_sources edgar-candidates \
    --cik 0000123456 --since 2026-07-01 --until 2026-07-31 \
    --output edgar-candidates.jsonl
python -m corvus.cli.collect_sources wikidata-latest \
    --qid Q123 --property P169 --attribute ceo_of \
    --canonical-entity-id CIK0000123456 --entity-type company \
    --effective-ts 2026-07-28T09:00:00Z \
    --output wikidata-events.jsonl
```

EDGAR collection deliberately emits candidates rather than guessing officer
names or effective dates from prose. A curated `OfficerTransition` must be
created from the filing before `EdgarAdapter.emit_fact` produces a `FactEvent`.

### Section 16 corroboration

The issuer's Item 5.02 filing is one account of an officer change. The incoming
officer's own Section 16 filing is a second, separately liable account — and
unlike the 8-K it is structured, so the corroborating value and date are read
rather than parsed out of prose:

- `periodOfReport` is the "Date of Event Requiring Statement", the date the
  person actually became an officer. Precision is one day; a Section 16-backed
  row must not be read at a finer recency rung than that.
- `reportingOwnerRelationship` carries `isOfficer`/`isDirector` and
  `officerTitle`, which the SEC makes mandatory whenever `isOfficer` is set.

Form 3 is the initial statement, due within ten days, and covers external
hires. An internal promotion files no new Form 3 — the person is already a
reporting owner — so `detect_promotion` looks for a changed `officerTitle` on
their next Form 4 instead.

Pairing runs entirely against the bulk submissions archive already on disk and
spends no request budget; only the paired documents are then fetched:

```bash
python -m corvus.cli.collect_sources section16-pair \
    --candidates edgar-item-502-candidates.jsonl \
    --archive sec_submissions.zip \
    --form 3 \
    --output ownership-refs.jsonl
python -m corvus.cli.collect_sources section16-fetch \
    --refs ownership-refs.jsonl \
    --output-dir section16/documents \
    --output ownership-filings.jsonl \
    --failures ownership-failures.jsonl
```

Titles map conservatively. `officer_role_attribute` refuses deputy, vice,
interim, acting, and co- variants, and refuses any office scoped below the
issuer, so "CEO of Acme Europe" and "President & CEO, Retail Division" never
corroborate a top-office transition. It also refuses "Chief Executive Officer
of Acme Bank" even when Acme Bank *is* the filer's operating subsidiary — a
false negative accepted so that a divisional title can never become a false
positive.

`rptOwnerName` is stored surname-first with no delimiter, which is lossy for
compound surnames: "Garcia Lopez Maria" could be either name. Agreement between
attesters therefore uses `person_names_agree`, which compares unordered token
sets and ignores middle initials, so "Amanda M. Cole" matches "Cole Amanda
Marie" and a wrong surname guess cannot cause a false reject.

`reportingOwnerAddress` and the signature block are never read. The reporting
owner's name is the answer to the question; every other personal field in the
filing is out of scope under the source policy.

Section 16 corroborates the **new value and the effective date only**. A Form 3
is an initial statement and says nothing about a predecessor, so
`previous_answer` stays single-attested from the curated 8-K, and every
Section 16 observation records `old_value_attested: false` in provenance.

Measured yield on the July 2026 development window: 490 Form 3 filings paired
against 798 Item 5.02 candidates, 30 carrying a top-office title — of which 11
are blank-check shells with no realistic search footprint. See
[the window's data card](data/corvus_live/2026-07-dev/README.md). A one-month
window is not enough for a publication test set; widen the window or add Form 4
before treating this as the test split.

#### Section 16 curation checks

Pairing is only candidate generation. Before emitting the issuer and reporting
owner as two `FactEvent` observations, the curator must confirm:

1. The Item 5.02 filing describes the same appointment as the paired Form 3.
2. The effective dates agree; a mismatch is rejected rather than corrected
   silently.
3. The published name and top-office mapping are unambiguous.
4. The issuer is useful for the benchmark rather than an obvious blank-check
   shell with no realistic search footprint.

The two observations must retain distinct `resolver_id`, `attester_id`, and
`attester_role` values even though both documents are published through SEC.
The predecessor remains single-attested unless another source confirms it.

### News and sports sources

The news/sports slice uses only metadata or derived results from sources with
documented reuse terms:

- Wikipedia Current Events supplies revision, section, permanent-link, and
  cited-source URL metadata under CC BY-SA 4.0. Event prose is not copied and
  cited pages are not fetched.
- OpenLigaDB supplies football results under ODbL 1.0.
- TheSportsDB supplies a small free-tier result sample through its official
  API. Artwork, videos, and third-party media are excluded.

Before making a live request, review the linked policies in
[the compliance guide](docs/corvus-compliance.md). Set only the corresponding
source attestation in `.env`—`CORVUS_WIKIPEDIA_TERMS_CONFIRMED`,
`CORVUS_OPENLIGADB_LICENSE_CONFIRMED`, or
`CORVUS_THESPORTSDB_TERMS_CONFIRMED`—after its attribution, share-alike, and API
conditions have been accepted. Then collect bounded slices:

```bash
python -m corvus.cli.collect_sources wikipedia-current-events \
    --date 2026-07-30 --output current-events.jsonl
python -m corvus.cli.collect_sources openligadb-results \
    --league bl1 --season 2026 --group-order 1 \
    --output openligadb-results.jsonl
python -m corvus.cli.collect_sources thesportsdb-results \
    --date 2026-07-30 --sport Soccer \
    --output thesportsdb-results.jsonl
```

Sports candidates do not become benchmark rows automatically. A curator must
map each provider's event and team names to one canonical match identity and
confirm when the result became effective; the scheduled start time is retained
separately and is never silently treated as the completion time.
Only matching final scores from two independent authorities qualify under the
existing dual-authority rule.

The reconciliation stage is executable. Its decision JSONL records the exact
provider event IDs, canonical team names, completion timestamp and evidence,
and approval state. It rejects missing or reused candidates, score or sport
disagreement, implausibly different start times, and insufficient independent
resolvers or authorities:

```bash
python -m corvus.cli.curate_sports \
    --candidates openligadb-results.jsonl \
    --candidates thesportsdb-results.jsonl \
    --decisions reconciliation-decisions.jsonl \
    --output sports-events.jsonl
python -m corvus.cli.build_dataset \
    sports-events.jsonl corvus-sports-dev.jsonl \
    --split dev --freeze-id sports-2026-05-bundesliga-dev \
    --as-of 2026-08-07T19:00:00Z
```

An end-to-end three-match freeze is available locally under
`data/corvus_live/2026-05-sports-dev/`. Its completion timestamps use a disclosed
end-of-event-date upper bound because the two APIs expose scheduled start and
completed status, but not a final-whistle timestamp. This is suitable for the
`gte_30d` development rung; do not use that bound for hour-level freshness work.

Optional trap JSONL can be supplied with `--traps-file` and `--run-end`. A trap
is accepted only when two resolvers and two authorities agree on its scheduled
resolution and it resolves more than seven days after the run:

```bash
python -m corvus.cli.build_dataset events-test.jsonl corvus-test.jsonl \
    --split test --freeze-id 2026-08-preregistered \
    --as-of 2026-08-30T17:00:00Z \
    --traps-file traps-test.jsonl --run-end 2026-09-02T17:00:00Z
```

Recency rungs are computed against `--as-of`. Coverage tiers may be loaded with
`--coverage-file`, but every assessment must affirm storage rights; unlicensed
search-result storage fails validation and its `compliance_source_id` must be
approved by the source policy. Review
[Corvus compliance review](docs/corvus-compliance.md) and the machine-readable
[source policy](config/corvus/source_compliance.json) before collecting or
publishing any source.

Use one `study-id`, pinned dataset version, and agent model across every
condition in a comparison. Interleave conditions in time—running them days
apart makes the web itself a variable:

Harness arms require `YDC_API_KEY` in `.env`; native arms require the model
vendor's key. Generic webpage fetching remains disabled.

```bash
python run_eval.py run --search-mode harness --arm native_fresh \
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
python run_eval.py run --search-mode harness --arm native_fresh \
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

1. **The native-vs-harness contrast varies the prompt too.** The native arm's
   system prompt cannot describe a tool schema it does not own, so that contrast
   changes both search mode and one prompt sentence. Budget and answer format are
   held identical and `prompt_version` records which text ran, but the two
   variables cannot be fully separated by this design. A prompt-only control arm
   (harness tool, native-style wording) would bound the effect; it is not built.
2. **The OpenAI native arm is not held to the search budget.** Its hosted
   `web_search` exposes no `max_uses`, so `search_budget_enforced` is False there
   and only there. `budget_economy` reports the observed count, so violations are
   visible, but the arm is genuinely less constrained than the seven others.
3. **Freshness is unanswerable on one native arm.** OpenAI native returns no
   per-result dates, so `temporal_grounding` — the repository's headline
   construct — is `None` for that entire arm. Freshness conclusions cover the
   harness arms and Anthropic native only.
4. **No search layer reports a true publication date.** You.com's `page_age`
   and Anthropic native's `page_age` are both last-modified; OpenAI native has no
   date field. Semantics are uniform, so there is no pooling hazard — but
   `temporal_grounding` measures "last touched" everywhere, and a re-rendered page
   looks fresh without carrying new information. Prefer Corvus-QA's
   `recency_rung` as the freshness variable.
5. **Judge and agent share a vendor on the OpenAI arms.** The default `gpt-4.1`
   judge cannot rule out self-preference when grading a `gpt-5.6-sol` agent, and
   that bias is confounded with the native-search treatment. Use a cross-vendor
   jury (`--judge` is repeatable, and `ANTHROPIC_API_KEY` is now configured) for
   any reported frontier comparison.
6. **No multiplicity control across 22 conditions.** The analysis computes
   paired effects per condition against one baseline; 22 contrasts inflate
   family-wise error. Pre-specify a small number of primary contrasts and label
   the rest exploratory.
6b. **Do not report a pooled cross-category number.** A retrieval effect can be
    substantial in one category and absent in another, so a pooled figure can hide
    a sign change. `benchmark_category` is logged on every row; report the
    breakdown, and treat the pooled mean as a summary of it rather than the
    result.
7. **`--trials 3` is unjustified.** No power analysis or minimum detectable
   effect has been computed for web-retrieval nondeterminism.
8. **`zero_search_row` needs a stated exclusion rule.** Rows where the tool was
   available and unused are no-search rows inside a search arm. They are flagged
   but not automatically excluded, and the exclusion rate is itself
   arm-dependent — an OSS tool-calling failure and a frontier model's decision
   not to search are different phenomena with the same flag.
9. **No automated time blocking.** Conditions must still be manually
   interleaved. Publication runs should randomize or round-robin conditions at
   task level to reduce web-time confounding.
10. **No expert multi-step task domain.** Results do not generalize to legal,
    finance, or other professional research workflows.
11. **Bootstrap, not mixed effects.** The included analysis is dependency-free
    and task-paired; final reporting should also fit condition fixed effects
    with task random intercepts.
12. **Row cost is priced from pinned list prices, not billed usage.**
    `model_cost_usd` and `total_cost_usd` are now computed per row from token
    counts and `agents.MODEL_USD_PER_MTOK`, so the cost frontier no longer
    depends on downstream span aggregation. Two caveats remain: the OSS arm is
    **unpriced** (Baseten publishes no per-token Model API rate in the pinned
    docs), so its `model_cost_usd` is `None` and `model_cost_confirmed=False`
    — it must be excluded from cost comparisons rather than read as cheap; and
    list prices ignore promotional rates, deliberately, so a recorded cost does
    not change meaning when a promotion lapses.
13. **Frontier gateway smoke checks pass; Baseten remains unverified.** On
    2026-08-10, `gateway-check` returned HTTP 200 for both frontier models and
    preserved the native-search blocks consumed by the adapters: OpenAI returned
    `web_search_call` and `url_citation`; Anthropic returned `server_tool_use`
    and `web_search_tool_result`. Re-run these billed checks whenever the gateway
    or provider configuration changes. The Baseten model slugs and harness path
    have not received the same live check.

## Open decisions

The code picks a default for each of these. The defaults are assumptions, not
conclusions. Settle them before publishing.

### Blocking

**1. One search API means one narrower claim.** Nothing here separates "You.com
beats native search" from "independent APIs beat native search." Accept the
narrower claim, or add a second API back and restore the cross-API contrast.

**2. No search layer reports a true publication date.** You.com's `page_age` and
Anthropic native's `page_age` are both *last-modified*; OpenAI native has no date
field. `temporal_grounding` therefore measures "last touched" everywhere, and a
re-rendered page looks fresh without carrying new information. The recommended
resolution is to make Corvus-QA's `recency_rung` the freshness variable — it is
dataset ground truth about when the fact changed rather than vendor metadata.
Decide which one the freshness claim rests on before running.

**3. The OpenAI native arm is not held to the search budget.** Its hosted
`web_search` publishes no `max_uses`, so `search_budget_enforced` is False there
and only there. Violations are visible via `budget_economy`, but that one arm is
genuinely less constrained than the other five.

**4. The native arms' prompt differs by one sentence.** A native prompt cannot
describe a tool schema it does not own. Budget and answer format are held
identical and `prompt_version` records which text ran, but the native-vs-harness
contrast varies search mode *and* that sentence. A prompt-only control arm would
bound it; it is not built.

### Important

**5. The judge and the agent can share a vendor.** The default `gpt-4.1` judge
cannot rule out self-preference when grading `gpt-5.6-sol`, and that bias is
confounded with the native-search treatment. `--judge` is repeatable; use a
cross-vendor jury through the configured gateway for any reported frontier
comparison.

**6. Search pricing must be rechecked before publication.** `YDC_USD_PER_CALL`
and `agents.NATIVE_SEARCH_USD_PER_CALL` were checked on 2026-08-05: You.com is
$5/1k calls independent of `count`, and both native tools are $10/1k searches.
`agents.MODEL_USD_PER_MTOK` uses list prices and deliberately ignores promotional
rates. The OpenAI native rate assumes a reasoning model; a non-reasoning model
routes through `web_search_preview` at $25/1k, which
`native_search_rate_confirmed` flags.

**7. Some rows have no leakage ground truth.** `leakage_guard` is applicable only
where a row carries gold source domains. LiveNewsBench rows that omit them and
all RetrievalQA rows return `None` rather than a passing score. Report the
applicable-row count alongside any leakage rate.

**8. Item pairing needs a pinned dataset.** The runner requires
`--dataset-version` unless `--allow-latest` is passed. Keep it pinned: unpaired
rows are dropped from every contrast, and the drop count is part of the result.

### Minor

**9. `trial_count = 3` is thin.** Enough to show retrieval is nondeterministic,
not enough to bound run-to-run variance. Compute the variance envelope before
interpreting any effect — see [docs/study-design.md](docs/study-design.md).

**10. Baseten sampling is unverified.** The OSS arm sends `temperature: 0`, which
OpenAI-compatible serving stacks normally accept, but Baseten does not document
per-model sampling support and both OSS rows are reasoning models. If a row 400s
on `temperature`, that is the first thing to check.

**11. RetrievalQA measures a different construct.** It runs through the same
harness but its questions are not time-sensitive, so freshness treatments should
not be expected to move it. Useful as a negative control, not as a freshness
result.

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
- **Runaway pressure.** `refused_searches` counts tool calls the five-search
  cap turned away. A provider that constantly pushes past the
  budget would cost far more uncapped — a real failure mode that a hard cap
  otherwise hides completely.

## Deliberate choices

Not open questions. Listed so they do not get changed by accident.

- Generic webpage fetching is disabled. Restoring it requires an explicit
  domain allowlist plus source-specific terms, robots, pacing, redirect, and
  retention controls.
- You.com `livecrawl` stays off in every setup. It only fills
  `result.contents`, which the agent never sees, so it would add latency and
  $1/1k pages without changing what is measured.
- You.com requests send `Cache-Control: no-cache`. Their docs note `GET`
  responses are cacheable at CDN and proxy layers while `POST` responses are not,
  and freshness is the quantity under measurement.
- You.com setups vary only `count` and `freshness`. Every other parameter —
  snippet budget, exclusion list, search budget — is held constant, so a setup
  difference is attributable to the parameters that differ.
- `count` does not affect You.com's price, so the `wide` setup's larger decision
  surface is free. All four harness setups cost the same per call.
- No setup derives dates from `event_date`, and queries avoid temporal words.
  You.com uses the broader of query-implied and parameter freshness, so a dated
  query would un-blind the arms.
- Raw pre-normalization provider payloads are not retained in traces.
- Every provider emits the same Braintrust search span input/output, metric
  names, and safe provider correlation ID (`requestId`, `search_id`, or
  `search_uuid`) for support/debugging without retaining raw payloads.
- `notrace_io=True` on the search tool span prevents `@traced` from logging the
  return tuple over the explicit `output=`.
- A single gold answer, not a multi-item rubric. Rubric grading suits open-ended
  professional tasks; LiveNewsBench and RetrievalQA are short-form factoid QA
  where the gold answer is one fact, so the SimpleQA three-way grade fits and a
  rubric would invent structure the dataset does not have. The rubric idea worth
  borrowing is *gating*, which is what `dealbreaker_gate` does.

> **Experimental principle:** swap the search layer while holding the agent setup
> fixed, and attribute only within a vendor. Applied here to freshness-focused,
> short-form QA, with added controls for public-dataset contamination, source
> leakage, and decision-surface observability.
