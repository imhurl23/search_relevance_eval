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
axis under test. See [The test matrix](#the-test-matrix) for the eight cells and
for which comparisons the instrumentation actually supports.

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
Parallel, and You.com. A first implementation of the event-sourced Corvus-QA
pipeline is also included. The **Remaining gaps** and **Open decisions**
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
- Search APIs (`--search-mode harness`): `exa`, `parallel`, `youdotcom`
- Freshness treatments (`--arm`, harness only): `normalized` (uniform snippet
  budget), `native_fresh` (each vendor's own freshness knobs)
- Datasets: `LiveNewsBench`, `RetrievalQA` — Braintrust, versioned with named
  snapshots; `Corvus-QA-dev` and `Corvus-QA-test` can be built and imported
  separately
- Scorers: 10 deterministic plus one judge score per run, single or jury. See
  [scorers.py](scorers.py)

## The test matrix

Two independent axes, set by `--model-vendor` and `--search-mode`:

| `model_class` | vendor | `none` | `harness` | `native` |
|---|---|---|---|---|
| oss | `baseten` (`openai/gpt-oss-120b`) | ✅ | ✅ | ⛔ structurally unavailable |
| frontier | `openai` (`gpt-5.6-sol`) | ✅ | ✅ | ✅ Responses `web_search` |
| frontier | `anthropic` (`claude-opus-5`) | ✅ | ✅ | ✅ `web_search_20250305` |

Model IDs are pinned snapshots, not aliases: `gpt-5.6` is an alias for
`gpt-5.6-sol` and will move.

**The two frontier models are not claimed to be equivalent.** There is no
vendor-neutral capability tier. Matching on price pairs `gpt-5.6-sol` with
`claude-opus-5` ($5/$30 vs $5/$25); matching on within-lineup position pairs sol
with `claude-fable-5`, Anthropic's flagship ($10/$50). The two framings disagree,
so the pairing is a declared choice, not a measurement.

It is tolerable because no primary contrast depends on it — `native` vs `harness`
and search vs `none` both hold the model fixed within a vendor, and cross-vendor
native comparisons are already ruled out. What the pairing does bound is how far
an oss-vs-frontier result generalizes: read that contrast as "vs *this* frontier
model", not "vs frontier models". For flagship-vs-flagship, pass
`--agent-model claude-fable-5` (about 2x the cost; requires the org not be on
zero data retention). No code change needed.

Eight cells. Three properties of this shape are load-bearing:

**`native` is only attributable within a vendor.** If GPT gets native search and
Claude gets the harness tool, model identity is confounded with search mode. So
each frontier vendor runs all three modes, and its native arm is compared to its
own harness and `none` arms — never across vendors.

**Every model needs a `none` arm.** Without a parametric floor, a high
native-search score cannot be separated from what the model already knew, and
LiveNewsBench rows predating a model's cutoff are answerable with no search.

**`--search-mode native` and `--arm native_fresh` are different things.**
`native_fresh` is a *search API's* freshness parameter (Exa `maxAgeHours`, You.com
`freshness=day`). `native` is the *model vendor's* own server-side search. The
runner rejects combining them, and `condition_id` keeps them distinct.

```bash
# frontier, native search (per vendor)
python run_eval.py run --model-vendor anthropic --search-mode native \
  --dataset-version <xact-id> --study-id matrix-v1 --trials 3
python run_eval.py run --model-vendor openai --search-mode native \
  --dataset-version <xact-id> --study-id matrix-v1 --trials 3

# frontier, harness search — same vendors, so native-vs-harness is attributable
python run_eval.py run --model-vendor anthropic --search-mode harness \
  --provider exa --arm native_fresh --dataset-version <xact-id> --study-id matrix-v1

# OSS with and without search
python run_eval.py run --model-vendor baseten --search-mode harness \
  --provider exa --arm native_fresh --dataset-version <xact-id> --study-id matrix-v1
python run_eval.py run --model-vendor baseten --search-mode none \
  --dataset-version <xact-id> --study-id matrix-v1
```

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
- **`date_field_semantics`** — `temporal_grounding` reads `published_date`, but
  Exa/Parallel report true publication dates while You.com and Anthropic native
  report *last-modified* (`page_age`). Those are different constructs: a
  re-rendered page looks fresh without carrying new information. Freshness results
  must not be pooled across the two semantics.
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

Search-provider arms require the matching API key from `.env`. Generic webpage
fetching remains disabled.

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
4. **Two date fields mean two different things.** Exa and Parallel report
   publication dates; You.com and Anthropic native report last-modified. Rows
   carry `date_field_semantics`; results must not be pooled across the two.
5. **Judge and agent share a vendor on the OpenAI arms.** The default `gpt-4.1`
   judge cannot rule out self-preference when grading a `gpt-5.6-sol` agent, and
   that bias is confounded with the native-search treatment. Use a cross-vendor
   jury (`--judge` is repeatable, and `ANTHROPIC_API_KEY` is now configured) for
   any reported frontier comparison.
6. **No multiplicity control across 23 conditions.** The analysis computes
   paired effects per condition against one baseline; 22 contrasts inflate
   family-wise error. Pre-specify a small number of primary contrasts and label
   the rest exploratory.
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
12. **Total cost requires trace aggregation.** Search fees and agent tokens are
    logged, but child LLM-span cost must be aggregated into `total_cost_usd`
    before a cost frontier is valid. Native arms make this mandatory rather than
    optional: their search-result tokens are billed on the model spans, so a
    search-fee-only comparison understates them.
13. **No live smoke run yet.** Every adapter is asserted against the vendors'
    published response schemas offline; none has been executed against the real
    APIs. The Baseten model slug in particular is documented both as
    `openai/gpt-oss-120b` and, in a migration note, as
    `baseten/openai/gpt-oss-120b`.

## Open decisions

The code picks a default for each of these. The defaults are assumptions, not
conclusions. Settle them before publishing.

### Blocking

**1. `native_fresh` uses aligned windows but different vendor semantics.**

| Provider | What `native_fresh` changes | Freshness? |
|---|---|---|
| Exa | `contents.maxAgeHours: 24` | Content cache age |
| You.com | `freshness: "day"` | Publication recency |
| Parallel | `advanced_settings.fetch_policy.max_age_seconds: 86400` | Content cache age |

All three now use a 24-hour setting, and Parallel stays on pinned `basic` mode in
both arms. The treatment still is not identical: Exa and Parallel bound the age
of fetched/indexed content, while You.com's `freshness` filters which results
are eligible by publication recency. Exa or Parallel can therefore return an
old article fetched recently; You.com cannot. Report this semantic limitation.

**2. The providers search different corpora.** Exa filters to
`category: "news"`. Parallel has no category filter, so it searches the general
web. You.com reads `results.web[]` because `results.news[]` returns no snippets
at all. A provider can win or lose on corpus scope rather than retrieval quality.
There is no clean fix, so choose which asymmetry to accept: drop Exa's news
filter, keep it and document it, or limit the eval to what all three do the same
way.

### Important

**3. Excerpt normalization removes a native Parallel advantage.**
`SNIPPET_CHARS = 400` now applies to every provider in both arms, including
Parallel's server-side excerpt setting and a client-side safety cap. This makes
the agent decision surface equivalent, but it intentionally does not measure
the value of Parallel's longer native excerpts.

**4. Exa's search tier is not frozen.** `type` defaults to `"auto"`, which routes
per query. Now pinned as `EXA_SEARCH_TYPE = "auto"` — same behavior, but visible.
Any of `instant`, `fast`, `deep-lite`, `deep`, or `deep-reasoning` would actually
freeze it, at the cost of changing retrieval quality and — for the `deep` tiers —
per-call price and latency.

**5. The judge and the agent share a vendor.** The agent is `gpt-4o` and
`--judge` defaults to `gpt-4.1`, which cannot rule out self-preference. Passing
`--judge` several times convenes a majority-vote jury across vendors instead;
every ballot and a `unanimous` flag land in row metadata. The default is still a
single judge, because that is what keeps parity with LiveNewsBench's published
numbers. Decide which one is authoritative for your headline — and note the
deployed QA Answer Correctness scorer uses `openai/gpt-oss-120b`, a third answer
to the same question. `qa_answer_match` needs no model at all, so its
disagreement with the judge is worth reporting rather than smoothing over.

**6. Search pricing must be rechecked before publication.** `SEARCH_PRICING` in
[run_eval.py](run_eval.py) now carries three documented terms per
`(provider, arm)` — per-call, per-result content, and per-result beyond 10 — and
`search_cost_usd()` applies them to the results a call actually returned. Exa and
You.com were checked against their pricing pages on 2026-07-30: Exa is $7/1k
requests plus $1/1k pages *per content type*, so an 8-result call with highlights
costs about $0.015, not the $0.005 previously recorded; You.com is $5/1k calls,
not $0.004. Parallel was checked on 2026-08-05: pinned `basic` mode is $5/1k
requests for the first 10 results plus $1/1k additional results, so both arms
cost $0.005 at this harness's eight-result limit.
`_tok()` estimates tokens as `chars / 4`; use tiktoken before publishing
token-normalized numbers, including `token_discounted_gain`.

**7. Domain exclusion works differently per vendor.** Exa takes an
`excludeDomains` array, capped at 1200 items. Parallel's
`source_policy.exclude_domains` covers subdomains automatically and caps at 200
combined. You.com takes a comma-separated string, mutually exclusive with
`include_domains` (sending both returns `422`); on `GET` it is bounded by URL
length, and their docs point to `POST` for lists past a handful, up to 500. This
eval sends well under that, so it stays on the fully documented `GET`. Apex-to-
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
- **Runaway pressure.** `refused_searches` counts tool calls the five-search
  cap turned away. A provider that constantly pushes past the
  budget would cost far more uncapped — a real failure mode that a hard cap
  otherwise hides completely.

## Deliberate choices

Not open questions. Listed so they do not get changed by accident.

- Generic webpage fetching is disabled. Restoring it requires an explicit
  domain allowlist plus source-specific terms, robots, pacing, redirect, and
  retention controls.
- You.com `livecrawl` stays off in `native_fresh`. It only fills
  `result.contents`, which the agent never sees, so it would add latency and
  $1/1k pages without changing what is measured.
- You.com requests send `Cache-Control: no-cache`. Their docs note `GET`
  responses are cacheable at CDN and proxy layers while `POST` responses are not,
  and freshness is the quantity under measurement.
- Exa's freshness arm sends `contents.maxAgeHours` only. The `livecrawl` string
  parameter is deprecated in favor of it as of February 2026, and sending both
  was self-contradictory — `livecrawl: "always"` refuses cache while
  `maxAgeHours: 24` accepts day-old cache.
- Exa's `contents.livecrawlTimeout` is pinned at the documented 10 s default so a
  crawl stall is a declared condition rather than an accident, and stays under
  the 30 s client timeout.
- No adapter derives dates from `event_date`, and queries avoid temporal words.
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

> **Relationship to Vals:** This framework uses the same basic experimental
> principle as [Vals AI's Web Search Index](https://www.vals.ai/benchmarks/web_search):
> swap the search tool while holding the agent setup fixed. It adapts that idea
> to freshness-focused, short-form QA and adds controls for public-dataset
> contamination and source leakage.
