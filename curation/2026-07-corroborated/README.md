# Corvus-QA freeze `2026-07-sec-officers` (dev)

29 dual-attested officer-transition rows built from the July 2026 EDGAR window
with **zero network requests** — every input was already on disk from
`data/corvus_live/2026-07-dev/`.

This freeze exists to fix a specific measurement failure. On LiveNewsBench,
`gpt-5.6-sol` scored **61.5% on the judge with no retrieval at all**, because the
benchmark's events run 2025-05-01 → 2025-12-31 and the model has largely
memorised them. That leaves ≤38 points of headroom for the entire study.

## Measured bounds (`gpt-5.6-sol`, `qa_answer_match`, 1 trial)

| Arm | Score |
|---|---|
| No search (parametric floor) | **0/29 = 0.0%** |
| You.com harness, 5-search budget | **14/29 = 48.3%** |

The floor is not merely low, it is zero, and the failure mode is the useful one:
the model answers confidently with the **predecessor**. Asked who is CFO of DLH
Holdings it says "Kathryn M. JohnBull" — the outgoing CFO named in the very 8-K
that corroborates this row. Sarepta returns "Douglas S. Ingram", Eagle Bancorp
"Susan G. Riel", Six Flags "Tim Fisher". Only 8 of 29 declined or cited a cutoff;
the other 21 were confidently wrong.

That makes this a harder and more informative test than LiveNewsBench: retrieval
must **override a confident wrong prior**, not just fill a gap. There is no
partial credit available from memorisation.

It also independently corroborates the gold answers — a model naming the
predecessor for most rows is evidence the rows captured genuine transitions.

## Scope and provenance

- Window: Item 5.02 filings 2026-07-01 → 2026-07-30; effective dates
  2026-06-29 → 2026-07-28
- `as_of`: 2026-08-06T00:00:00Z → recency rungs `7d_30d` 16, `gte_30d` 13
- Attributes: `cfo_of` 14, `coo_of` 8, `ceo_of` 7
- Artifact SHA-256: `cdaf490e9972a3eff90322aea0c53a4724ec50dff8cee636a49095bf81a3f7f1`
- Both attesters per row, agreeing on person, office and date:
  the **issuer** (Item 5.02 8-K, `attester_role=issuer`) and the **reporting
  owner** (own Form 3, `attester_role=reporting_owner`). `build_dataset`
  rejected **0** groups — no `effective_ts_disagreement`, no
  `insufficient_attesters`.

## Yield, and what was refused

490 paired Section 16 filings → 88 with a benchmark office → **29 confirmed**.

| Outcome | n |
|---|---|
| `title_not_a_benchmark_office` | 402 |
| **confirmed** | **29** |
| `name_not_found` | 23 |
| `date_not_stated` | 10 |
| `role_phrase_absent` | 7 |
| `signature_block_only` | 7 |
| `date_not_marked_effective` | 5 |
| `another_office_is_closer` | 4 |
| `commences_after_event_date` | 2 |
| `appointment_not_described` | 1 |

The refusals are mostly correct rather than lost yield. `name_not_found` is
dominated by two cases: the generous ±14/45-day pairing window attaching a Form 3
to an unrelated Item 5.02, and nickname mismatches EDGAR cannot resolve
("Billy Miller" against `Miller William Dawson`). Both should refuse.

**Every confirmation was read by hand.** Four false-positive classes were found
that way and each now has a gate, with a test naming the filing that motivated it
(`tests/test_issuer_corroboration.py`):

- an office at a *different company* in the same sentence (Columbus Circle:
  "Kevin Shannon was appointed as Chief Executive Officer of Inflection Point")
- a *signature block*, where a name, an office and a date sit adjacent and no
  appointment is asserted (Meridian3)
- an office that *begins later* than Section 16's event date (SharonAI: event date
  is the agreement date, "commencing August 24, 2026" is when he takes office)
- *cross-confirmation* between officers appointed in one sentence (Yarrow appoints
  five at once; proximity alone made Zeronda the CEO)

## Known limitations

- **`answer_class` is `post_cutoff_novel` on every row.** It means "no predecessor
  was attested", NOT "the office is new". A CEO succession has a predecessor and
  this pipeline does not attest it, because neither a Form 3 (an initial
  statement) nor a name match in the successor's announcement establishes who held
  the office before. **Do not stratify on `answer_class` for this freeze.**
- **29 rows is a pilot, not a benchmark.** One month is enough to validate the
  pipeline and measure the floor; it is not enough for a dev/test split or for the
  4 primary Holm-corrected contrasts. The Feb–Jul window (3,932 paired filings) is
  the same pipeline over 6× the input.
- **Micro-cap skew.** Several issuers are SPAC shells or pre-revenue, which the
  July window's own data card already flags as a `tail` coverage stratum. Three of
  29 rows are one SPAC (Osprey). A headline number should not rest on this
  composition alone.
- **`recency_rung` has little spread** — a July window measured in August is all
  `7d_30d` or `gte_30d`. Rebuilding closer to collection would populate the
  shorter rungs.
- **Single trials are noisy.** Two ceiling runs on identical rows disagreed on
  Sarepta, Eagle and Neuronetics: live web results move. Use `--trials 3`.
- **The strict matcher still undercounts**, even after curated aliases were wired
  in (which took the ceiling from 34.5% to 48.3%). Residual misses are middle
  initials inserted into a matched name — "Rebecca V. Frey" against alias
  "Rebecca Frey", "Steve Oroho" against "Steven Oroho". This is the
  judge-versus-deterministic gap in a form where the judge is clearly right.

## Not yet published

`corvus.cli.import_dataset` refuses without an authorised, artifact-bound approval
(`require_import_approval`). That is a human gate and was not bypassed: it needs a
named `approved_by`, a `written_basis_reference`, and explicit `true` for
`braintrust_dpa_confirmed`, `braintrust_retention_policy_confirmed` and
`source_distribution_rights_confirmed`, plus `artifact_sha256` matching the hash
above. The floor and ceiling figures were measured locally, against the file, with
nothing uploaded.

## Reproducing

```bash
python -m corvus.cli.corroborate_issuer \
    --candidates      data/corvus_live/2026-07-dev/edgar_item_502_candidates.jsonl \
    --form3-filings   data/corvus_live/2026-07-dev/section16/form3_filings.jsonl \
    --form3-refs      data/corvus_live/2026-07-dev/section16/form3_refs.jsonl \
    --filings-ledger  data/corvus_live/2026-07-dev/filing_download_ledger.jsonl \
    --output          curation/2026-07-corroborated/events.jsonl

python -m corvus.cli.build_dataset \
    curation/2026-07-corroborated/events.jsonl \
    curation/2026-07-corroborated/corvus-dev.jsonl \
    --split dev --freeze-id 2026-07-sec-officers \
    --as-of 2026-08-06T00:00:00Z
```
