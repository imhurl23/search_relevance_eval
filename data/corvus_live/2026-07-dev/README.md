# July 2026 EDGAR candidate universe

This is the bounded, full-filer Corvus-QA development-window pull for
2026-07-01 through 2026-07-30.

The collector made one request for the SEC's official nightly bulk submissions
archive, scanned 979,030 filer records locally, and retained 798 Form 8-K Item
5.02 candidate records across 737 entities.

Files:

- `manifest.json`: immutable collection scope, hashes, counts, and provenance.
- `edgar_item_502_candidates.jsonl`: metadata requiring human filing review.
- `sec_submissions.zip`: the official nightly bulk archive, retained locally
  for reproducibility and excluded from git.

This is not yet a curated benchmark dataset. Item 5.02 covers multiple kinds of
director and officer events; each candidate must be reviewed to determine
whether it supports a Corvus question, the old/new value, and the actual
effective timestamp. Filing acceptance time is observation time only.

Review preparation:

- All 798 primary filing documents were downloaded serially from SEC and
  retained locally; their bodies remain git-ignored and must not be distributed.
- `filing_download_ledger.jsonl` records document hashes and provenance.
- `filing_review_queue.jsonl` contains hash-only triage metadata: 143 high,
  529 medium, and 126 low priority records.
- `human_review_packet.local.jsonl` contains 143 local-only review forms; 74
  have automatically located transition contexts. Evidence text in this file
  must not be published.
- `wikidata_current_ceo_crosswalk.jsonl` records the batched Wikidata CIK
  crosswalk. Coverage was sparse: 5 current CEO rows for 141 unique CIKs and
  only one reference URL, and that reference is a 2014 article about a CEO who
  was already in post. Wikidata does not corroborate this window and has been
  superseded as the second attester by Section 16.

## Section 16 corroboration

`section16/` holds the second attester for this window: the incoming officer's
own Form 3, filed under Section 16(a) within ten days of the event. It is
independent of the issuer's Item 5.02 filing because a different legal person
prepares and signs it, and it is structured, so the corroborating name and
effective date are read rather than parsed from prose.

- `section16/form3_refs.jsonl`: 490 Form 3 filings across 240 of the 737
  candidate issuers, located entirely inside `sec_submissions.zip` with no
  network requests. Pairing window is 14 days before to 45 days after each
  Item 5.02 filing, anchored per filing rather than on the issuer's earliest
  one, so an issuer that files several does not attach all of them to every
  ownership reference.
- `section16/documents/`: the 490 ownership XML documents, git-ignored.
  Addresses and signature blocks are never parsed.
- `section16/form3_filings.jsonl`: 490 parsed filings, 0 failures.
- `section16/form3_failures.jsonl`: empty.

Of the 490, **30 carry a CEO or chair title** that survives conservative
mapping, across 29 issuers. Eight more name a top office but are deliberately
refused: `Interim CEO`, `Co-CEO`, `Co-Chief Executive Officer`, and
`Vice Chairman`.

The composition matters more than the count:

| | Count | Benchmark value |
|---|---|---|
| Blank-check / SPAC shells | 11 | Low — the "transition" is a formation-time appointment, the entity has no search footprint, and no user would ask the question |
| Operating companies | 19 | Usable, but skewed to micro-caps |

Recognizable names in the operating set: Sarepta Therapeutics, FuboTV, Lands'
End, Digimarc, Eagle Bancorp, RxSight, Freenome, Nova Minerals. The remainder
are micro-cap or pre-revenue issuers, which is a legitimate `tail` coverage
stratum but cannot carry a headline freshness number on its own.

These are corroboration candidates, not rows. Each still needs the issuer side
reviewed, and the generous pairing window means a Form 3 can sit next to an
unrelated Item 5.02 — resolving that is what the review confirms. Section 16
corroborates the new value and the effective date only; `previous_answer`
remains single-attested from the 8-K.
