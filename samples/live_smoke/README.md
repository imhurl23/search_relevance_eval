# Corvus live connection smoke sample

Collected on 2026-07-30 through the compliance-gated source adapters.

Scope:

- Three SEC submissions API requests: Apple (`0000320193`), Microsoft
  (`0000789019`), and NVIDIA (`0001045810`).
- Filing window: 2026-01-01 through 2026-07-30.
- One Wikidata revision-pair check: Microsoft (`Q2283`), CEO property (`P169`).
- No filing bodies, webpage contents, raw authenticated responses, or search
  snippets were retained.
- No Braintrust or other external upload was performed.

Results:

- `edgar_candidates.jsonl`: 9 Item 5.02 candidates.
- `wikidata_events.jsonl`: 0 qualifying changes in the latest revision pair.
- EDGAR SHA-256:
  `5115f02e800b9506958f32750887f89a86f5bceb2c06f5d840da19d34a5baec3`.
- Wikidata SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

These are candidates, not curated benchmark facts. Each relevant filing must
be reviewed to establish the officer transition and effective date before
`EdgarAdapter.emit_fact` may create a `FactEvent`.
