# Corvus-QA sports development freeze

This local, non-published freeze proves the sports path from two live result
sources through reconciliation and the standard Corvus row builder.

- Scope: three German Bundesliga matches starting 2026-05-16 13:30 UTC.
- Sources: OpenLigaDB and TheSportsDB official APIs.
- Agreement: both sources agree on each home/away pairing and final score.
- Yield: six `FactEvent` observations, three eligible dev rows, zero rejected
  groups.
- Freeze ID: `sports-2026-05-bundesliga-dev`.
- As-of timestamp: 2026-08-07 19:00 UTC; every row is in the `gte_30d` rung.
- External uploads: none.

Provider team aliases were mapped explicitly in
`reconciliation-decisions.jsonl`. Neither source exposes a final-whistle
timestamp, so the reviewed decisions use 23:59:59 UTC on the event date as a
disclosed conservative upper bound. That precision is adequate only for this
older-than-30-days development slice, not for hour-level freshness analysis.

Collected candidates, decisions, normalized events, rows, rejection ledgers,
and manifests remain gitignored under this directory because they include gold
answers and have not received artifact-bound publication approval.
