# July 2026 schema-v2 regeneration (r2)

Regeneration of the fact-verification artifacts from
[`2026-07-schema-v2/`](../2026-07-schema-v2/README.md) after `EvidenceReference`
gained the optional `attester_id` and `attester_role` fields.

Same window, same inputs, same generator
(`corvus.cli.prepare_claim_review`) — this is a schema-shape alignment, not a
content change.

## What changed

- `fact_verification_upload.jsonl`: 12 rows, **identical row IDs and identical
  content** to the 2026-07-schema-v2 artifact. The only delta is
  `attester_id: null` and `attester_role: null` on each evidence object.
  Verified by stripping those two keys and comparing: the results are equal.
- `fact_verification_schema.json`: now declares the two new evidence properties.

Because the row IDs are unchanged, importing this upserts the existing rows in
place rather than creating duplicates.

## What did not change

`claim_preparation_upload.jsonl` and `claim_preparation_schema.json` regenerate
**byte-identical** to `2026-07-schema-v2/`, because claim-preparation rows carry
`source_urls` rather than `EvidenceReference` objects and are untouched by the
schema addition. They are deliberately not duplicated here; the approved
artifacts at `2026-07-schema-v2/` remain current.

## Why a new directory

`2026-07-schema-v2/fact_verification_approval.json` binds the sha256 of the
artifact it approved. Overwriting that file in place would silently invalidate
the approval binding and destroy the record of what was actually reviewed and
imported on 2026-07-30. The regenerated artifact therefore lives here and needs
its own approval before import.

| File | sha256 |
|---|---|
| `fact_verification_upload.jsonl` | `77aae21232b6a867558942b26615d92f171a1735d1fd6f3d600f96c04a401810` |
| `fact_verification_schema.json` | `37d624ad7d1baa1252ffb998308802fa6c7f0a553a234d530ff4293a5d510c8c` |

## Shared schema with Section 16

`fact_verification_schema.json` here is byte-identical to
`2026-07-dev/section16/fact_verification_schema.json`. Both fact-verification
sources — the 12 sports rows and the 30 Section 16 rows — now emit the same
schema, so a single dataset can enforce one schema across both. They remain
separate artifacts with separate approvals.

## Status

Both artifacts pass the schema-v2 import gate. Neither has been imported: each
still needs an approval record (see `corvus.compliance.require_import_approval`),
which is a human sign-off and is not generated automatically.
