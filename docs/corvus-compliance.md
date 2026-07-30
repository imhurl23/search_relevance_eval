# Corvus-QA source and distribution review

Review date: 2026-07-30. This is an engineering compliance gate, not legal
advice. Counsel or an authorized business owner must sign off before public
release, and the linked terms must be rechecked at the test-set freeze.
The machine-readable review expires on 2026-08-29 and the tools fail closed
after that date.

## Release boundary

The public dataset may contain derived factual values, timestamps, stable
identifiers, source URLs, authority families, hashes of reviewed evidence, and
the generated question. It must not contain filing bodies, copied news or
release text, search snippets, raw authenticated API responses, API keys, or
unnecessary personal information.

The machine-readable version of this review is
[`source_compliance.json`](../config/corvus/source_compliance.json).
Official-source clients
refuse sources that are not `approved_with_controls`. Every `FactEvent` and
trap observation must also affirm `distribution_rights_confirmed`.
The builder verifies every declared source ID against the current policy. The
Braintrust importer additionally requires an artifact-bound approval based on
[`compliance_approval.example.json`](../config/corvus/compliance_approval.example.json).

## Enabled sources

### SEC/EDGAR

Status: approved with controls for derived facts and provenance URLs.

- The SEC provides public EDGAR submissions and XBRL APIs without
  authentication.
- Automated access must identify the client, download only what is needed, and
  stay below the SEC's aggregate limit of 10 requests/second.
- The adapter runs serially at one request/second and requires
  `CORVUS_CONTACT_EMAIL` from `.env`. A file lock shares that budget across
  processes in this workspace.
- Because the SEC limit is aggregate across machines, collection also requires
  `CORVUS_SINGLE_DEPLOYMENT_CONFIRMED=yes`. Set it only after confirming no
  other machine or service uses this collector's identity and request budget.
- Retries cover 429 and transient 5xx responses, honor numeric or HTTP-date
  `Retry-After`, and use capped exponential backoff with jitter.
- Item 5.02 filing acceptance is recorded as `observed_ts`, never as the fact's
  effective time. The effective time comes from a reviewed filing extraction.
- Only the excerpt's SHA-256 is retained; filing text is not distributed.

Official references:
[EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[SEC Webmaster FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions),
[SEC aggregate rate-control notice](https://www.sec.gov/filergroup/announcements-old/new-rate-control-limits).

### Wikidata

Status: approved with controls for structured data.

- Wikidata structured data is CC0.
- API clients must send a meaningful User-Agent with contact information,
  follow the robot/API policies, limit concurrency, and respect rate-limit
  responses.
- The adapter is serial and paced at one request/second. It sends gzip support,
  includes `maxlag=1`, honors `Retry-After`, and exponentially backs off on
  `maxlag`, rate-limit, and transient HTTP errors.
- A Wikidata edit timestamp is only `observed_ts`. `effective_ts` must come
  from the cited authority.
- Authority independence is computed from the reference URL. A Wikidata claim
  citing the SEC and a direct EDGAR observation are one authority, not two.

Official references:
[Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing),
[Wikimedia User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy/en),
[Wikimedia API etiquette](https://www.mediawiki.org/wiki/API:Etiquette/en),
[Wikimedia maxlag guidance](https://www.mediawiki.org/wiki/Manual:Maxlag_parameter).

### Contact identity

`CORVUS_CONTACT_EMAIL` lets the SEC or Wikimedia reach the operator about
traffic, errors, or blocking. Use a monitored organizational role address such
as `data-ops@your-domain`, not a personal mailbox: it avoids publishing an
employee's address, survives staff changes, and supports incident ownership.
It must be a real inbox; a dummy address is not compliant.
Collection remains blocked until
`CORVUS_CONTACT_EMAIL_IS_ROLE_ACCOUNT=yes` is explicitly confirmed in `.env`.

### NVD, CVE List, and CISA KEV

Status: approved with source-specific controls; adapters are a later slice.

- NVD data is a strong second security resolver. Display the required
  non-endorsement notice, use `NVD_API_KEY` only from `.env`, do not share it,
  obey the current API limits, and do not attribute modified data to NVD.
- The official CVE List permits public use and redistribution under the CVE
  Program Terms. Preserve required copyright/license notices if copying CVE
  content; Corvus should publish derived fields instead.
- CISA KEV is the authoritative dated transition source for catalog inclusion.
  Use the official CSV/JSON, retain retrieval provenance, and do not imply CISA
  endorsement.

Official references:
[NVD terms](https://nvd.nist.gov/developers/terms-of-use),
[NVD developer guidance](https://nvd.nist.gov/developers/start-here),
[CVE downloads](https://www.cve.org/Downloads),
[CVE terms](https://www.cve.org/Legal/TermsOfUse),
[CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog).

## Conditional or blocked sources

- GitHub release metadata is conditional. GitHub permits open-access research
  using public, non-personal information, but each repository's content has its
  own license. Record the repository license and publish tags/dates/URLs, not
  release-note text. Respect both primary and secondary API limits.
- Brave Search is blocked for coverage measurement unless the subscribed plan
  explicitly grants storage rights.
- GDELT Cloud is blocked for dataset distribution without a separate written
  redistribution license.
- Common Crawl content is blocked as a publication source because origin-site
  terms and rights still apply. The index is not a blanket content license.
- Sports feeds and election feeds require a per-league or per-jurisdiction
  review. A page being public does not establish redistribution rights.
- Company press releases may confirm facts, but only derived facts and URLs
  should be released. Any automated collection must be checked against that
  site's robots and terms first.
- GLEIF's CC0 LEI data is a good entity-identity crosswalk, but not an
  independent resolver for leadership changes. It is reviewed only for that
  crosswalk role and is not enabled as a fact-event source.
- Companies House is a promising official UK officer source (600 requests per
  five minutes), but it remains blocked pending a UK privacy/data-protection
  review because organizations reusing officer data become responsible for
  their own compliance.
- Exa, Parallel, and You.com benchmark runs are blocked by
  [`provider_permissions.json`](../config/corvus/provider_permissions.json)
  until written
  permission covers benchmarking, result retention, and Braintrust storage.
  Parallel expressly requires prior written consent for third-party benchmark
  results; Exa's current terms broadly restrict copying or distributing
  information obtained through the service; You.com's current API policy
  restricts storing significant API content for reuse without contractual
  authorization.
  The gate also requires provider-specific permission for Exa output reuse,
  Parallel third-party benchmark results, and uncached repeated You.com
  benchmark queries.
- Generic webpage fetching is disabled. Adding an origin requires a separate
  terms, robots, privacy, and redistribution review for that host.

Official references:
[GitHub Terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service),
[GitHub Acceptable Use](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies),
[GitHub API limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api),
[Brave Search API storage note](https://brave.com/search/api/),
[GDELT Cloud terms](https://gdeltcloud.com/terms),
[Common Crawl terms](https://commoncrawl.org/terms-of-use).

## Freeze checklist

1. Re-open every linked policy and update `reviewed_at`.
2. Record the person or counsel approving each conditional source.
3. Verify `.env` contains `CORVUS_CONTACT_EMAIL` and any source API key.
4. Run only the approved adapter factory; do not bypass pacing.
5. Inspect the rejection ledger for authority collisions and timestamp
   disagreement.
6. Confirm traps resolve more than seven days after `run_end`.
7. Confirm any coverage engine contract grants storage and publication rights.
8. Scan the final artifact for copied text, raw responses, secrets, and
   unnecessary personal information.
9. Include required NVD/CVE notices and sponsor disclosure in the dataset card.
10. Sign and tag the manifest before running the test set.
11. Confirm Braintrust workspace region, retention/deletion settings, DPA, and
    authorized users before private import.
12. Preserve upstream license notices and citations for LiveNewsBench and
    RetrievalQA if either dataset is redistributed.

## What remains a human gate

No crawler can prove legal compliance by itself. Before collection outside the
approved EDGAR/Wikidata slice, or before any external upload/public release, an
authorized owner must confirm contracts, privacy obligations, data residency,
retention, and the release jurisdiction. Do not change a `blocked` status to
`approved` merely to make a run pass; attach the actual written permission or
review reference.
