from datetime import date, datetime, timezone
import hashlib
import unittest

import httpx

from corvus.sources import (
    EdgarAdapter,
    OfficerTransition,
    PolicyHttpClient,
    WikidataAdapter,
    assert_source_approved,
    authority_family_for_url,
)


class SourceAdapterTests(unittest.TestCase):
    def test_edgar_filters_item_502_and_emits_reviewed_fact(self):
        payload = {
            "name": "Example Corp",
            "filings": {
                "recent": {
                    "accessionNumber": ["0000123456-26-000001", "0000123456-26-000002"],
                    "filingDate": ["2026-07-29", "2026-07-29"],
                    "reportDate": ["2026-07-28", "2026-07-28"],
                    "acceptanceDateTime": [
                        "2026-07-29T12:00:00-04:00",
                        "2026-07-29T13:00:00-04:00",
                    ],
                    "form": ["8-K", "8-K"],
                    "primaryDocument": ["change.htm", "other.htm"],
                    "items": ["5.02", "8.01"],
                }
            },
        }
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload, request=request)
        )
        client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("sec.gov",),
            rate_limit_key="test-sec",
            transport=transport,
        )
        candidates = EdgarAdapter(client).list_item_502(
            "123456",
            since=date(2026, 7, 28),
            until=date(2026, 7, 30),
        )
        self.assertEqual(len(candidates), 1)
        transition = OfficerTransition(
            role_attribute="ceo_of",
            old_value="Jane Old",
            new_value="Jane New",
            effective_ts=datetime(2026, 7, 28, tzinfo=timezone.utc),
            evidence_excerpt_sha256=hashlib.sha256(b"reviewed excerpt").hexdigest(),
        )
        fact = EdgarAdapter.emit_fact(candidates[0], transition)
        self.assertEqual(fact.entity_id, "CIK0000123456")
        self.assertEqual(fact.authority_family, "sec")
        self.assertEqual(
            fact.provenance["evidence_excerpt_sha256"],
            transition.evidence_excerpt_sha256,
        )

    def test_wikidata_revision_uses_reference_authority_and_crosswalk(self):
        old_claim = {
            "mainsnak": {"datavalue": {"value": {"id": "QOLD"}}},
        }
        new_claim = {
            "mainsnak": {"datavalue": {"value": {"id": "QNEW"}}},
            "references": [
                {
                    "snaks": {
                        "P854": [
                            {
                                "datavalue": {
                                    "value": "https://www.sec.gov/Archives/example"
                                }
                            }
                        ]
                    }
                }
            ],
        }
        adapter = WikidataAdapter.__new__(WikidataAdapter)
        events = adapter.events_from_revision_pair(
            qid="QCOMPANY",
            property_id="P169",
            before={"claims": {"P169": [old_claim]}},
            after={"claims": {"P169": [new_claim]}},
            observed_ts=datetime(2026, 7, 29, tzinfo=timezone.utc),
            effective_ts=datetime(2026, 7, 28, tzinfo=timezone.utc),
            labels={
                "QCOMPANY": "Example Corp",
                "QOLD": "Jane Old",
                "QNEW": "Jane New",
            },
            attribute="ceo_of",
            canonical_entity_id="CIK0000123456",
            entity_type="company",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].entity_id, "CIK0000123456")
        self.assertEqual(events[0].authority_family, "sec")

    def test_authority_family_is_based_on_underlying_reference(self):
        self.assertEqual(authority_family_for_url("https://data.sec.gov/x"), "sec")
        self.assertEqual(
            authority_family_for_url("https://investor.example.com/news"),
            "publisher:investor.example.com",
        )

    def test_compliance_policy_blocks_conditional_source(self):
        assert_source_approved("sec_edgar")
        with self.assertRaises(ValueError):
            assert_source_approved("github_releases")

    def test_http_client_rejects_unapproved_host_before_request(self):
        client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("sec.gov",),
            rate_limit_key="test-host-block",
            transport=httpx.MockTransport(
                lambda request: self.fail("transport should not be called")
            ),
        )
        with self.assertRaisesRegex(ValueError, "not approved"):
            client.get_json("https://example.com/data.json")


if __name__ == "__main__":
    unittest.main()
