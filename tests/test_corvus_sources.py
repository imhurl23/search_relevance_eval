from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx

from corvus.models import DatasetSplit, build_rows
from corvus.news_sports_sources import (
    OpenLigaDbAdapter,
    TheSportsDbAdapter,
    WikipediaCurrentEventsAdapter,
)
from corvus.sources import (
    EdgarAdapter,
    OfficerTransition,
    PolicyHttpClient,
    WikidataAdapter,
    assert_source_approved,
    authority_family_for_url,
    make_news_sports_source_adapters,
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

    def test_wikipedia_current_events_collects_metadata_without_prose(self):
        def handler(request):
            action = request.url.params["action"]
            if action == "query":
                payload = {
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "title": "Portal:Current events/2026 July 30",
                                "revisions": [
                                    {
                                        "revid": 123456,
                                        "timestamp": "2026-07-30T20:00:00Z",
                                    }
                                ],
                            }
                        ]
                    }
                }
            else:
                payload = {
                    "parse": {
                        "sections": [
                            {"line": "Sports", "level": "2"},
                            {"line": "Football", "level": "3"},
                        ],
                        "externallinks": [
                            "https://publisher.example/news/story",
                            "https://publisher.example/news/story",
                        ],
                    }
                }
            return httpx.Response(200, json=payload, request=request)

        client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("wikipedia.org",),
            rate_limit_key="test-wikipedia-current-events",
            transport=httpx.MockTransport(handler),
        )
        candidate = WikipediaCurrentEventsAdapter(client).page_candidate(
            date(2026, 7, 30)
        )
        self.assertEqual(candidate.revision_id, 123456)
        self.assertEqual(candidate.section_titles, ["Sports", "Football"])
        self.assertEqual(
            candidate.external_source_urls,
            ["https://publisher.example/news/story"],
        )
        self.assertFalse(candidate.contains_page_text)
        self.assertNotIn("text", candidate.model_dump())

    def test_sports_adapters_emit_reconcilable_independent_results(self):
        openliga_payload = [
            {
                "matchID": 111,
                "matchDateTimeUTC": "2026-07-30T18:30:00Z",
                "leagueName": "Example League",
                "leagueSeason": "2026",
                "matchIsFinished": True,
                "team1": {"teamName": "Alpha FC"},
                "team2": {"teamName": "Beta FC"},
                "matchResults": [
                    {
                        "resultOrderID": 1,
                        "resultTypeID": 2,
                        "pointsTeam1": 2,
                        "pointsTeam2": 1,
                    }
                ],
            }
        ]
        sportsdb_payload = {
            "events": [
                {
                    "idEvent": "222",
                    "strTimestamp": "2026-07-30T18:30:00Z",
                    "strSport": "Soccer",
                    "strLeague": "Example League",
                    "strSeason": "2026",
                    "strHomeTeam": "Alpha",
                    "strAwayTeam": "Beta",
                    "intHomeScore": "2",
                    "intAwayScore": "1",
                }
            ]
        }
        openliga_client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("api.openligadb.de",),
            rate_limit_key="test-openligadb",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=openliga_payload, request=request
                )
            ),
        )
        sportsdb_client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("thesportsdb.com",),
            rate_limit_key="test-thesportsdb",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=sportsdb_payload, request=request
                )
            ),
        )
        observed = datetime(2026, 7, 30, 20, tzinfo=timezone.utc)
        openliga = OpenLigaDbAdapter(openliga_client).completed_results(
            "example",
            "2026",
            observed_ts=observed,
        )[0]
        sportsdb = TheSportsDbAdapter(sportsdb_client).completed_results(
            date(2026, 7, 30),
            sport="Soccer",
            observed_ts=observed,
        )[0]
        facts = [
            candidate.emit_fact(
                canonical_event_id="sports:example:2026-07-30:alpha:beta",
                canonical_home_team="Alpha",
                canonical_away_team="Beta",
                effective_ts=datetime(2026, 7, 30, 21, tzinfo=timezone.utc),
            )
            for candidate in (openliga, sportsdb)
        ]
        self.assertEqual({fact.new_value for fact in facts}, {"Alpha 2-1 Beta"})
        self.assertEqual(
            {fact.authority_family for fact in facts},
            {"openligadb", "thesportsdb"},
        )
        rows, rejected = build_rows(
            facts,
            split=DatasetSplit.DEV,
            freeze_id="sports-smoke-v1",
            as_of_ts=datetime(2026, 7, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0].expected, "Alpha 2-1 Beta")

    def test_news_sports_factory_requires_terms_attestation(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "CORVUS_CONTACT_EMAIL=data-ops@example.com\n"
                "CORVUS_CONTACT_EMAIL_IS_ROLE_ACCOUNT=yes\n"
            )
            with self.assertRaisesRegex(ValueError, "WIKIPEDIA_TERMS_CONFIRMED"):
                make_news_sports_source_adapters(
                    env_path,
                    enabled_sources={"wikipedia_current_events"},
                )


if __name__ == "__main__":
    unittest.main()
