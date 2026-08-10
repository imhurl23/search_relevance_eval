from datetime import datetime, timedelta, timezone
import unittest

from corvus.models import DatasetSplit, build_rows
from corvus.news_sports_sources import SportsResultCandidate
from corvus.sports_curation import (
    SportsCandidateRef,
    SportsMatchDecision,
    reconcile_sports,
)


OBSERVED = datetime(2026, 8, 7, 18, tzinfo=timezone.utc)
START = datetime(2026, 5, 16, 13, 30, tzinfo=timezone.utc)


def candidate(
    resolver: str,
    authority: str,
    event_id: str,
    home: str,
    away: str,
    home_score: int = 5,
    away_score: int = 1,
) -> SportsResultCandidate:
    return SportsResultCandidate(
        source_event_id=event_id,
        sport="Soccer",
        competition="German Bundesliga",
        season="2025-2026",
        event_start_ts=START,
        observed_ts=OBSERVED,
        home_team=home,
        away_team=away,
        home_score=home_score,
        away_score=away_score,
        source_url=f"https://{authority}.example/match/{event_id}",
        source_type=f"{authority}_completed_match",
        resolver_id=resolver,
        authority_family=authority,
        compliance_source_id="test_fixture",
        attribution=f"Data from {authority}",
    )


def decision(*, approved: bool = True) -> SportsMatchDecision:
    return SportsMatchDecision(
        decision_id="bundesliga-2026-05-16-bayern-koln",
        approved=approved,
        canonical_event_id="sports:bundesliga:2026-05-16:bayern:koln",
        canonical_home_team="Bayern Munich",
        canonical_away_team="FC Köln",
        effective_ts=datetime(2026, 5, 16, 15, 25, tzinfo=timezone.utc),
        effective_time_evidence="reviewed official final-whistle record",
        observations=[
            SportsCandidateRef(resolver_id="openliga", source_event_id="1"),
            SportsCandidateRef(resolver_id="sportsdb", source_event_id="2"),
        ],
    )


class SportsCurationTests(unittest.TestCase):
    def setUp(self):
        values = [
            candidate("openliga", "openligadb", "1", "FC Bayern München", "1. FC Köln"),
            candidate("sportsdb", "thesportsdb", "2", "Bayern Munich", "FC Köln"),
        ]
        self.candidates = {
            (item.resolver_id, item.source_event_id): item for item in values
        }

    def test_reconciles_aliases_and_builds_a_corvus_row(self):
        facts, skipped = reconcile_sports(self.candidates, [decision()])
        self.assertEqual(skipped, [])
        self.assertEqual(len(facts), 2)
        self.assertEqual({fact.new_value for fact in facts}, {"Bayern Munich 5-1 FC Köln"})
        self.assertEqual(
            {fact.attester_id for fact in facts}, {"openligadb", "thesportsdb"}
        )
        rows, rejected = build_rows(
            facts,
            split=DatasetSplit.DEV,
            freeze_id="sports-dev-v1",
            as_of_ts=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].input["question"],
            "What was the final score of Bayern Munich vs FC Köln?",
        )

    def test_rejects_score_disagreement(self):
        bad = candidate(
            "sportsdb", "thesportsdb", "2", "Bayern Munich", "FC Köln", 4, 1
        )
        candidates = dict(self.candidates)
        candidates[(bad.resolver_id, bad.source_event_id)] = bad
        with self.assertRaisesRegex(ValueError, "final score disagreement"):
            reconcile_sports(candidates, [decision()])

    def test_rejects_completion_before_start(self):
        item = decision().model_copy(update={"effective_ts": START - timedelta(minutes=1)})
        with self.assertRaisesRegex(ValueError, "precedes an event start"):
            reconcile_sports(self.candidates, [item])

    def test_unapproved_decision_is_skipped(self):
        facts, skipped = reconcile_sports(self.candidates, [decision(approved=False)])
        self.assertEqual(facts, [])
        self.assertEqual(skipped, ["bundesliga-2026-05-16-bayern-koln"])


if __name__ == "__main__":
    unittest.main()
