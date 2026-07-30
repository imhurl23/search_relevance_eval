from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from corvus.models import (
    CoverageAssessment,
    DatasetSplit,
    FactEvent,
    TrapObservation,
    build_rows,
    build_trap_rows,
)
from scorers import qa_answer_match


def event(
    resolver_id: str,
    authority_family: str,
    *,
    old_value: str | None = "Jane Old",
    new_value: str = "Jane New",
) -> FactEvent:
    return FactEvent(
        entity_id="CIK0000123456",
        entity_name="Example Corp",
        entity_type="company",
        attribute="ceo_of",
        old_value=old_value,
        new_value=new_value,
        effective_ts=datetime(2026, 7, 28, 9, tzinfo=timezone.utc),
        observed_ts=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        source_url=f"https://example.com/{resolver_id}",
        source_type="filing",
        resolver_id=resolver_id,
        authority_family=authority_family,
        compliance_source_id="test_fixture",
        distribution_rights_confirmed=True,
        aliases=["J. New"],
    )


class CorvusBuilderTests(unittest.TestCase):
    def test_builds_changed_row_after_dual_authority_agreement(self):
        rows, rejected = build_rows(
            [event("edgar", "sec"), event("company-ir", "company")],
            split=DatasetSplit.DEV,
            freeze_id="dev-2026-07",
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.input["question"], "Who is the CEO of Example Corp?")
        self.assertEqual(row.expected, "Jane New")
        self.assertEqual(row.metadata["previous_answer"], "Jane Old")
        self.assertEqual(row.metadata["answer_class"], "changed")
        self.assertEqual(row.metadata["corvus_split"], "dev")
        self.assertEqual(len(row.metadata["articles"]), 2)

    def test_rejects_two_resolvers_backed_by_one_authority(self):
        rows, rejected = build_rows(
            [event("wikidata", "sec"), event("edgar", "sec")],
            split=DatasetSplit.TEST,
            freeze_id="test-2026-08",
        )

        self.assertEqual(rows, [])
        self.assertIn("insufficient_authorities", rejected[0]["reasons"])

    def test_rejects_disagreement_on_previous_answer(self):
        rows, rejected = build_rows(
            [
                event("edgar", "sec", old_value="Jane Old"),
                event("company-ir", "company", old_value="John Old"),
            ],
            split=DatasetSplit.DEV,
            freeze_id="dev-2026-07",
        )

        self.assertEqual(rows, [])
        self.assertIn("old_value_disagreement", rejected[0]["reasons"])

    def test_rejects_changed_vs_novel_disagreement(self):
        rows, rejected = build_rows(
            [
                event("edgar", "sec", old_value="Jane Old"),
                event("company-ir", "company", old_value=None),
            ],
            split=DatasetSplit.DEV,
            freeze_id="dev-2026-07",
        )

        self.assertEqual(rows, [])
        self.assertIn("old_value_disagreement", rejected[0]["reasons"])

    def test_classifies_novel_fact(self):
        rows, _ = build_rows(
            [
                event("cvelist", "cve-org", old_value=None, new_value="9.8"),
                event("nvd", "nist", old_value=None, new_value="9.8"),
            ],
            split=DatasetSplit.TEST,
            freeze_id="test-2026-08",
        )
        self.assertEqual(rows[0].metadata["answer_class"], "post_cutoff_novel")

    def test_requires_timezone_aware_timestamps(self):
        with self.assertRaises(ValidationError):
            FactEvent(
                entity_id="Q1",
                entity_name="Example",
                entity_type="company",
                attribute="ceo_of",
                new_value="Jane New",
                effective_ts=datetime(2026, 7, 28, 9),
                observed_ts=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
                source_url="https://example.com",
                source_type="filing",
                resolver_id="edgar",
                authority_family="sec",
                compliance_source_id="test_fixture",
                distribution_rights_confirmed=True,
            )

    def test_assigns_recency_and_coverage_strata(self):
        assessment = CoverageAssessment(
            entity_id="CIK0000123456",
            attribute="ceo_of",
            new_value="Jane New",
            reference_engine="licensed-reference-engine-v1",
            compliance_source_id="test_fixture",
            query="Example Corp CEO",
            queried_ts=datetime(2026, 7, 29, tzinfo=timezone.utc),
            answer_bearing_domains=["a.example", "b.example"],
            storage_rights_confirmed=True,
            evidence_artifact_sha256="a" * 64,
        )
        rows, _ = build_rows(
            [event("edgar", "sec"), event("company-ir", "company")],
            split=DatasetSplit.TEST,
            freeze_id="test-2026-08",
            as_of_ts=datetime(2026, 7, 29, 8, tzinfo=timezone.utc),
            coverage={
                ("cik0000123456", "ceo_of", "jane new"): assessment,
            },
        )
        self.assertEqual(rows[0].metadata["recency_rung"], "lt_24h")
        self.assertEqual(rows[0].metadata["coverage_tier"], "tail")

    def test_builds_trap_only_outside_safety_window(self):
        run_end = datetime(2026, 8, 1, tzinfo=timezone.utc)
        observations = [
            TrapObservation(
                trap_id="fixture-1",
                question="Who won the Example Final?",
                entity_id="fixture-1",
                entity_name="Example Final",
                entity_type="sports_fixture",
                attribute="winner_of",
                scheduled_resolution_ts=run_end + timedelta(days=8),
                observed_ts=datetime(2026, 7, 30, tzinfo=timezone.utc),
                source_url=f"https://{host}/fixture-1",
                source_type="official_fixture",
                resolver_id=resolver,
                authority_family=authority,
                compliance_source_id="test_fixture",
                distribution_rights_confirmed=True,
            )
            for host, resolver, authority in [
                ("league.example", "league-feed", "league"),
                ("venue.example", "venue-feed", "venue"),
            ]
        ]
        rows, rejected = build_trap_rows(
            observations,
            split=DatasetSplit.TEST,
            freeze_id="test-2026-08",
            run_end=run_end,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0].metadata["answer_class"], "unanswerable_trap")
        self.assertEqual(
            qa_answer_match(
                rows[0].input,
                "I could not find this.",
                rows[0].expected,
            )["score"],
            1.0,
        )

    def test_rejects_trap_inside_safety_window(self):
        run_end = datetime(2026, 8, 1, tzinfo=timezone.utc)
        observations = [
            TrapObservation(
                trap_id="fixture-1",
                question="Who won the Example Final?",
                entity_id="fixture-1",
                entity_name="Example Final",
                entity_type="sports_fixture",
                attribute="winner_of",
                scheduled_resolution_ts=run_end + timedelta(days=2),
                observed_ts=datetime(2026, 7, 30, tzinfo=timezone.utc),
                source_url=f"https://{resolver}.example/fixture-1",
                source_type="official_fixture",
                resolver_id=resolver,
                authority_family=resolver,
                compliance_source_id="test_fixture",
                distribution_rights_confirmed=True,
            )
            for resolver in ("league", "venue")
        ]
        rows, rejected = build_trap_rows(
            observations,
            split=DatasetSplit.TEST,
            freeze_id="test-2026-08",
            run_end=run_end,
        )
        self.assertEqual(rows, [])
        self.assertIn("resolves_inside_safety_window", rejected[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
