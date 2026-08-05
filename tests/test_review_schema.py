from __future__ import annotations

import unittest

from pydantic import ValidationError

from corvus.cli.prepare_claim_review import (
    edgar_preparation_row,
    sports_verification_row,
)
from corvus.review_schema import (
    ClaimPreparationExpected,
    ClaimPreparationInput,
    FactVerificationExpected,
    FactVerificationInput,
    assert_metadata_only,
    braintrust_schemas,
    validate_review_row,
)


class ReviewSchemaTests(unittest.TestCase):
    def test_verification_requires_an_explicit_claim(self):
        with self.assertRaises(ValidationError):
            FactVerificationInput.model_validate(
                {
                    "schema_version": "corvus.review.v2",
                    "task": "Verify whether the stated atomic claim is true.",
                    "evidence": [],
                    "instructions": ["Verify it."],
                }
            )

    def test_verification_claim_is_strict_and_atomic(self):
        row = FactVerificationInput.model_validate(
            {
                "schema_version": "corvus.review.v2",
                "task": "Verify whether the stated atomic claim is true.",
                "claim": {
                    "claim_id": "claim-1",
                    "statement": "The final score of Alpha vs Beta was 2–1.",
                    "subject_id": "event-1",
                    "subject_name": "Alpha vs Beta",
                    "subject_type": "sports_event",
                    "predicate": "final_score",
                    "object_value": "Alpha 2–1 Beta",
                    "time_basis": "Completion time pending reviewer confirmation.",
                },
                "evidence": [
                    {
                        "url": "https://example.com/event/1",
                        "source_role": "candidate",
                        "compliance_source_id": "fixture",
                        "authority_family": "fixture",
                    }
                ],
                "instructions": ["Verify every material part."],
            }
        )
        self.assertEqual(row.claim.predicate, "final_score")

    def test_preparation_is_not_verification(self):
        row = ClaimPreparationInput.model_validate(
            {
                "schema_version": "corvus.review.v2",
                "task": "Prepare one explicit atomic claim for later verification.",
                "source_kind": "sec_8k_item_5_02",
                "source_record_id": "accession-1",
                "source_urls": ["https://www.sec.gov/example"],
                "instructions": ["Draft one claim."],
            }
        )
        self.assertEqual(row.source_kind, "sec_8k_item_5_02")

    def test_braintrust_schemas_enforce_input_and_expected(self):
        schemas = braintrust_schemas(
            ClaimPreparationInput, ClaimPreparationExpected
        )
        self.assertTrue(schemas["input"]["enforce"])
        self.assertTrue(schemas["expected"]["enforce"])
        verification = braintrust_schemas(
            FactVerificationInput, FactVerificationExpected
        )
        self.assertTrue(verification["input"]["enforce"])
        self.assertTrue(verification["expected"]["enforce"])

    def test_metadata_only_gate_rejects_source_excerpt(self):
        with self.assertRaisesRegex(ValueError, "forbidden source-content field"):
            assert_metadata_only({"evidence_excerpt": "copied source text"})

    def test_raw_edgar_candidate_goes_to_preparation(self):
        row = edgar_preparation_row(
            {
                "cik": "0000000001",
                "accession_number": "0000000001-26-000001",
                "entity_name": "Example Corp",
                "filing_url": "https://www.sec.gov/Archives/example.htm",
                "filing_date": "2026-07-30",
                "strong_transition_role_signals": ["ceo"],
                "review_priority": "high",
                "document_sha256": "a" * 64,
            }
        )
        self.assertEqual(row["metadata"]["review_stage"], "claim_preparation")
        self.assertFalse(row["metadata"]["verification_eligible"])

    def test_raw_sports_result_has_atomic_verification_claim(self):
        row = sports_verification_row(
            {
                "source_event_id": "event-1",
                "sport": "Soccer",
                "competition": "Example League",
                "season": "2026",
                "event_start_ts": "2026-07-30T19:00:00Z",
                "observed_ts": "2026-07-30T22:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 2,
                "away_score": 1,
                "source_url": "https://example.com/event/1",
                "source_type": "fixture_completed_match",
                "resolver_id": "fixture",
                "authority_family": "fixture",
                "compliance_source_id": "fixture",
                "license": None,
                "attribution": "Fixture data.",
            }
        )
        self.assertEqual(row["metadata"]["review_stage"], "fact_verification")
        self.assertEqual(
            row["input"]["claim"]["statement"],
            "The final score of Alpha vs Beta was 2–1.",
        )

    def test_evidence_without_attester_fields_still_validates(self):
        # attester_id/attester_role were added for Section 16 corroboration.
        # Sports and news evidence predates them and must keep validating, so
        # the fields stay optional and absent rather than defaulted to a value.
        row = sports_verification_row(
            {
                "source_event_id": "event-1",
                "sport": "Soccer",
                "competition": "Example League",
                "season": "2026",
                "event_start_ts": "2026-07-30T19:00:00Z",
                "observed_ts": "2026-07-30T22:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 2,
                "away_score": 1,
                "source_url": "https://example.com/event/1",
                "source_type": "fixture_completed_match",
                "resolver_id": "fixture",
                "authority_family": "fixture",
                "compliance_source_id": "fixture",
                "license": None,
                "attribution": "Fixture data.",
            }
        )
        evidence = row["input"]["evidence"][0]
        self.assertIsNone(evidence["attester_id"])
        self.assertIsNone(evidence["attester_role"])
        validate_review_row(row, stage="fact_verification")


if __name__ == "__main__":
    unittest.main()
