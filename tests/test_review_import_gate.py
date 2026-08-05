"""The import gate that pre-v2 review artifacts must not get past.

Two v1 queues reached Braintrust as unlabelable rows because the importer
checked only for source text, never for schema conformance. These tests pin the
gate that closes that path.
"""

import unittest

from corvus.cli.import_review_dataset import STAGE_EXPECTED_MARKER, validate_rows
from corvus.cli.prepare_claim_review import edgar_preparation_row, sports_verification_row
from corvus.review_schema import (
    ClaimPreparationExpected,
    ClaimPreparationInput,
    FactVerificationExpected,
    FactVerificationInput,
    braintrust_schemas,
)


EDGAR_CANDIDATE = {
    "cik": "0000055242",
    "entity_name": "EXAMPLE CORP",
    "accession_number": "0000055242-26-000010",
    "filing_date": "2026-07-28",
    "filing_url": "https://www.sec.gov/Archives/edgar/data/55242/a/example-8k.htm",
    "document_sha256": "c" * 64,
    "review_priority": "high",
    "strong_transition_role_signals": ["ceo"],
}

SPORTS_RESULT = {
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

# Shape of the pre-v2 rows that were actually uploaded: no schema_version, no
# review_stage, flat input.
V1_ROW = {
    "id": "legacy-1",
    "input": {
        "task": "Review this filing.",
        "cik": "0000055242",
        "official_sec_url": "https://www.sec.gov/Archives/edgar/data/55242/a/x.htm",
        "instructions": ["Open the filing."],
    },
    "expected": {},
    "metadata": {
        "dataset": "Corvus-QA",
        "workflow": "edgar_human_review",
        "contains_source_text": False,
    },
    "tags": ["corvus"],
}


class ImportGateTests(unittest.TestCase):
    def test_refuses_a_pre_v2_row_and_says_how_to_fix_it(self):
        with self.assertRaises(ValueError) as caught:
            validate_rows([V1_ROW], source="legacy.jsonl")
        message = str(caught.exception)
        self.assertIn("review_stage", message)
        self.assertIn("prepare_claim_review", message)

    def test_accepts_a_claim_preparation_queue(self):
        rows = [edgar_preparation_row(EDGAR_CANDIDATE)]
        self.assertEqual(validate_rows(rows, source="prep.jsonl"), "claim_preparation")

    def test_accepts_a_fact_verification_queue(self):
        rows = [sports_verification_row(SPORTS_RESULT)]
        self.assertEqual(validate_rows(rows, source="fact.jsonl"), "fact_verification")

    def test_refuses_a_queue_that_mixes_stages(self):
        rows = [edgar_preparation_row(EDGAR_CANDIDATE), sports_verification_row(SPORTS_RESULT)]
        with self.assertRaises(ValueError) as caught:
            validate_rows(rows, source="mixed.jsonl")
        self.assertIn("mixes review stages", str(caught.exception))

    def test_refuses_a_row_that_admits_source_text(self):
        row = edgar_preparation_row(EDGAR_CANDIDATE)
        row["metadata"]["contains_source_text"] = True
        with self.assertRaises(ValueError):
            validate_rows([row], source="leaky.jsonl")

    def test_refuses_a_row_carrying_copied_prose(self):
        row = edgar_preparation_row(EDGAR_CANDIDATE)
        row["metadata"]["evidence_excerpt"] = "Mr. Example resigned effective..."
        with self.assertRaises(ValueError):
            validate_rows([row], source="excerpt.jsonl")

    def test_row_index_is_reported_so_a_bad_row_can_be_found(self):
        rows = [edgar_preparation_row(EDGAR_CANDIDATE), dict(V1_ROW)]
        with self.assertRaises(ValueError) as caught:
            validate_rows(rows, source="queue.jsonl")
        self.assertIn("queue.jsonl:2", str(caught.exception))

    def test_stage_markers_distinguish_the_two_schemas(self):
        prep = braintrust_schemas(ClaimPreparationInput, ClaimPreparationExpected)
        fact = braintrust_schemas(FactVerificationInput, FactVerificationExpected)
        self.assertIn(
            STAGE_EXPECTED_MARKER["claim_preparation"], prep["expected"]["properties"]
        )
        self.assertIn(
            STAGE_EXPECTED_MARKER["fact_verification"], fact["expected"]["properties"]
        )
        # The marker must not appear in the other stage's schema, or the
        # importer's wrong-schema check would never fire.
        self.assertNotIn(
            STAGE_EXPECTED_MARKER["fact_verification"], prep["expected"]["properties"]
        )
        self.assertNotIn(
            STAGE_EXPECTED_MARKER["claim_preparation"], fact["expected"]["properties"]
        )


if __name__ == "__main__":
    unittest.main()
