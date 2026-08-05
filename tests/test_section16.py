"""Offline tests for Section 16 corroboration of EDGAR officer transitions."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import unittest

from corvus.cli.prepare_section16_review import section16_verification_row
from corvus.models import DatasetSplit, build_rows
from corvus.section16 import (
    Section16Adapter,
    Section16Filing,
    detect_promotion,
    normalize_person_name,
    officer_role_attribute,
    ownership_document_url,
    pair_item_502_with_ownership,
    parse_ownership_document,
    person_names_agree,
)
from corvus.sources import EdgarAdapter, EdgarFilingCandidate, OfficerTransition


FORM3_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0607</schemaVersion>
    <documentType>3</documentType>
    <periodOfReport>2026-07-21</periodOfReport>
    <issuer>
        <issuerCik>0000055242</issuerCik>
        <issuerName>EXAMPLE CORP</issuerName>
        <issuerTradingSymbol>EXC</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0002147996</rptOwnerCik>
            <rptOwnerName>Cole Amanda Marie</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerAddress>
            <rptOwnerStreet1>525 WILLIAM PENN PLACE</rptOwnerStreet1>
            <rptOwnerCity>PITTSBURGH</rptOwnerCity>
            <rptOwnerState>PA</rptOwnerState>
            <rptOwnerZipCode>15219</rptOwnerZipCode>
        </reportingOwnerAddress>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <ownerSignature>
        <signatureName>Michelle R. Keating, attorney-in-fact</signatureName>
        <signatureDate>2026-07-29</signatureDate>
    </ownerSignature>
</ownershipDocument>
"""


def parse(body: bytes = FORM3_XML, **overrides) -> Section16Filing:
    kwargs = {
        "accession_number": "0002147996-26-000001",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/55242/x/primary_doc.xml",
        "document_sha256": hashlib.sha256(body).hexdigest(),
    }
    kwargs.update(overrides)
    return parse_ownership_document(body, **kwargs)


def filing(**overrides) -> Section16Filing:
    base = {
        "form_type": "4",
        "issuer_cik": "0000055242",
        "issuer_name": "EXAMPLE CORP",
        "owner_cik": "0002147996",
        "owner_name": "Cole Amanda Marie",
        "period_of_report": date(2026, 7, 21),
        "is_officer": True,
        "is_director": False,
        "officer_title": "Chief Executive Officer",
        "accession_number": "0002147996-26-000009",
        "filing_url": "https://www.sec.gov/x",
        "document_sha256": "a" * 64,
    }
    base.update(overrides)
    return Section16Filing(**base)


class OwnershipParsingTests(unittest.TestCase):
    def test_parses_structured_event_date_and_title(self):
        parsed = parse()
        self.assertEqual(parsed.form_type, "3")
        self.assertEqual(parsed.period_of_report, date(2026, 7, 21))
        self.assertEqual(parsed.officer_title, "Chief Executive Officer")
        self.assertTrue(parsed.is_officer)
        self.assertEqual(parsed.natural_owner_name, "Amanda Marie Cole")
        self.assertEqual(parsed.role_attribute, "ceo_of")

    def test_retains_no_personal_data_beyond_the_answer(self):
        # The filing carries a home address; nothing but the name may survive.
        serialized = parse().model_dump_json()
        for leaked in ("WILLIAM PENN", "PITTSBURGH", "15219", "Keating"):
            self.assertNotIn(leaked, serialized)

    def test_rejects_officer_without_mandatory_title(self):
        body = FORM3_XML.replace(
            b"<officerTitle>Chief Executive Officer</officerTitle>", b""
        )
        with self.assertRaises(ValueError):
            parse(body)

    def test_rejects_non_ownership_form(self):
        body = FORM3_XML.replace(b"<documentType>3</documentType>",
                                 b"<documentType>5</documentType>")
        with self.assertRaises(ValueError):
            parse(body)

    def test_accepts_a_period_with_a_filing_agent_timezone_offset(self):
        # Seen live: "2026-07-20-05:00". The offset is the agent's timezone, not
        # a time of day, so the calendar date is taken as filed.
        body = FORM3_XML.replace(
            b"<periodOfReport>2026-07-21</periodOfReport>",
            b"<periodOfReport>2026-07-20-05:00</periodOfReport>",
        )
        self.assertEqual(parse(body).period_of_report, date(2026, 7, 20))

    def test_rejects_an_unparseable_period(self):
        body = FORM3_XML.replace(
            b"<periodOfReport>2026-07-21</periodOfReport>",
            b"<periodOfReport>July 21, 2026</periodOfReport>",
        )
        with self.assertRaises(ValueError):
            parse(body)

    def test_event_date_is_day_precision_at_utc_midnight(self):
        parsed = parse()
        self.assertEqual(
            parsed.effective_ts, datetime(2026, 7, 21, tzinfo=timezone.utc)
        )

    def test_raw_xml_url_drops_the_xsl_render_prefix(self):
        self.assertEqual(
            ownership_document_url(
                "0000055242", "0002147996-26-000001", "xslF345X06/primary_doc.xml"
            ),
            "https://www.sec.gov/Archives/edgar/data/55242/"
            "000214799626000001/primary_doc.xml",
        )


class TitleMappingTests(unittest.TestCase):
    def test_maps_top_office_spellings(self):
        for title in (
            "Chief Executive Officer",
            "President and CEO",
            "President and Chief Executive Officer of the Company",
        ):
            self.assertEqual(officer_role_attribute(title), "ceo_of", title)
        for title in ("Chairman of the Board", "Executive Chairman",
                      "Chairman of the Board of Directors",
                      "Board of Directors Chairman"):
            self.assertEqual(officer_role_attribute(title), "chairperson_of", title)

    def test_refuses_co_held_offices(self):
        # "Co-CEO" has no single answer to "who is the CEO", so it cannot be a
        # benchmark row however well it is corroborated.
        for title in ("Co-CEO", "Co-Chief Executive Officer", "Co Chief Executive Officer"):
            self.assertIsNone(officer_role_attribute(title), title)

    def test_refuses_deputy_interim_and_scoped_offices(self):
        for title in (
            "Vice President",
            "Deputy Chief Executive Officer",
            "Interim CEO",
            "Acting Chief Executive Officer",
            "Vice Chairman",
            "Chief Financial Officer",
            "CEO of Acme Europe",
            "President & CEO, Retail Division",
            "",
            None,
        ):
            self.assertIsNone(officer_role_attribute(title), title)


class NameHandlingTests(unittest.TestCase):
    def test_reorders_edgar_surname_first_names(self):
        self.assertEqual(normalize_person_name("Cole Amanda Marie"), "Amanda Marie Cole")
        self.assertEqual(normalize_person_name("Smith Jr John A"), "John A Smith Jr")
        self.assertEqual(normalize_person_name("Cher"), "Cher")

    def test_agreement_survives_initials_and_compound_surnames(self):
        self.assertTrue(person_names_agree("Amanda M. Cole", "Cole Amanda Marie"))
        self.assertTrue(person_names_agree("Maria Garcia Lopez", "Garcia Lopez Maria"))

    def test_agreement_rejects_different_people_and_bare_surnames(self):
        self.assertFalse(person_names_agree("Amanda Cole", "Jane Cole"))
        self.assertFalse(person_names_agree("Cole", "Cole Amanda"))


class PairingTests(unittest.TestCase):
    def candidate(self, **overrides) -> EdgarFilingCandidate:
        base = {
            "cik": "0000055242",
            "entity_name": "EXAMPLE CORP",
            "accession_number": "0000055242-26-000010",
            "filing_date": date(2026, 7, 28),
            "report_date": date(2026, 7, 21),
            "acceptance_ts": datetime(2026, 7, 28, 16, 5, tzinfo=timezone.utc),
            "primary_document": "example-8k.htm",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/55242/a/example-8k.htm",
            "items": ["5.02"],
        }
        base.update(overrides)
        return EdgarFilingCandidate(**base)

    def submissions(self, *rows) -> dict:
        return {
            "0000055242": {
                "filings": {
                    "recent": {
                        "form": [r[0] for r in rows],
                        "filingDate": [r[1] for r in rows],
                        "reportDate": [r[2] for r in rows],
                        "accessionNumber": [r[3] for r in rows],
                        "primaryDocument": ["xslF345X06/primary_doc.xml" for _ in rows],
                    }
                }
            }
        }

    def test_pairs_ownership_filings_inside_the_window(self):
        refs = pair_item_502_with_ownership(
            [self.candidate()],
            self.submissions(
                ("3", "2026-07-29", "2026-07-21", "0002147996-26-000001"),
                ("4", "2026-08-03", "2026-08-01", "0002147996-26-000002"),
            ),
        )
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].period_of_report, date(2026, 7, 21))
        self.assertEqual(
            refs[0].paired_item_502_accessions, ["0000055242-26-000010"]
        )

    def test_excludes_ownership_filings_outside_the_window(self):
        refs = pair_item_502_with_ownership(
            [self.candidate()],
            self.submissions(
                ("3", "2026-01-04", "2026-01-01", "0002147996-26-000003"),
                ("4", "2026-11-30", "2026-11-28", "0002147996-26-000004"),
            ),
        )
        self.assertEqual(refs, [])

    def test_ignores_forms_that_are_not_ownership_filings(self):
        refs = pair_item_502_with_ownership(
            [self.candidate()],
            self.submissions(("8-K", "2026-07-28", "2026-07-21", "x-26-1")),
        )
        self.assertEqual(refs, [])

    def test_window_anchors_on_each_item_502_not_the_earliest(self):
        # Over a multi-month collection an issuer files several Item 5.02s. A
        # Form 3 next to the later one must still pair, and must not claim the
        # unrelated February filing as its evidence.
        february = self.candidate(
            accession_number="0000055242-26-000002", filing_date=date(2026, 2, 10)
        )
        july = self.candidate(
            accession_number="0000055242-26-000010", filing_date=date(2026, 7, 28)
        )
        refs = pair_item_502_with_ownership(
            [february, july],
            self.submissions(
                ("3", "2026-07-29", "2026-07-21", "0002147996-26-000001"),
            ),
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(
            refs[0].paired_item_502_accessions, ["0000055242-26-000010"]
        )

    def test_an_ownership_filing_near_two_candidates_lists_both(self):
        refs = pair_item_502_with_ownership(
            [
                self.candidate(
                    accession_number="0000055242-26-000010",
                    filing_date=date(2026, 7, 28),
                ),
                self.candidate(
                    accession_number="0000055242-26-000011",
                    filing_date=date(2026, 8, 5),
                ),
            ],
            self.submissions(
                ("3", "2026-07-29", "2026-07-21", "0002147996-26-000001"),
            ),
        )
        self.assertEqual(
            refs[0].paired_item_502_accessions,
            ["0000055242-26-000010", "0000055242-26-000011"],
        )

    def test_pairing_handles_a_period_with_a_timezone_offset(self):
        refs = pair_item_502_with_ownership(
            [self.candidate()],
            self.submissions(
                ("3", "2026-07-29", "2026-07-20-05:00", "0002147996-26-000001"),
            ),
        )
        self.assertEqual(refs[0].period_of_report, date(2026, 7, 20))

    def test_window_is_configurable(self):
        refs = pair_item_502_with_ownership(
            [self.candidate()],
            self.submissions(("3", "2026-09-20", "2026-09-15", "0002147996-26-000005")),
            window=(timedelta(days=0), timedelta(days=90)),
        )
        self.assertEqual(len(refs), 1)


class PromotionDetectionTests(unittest.TestCase):
    def test_detects_a_title_change_for_the_same_person_and_issuer(self):
        prior = filing(
            officer_title="Chief Financial Officer", period_of_report=date(2025, 3, 1)
        )
        self.assertEqual(
            detect_promotion(filing(), [prior]), "Chief Financial Officer"
        )

    def test_routine_form_4_with_an_unchanged_title_is_not_a_promotion(self):
        prior = filing(period_of_report=date(2025, 3, 1))
        self.assertIsNone(detect_promotion(filing(), [prior]))

    def test_no_prior_filing_is_not_evidence_of_a_promotion(self):
        self.assertIsNone(detect_promotion(filing(), []))

    def test_another_persons_history_is_not_used(self):
        other = filing(
            owner_cik="0009999999",
            officer_title="Chief Financial Officer",
            period_of_report=date(2025, 3, 1),
        )
        self.assertIsNone(detect_promotion(filing(), [other]))


class CorroborationTests(unittest.TestCase):
    def issuer_event(self):
        candidate = EdgarFilingCandidate(
            cik="0000055242",
            entity_name="EXAMPLE CORP",
            accession_number="0000055242-26-000010",
            filing_date=date(2026, 7, 28),
            report_date=date(2026, 7, 21),
            acceptance_ts=datetime(2026, 7, 28, 16, 5, tzinfo=timezone.utc),
            primary_document="example-8k.htm",
            filing_url="https://www.sec.gov/Archives/edgar/data/55242/a/example-8k.htm",
            items=["5.02"],
        )
        transition = OfficerTransition(
            role_attribute="ceo_of",
            old_value="Prior Chief",
            new_value="Amanda Marie Cole",
            effective_ts=datetime(2026, 7, 21, tzinfo=timezone.utc),
            evidence_excerpt_sha256="b" * 64,
        )
        return EdgarAdapter.emit_fact(candidate, transition)

    def test_issuer_and_reporting_owner_build_an_eligible_row(self):
        owner_event = Section16Adapter.emit_fact(
            parse(), entity_name="EXAMPLE CORP", old_value="Prior Chief"
        )
        rows, rejected = build_rows(
            [self.issuer_event(), owner_event],
            split=DatasetSplit.DEV,
            freeze_id="dev-2026-07",
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].expected, "Amanda Marie Cole")
        self.assertEqual(
            rows[0].metadata["attester_roles"], ["issuer", "reporting_owner"]
        )
        self.assertEqual(
            rows[0].metadata["resolver_ids"], ["edgar-8k", "edgar-section16"]
        )

    def test_disagreeing_effective_dates_are_rejected(self):
        # The 8-K prose and the Section 16 event date must match; a mismatch is
        # exactly the transcription error dual attestation exists to catch.
        owner_event = Section16Adapter.emit_fact(
            parse(
                FORM3_XML.replace(
                    b"<periodOfReport>2026-07-21</periodOfReport>",
                    b"<periodOfReport>2026-07-14</periodOfReport>",
                )
            ),
            entity_name="EXAMPLE CORP",
            old_value="Prior Chief",
        )
        rows, rejected = build_rows(
            [self.issuer_event(), owner_event],
            split=DatasetSplit.DEV,
            freeze_id="dev-2026-07",
        )

        self.assertEqual(rows, [])
        self.assertIn("effective_ts_disagreement", rejected[0]["reasons"])

    def test_refuses_to_emit_for_a_title_that_is_not_a_corvus_attribute(self):
        body = FORM3_XML.replace(
            b"<officerTitle>Chief Executive Officer</officerTitle>",
            b"<officerTitle>Vice President</officerTitle>",
        )
        with self.assertRaises(ValueError):
            Section16Adapter.emit_fact(parse(body), entity_name="EXAMPLE CORP")

    def test_previous_answer_is_recorded_as_unattested(self):
        # The reviewed 8-K supplies old_value so the two observations group;
        # nothing in a Form 3 confirms it, and provenance must say so.
        owner_event = Section16Adapter.emit_fact(
            parse(), entity_name="EXAMPLE CORP", old_value="Prior Chief"
        )
        self.assertIs(owner_event.provenance["old_value_attested"], False)

    def test_reporting_owner_is_the_attester_not_the_issuer(self):
        owner_event = Section16Adapter.emit_fact(
            parse(), entity_name="EXAMPLE CORP"
        )
        self.assertEqual(owner_event.attester_id, "CIK0002147996")
        self.assertEqual(owner_event.attester_role, "reporting_owner")
        self.assertEqual(owner_event.authority_family, "sec")
        self.assertNotEqual(owner_event.attester_id, self.issuer_event().attester_id)


class ReviewQueueTests(unittest.TestCase):
    def row(self, **overrides):
        kwargs = {
            "entity_name": "EXAMPLE CORP",
            "item_502_urls": [
                "https://www.sec.gov/Archives/edgar/data/55242/a/example-8k.htm"
            ],
            "item_502_accessions": ["0000055242-26-000010"],
            "review_priority": "high",
        }
        filing_overrides = overrides.pop("filing", {})
        kwargs.update(overrides)
        return section16_verification_row(filing(**filing_overrides), **kwargs)

    def test_row_is_a_valid_fact_verification_row(self):
        # section16_verification_row validates internally; assert the shape a
        # reviewer and the Braintrust scorer depend on.
        row = self.row()
        self.assertEqual(row["metadata"]["review_stage"], "fact_verification")
        self.assertTrue(row["metadata"]["verification_eligible"])
        self.assertIsNone(row["expected"]["fact_verification"])
        self.assertEqual(
            row["input"]["claim"]["statement"],
            "Amanda Marie Cole became Chief Executive Officer of EXAMPLE CORP, "
            "effective 2026-07-21.",
        )
        self.assertEqual(row["input"]["claim"]["asserted_effective_ts"], "2026-07-21T00:00:00Z")

    def test_both_attesters_are_shown_with_distinct_roles(self):
        evidence = self.row()["input"]["evidence"]
        self.assertEqual(
            [(e["attester_role"], e["source_role"]) for e in evidence],
            [("issuer", "candidate"), ("reporting_owner", "independent_verification")],
        )
        self.assertEqual({e["authority_family"] for e in evidence}, {"sec"})
        self.assertEqual(
            row_attesters := {e["attester_id"] for e in evidence},
            {"CIK0000055242", "CIK0002147996"},
        )
        self.assertEqual(len(row_attesters), 2)

    def test_previous_value_is_left_for_the_reviewer(self):
        row = self.row()
        self.assertIsNone(row["input"]["claim"]["previous_value"])
        self.assertIs(row["metadata"]["old_value_attested"], False)

    def test_blank_check_issuers_are_flagged_not_silently_kept(self):
        row = self.row(entity_name="Osprey Acquisition Corp. III")
        self.assertTrue(row["metadata"]["possible_blank_check_issuer"])
        self.assertIn("possible-blank-check", row["tags"])

    def test_operating_issuer_is_not_flagged(self):
        self.assertFalse(self.row()["metadata"]["possible_blank_check_issuer"])
        self.assertNotIn("possible-blank-check", self.row()["tags"])

    def test_refuses_a_filing_with_no_mappable_office(self):
        with self.assertRaises(ValueError):
            self.row(filing={"officer_title": "Vice President"})

    def test_name_as_filed_is_carried_for_spelling_review(self):
        context = self.row()["input"]["context"]
        self.assertEqual(context["edgar_owner_name_as_filed"], "Cole Amanda Marie")
        self.assertEqual(context["section16_officer_title"], "Chief Executive Officer")

    def test_row_id_is_stable_across_runs(self):
        self.assertEqual(self.row()["id"], self.row()["id"])

    def test_distinct_offices_at_one_issuer_are_distinct_rows(self):
        ceo = self.row()
        chair = self.row(
            filing={"officer_title": "Chairman of the Board", "owner_cik": "0003000000"}
        )
        self.assertNotEqual(ceo["id"], chair["id"])
        self.assertNotEqual(
            ceo["input"]["claim"]["claim_id"], chair["input"]["claim"]["claim_id"]
        )


if __name__ == "__main__":
    unittest.main()
