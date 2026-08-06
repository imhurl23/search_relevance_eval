"""Pin the issuer-side corroboration gates to the filings that motivated them.

Every fixture below is a reduced form of a real 2026-07 EDGAR Item 5.02 filing,
named in its test. That matters more here than in most suites: this module decides
what becomes GOLD ANSWER data, and each gate exists because a specific filing was
either wrongly accepted or wrongly refused without it. A regression would not
raise — it would publish a benchmark row asserting the wrong person holds an
office, and every model would be scored against it.

The precision/recall balance these tests encode was measured by reading all
confirmations from a 490-filing window by hand, not chosen a priori.
"""

import unittest
from datetime import date

from corvus.issuer_corroboration import (IssuerConfirmation, NotCorroborated,
                                         confirm_appointment, entity_tokens,
                                         filing_text, find_name_spans)


def confirm(text, name, attribute, day, issuer=""):
    return confirm_appointment(text, edgar_owner_name=name,
                               role_attribute=attribute, effective_date=day,
                               issuer_name=issuer)


def reason(text, name, attribute, day, issuer=""):
    try:
        confirm(text, name, attribute, day, issuer)
    except NotCorroborated as error:
        return error.reason
    return "CONFIRMED"


# --- fixtures, each reduced from a real filing ------------------------------

# EAGLE BANCORP INC, 0001050441-26-000083. The office is attributed to "Mr.
# Curley" in a LATER sentence than the one carrying his full name, and the
# board-action date (June 29) precedes the effective date (July 6).
EAGLE = (
    "Item 5.02. Departure of Directors or Certain Officers. (d) On June 29, 2026, "
    "the Board of Directors (the \"Board\") of Eagle Bancorp, Inc. (the "
    "\"Company\"), upon the recommendation of the Governance and Nominating "
    "Committee of the Board, appointed Stephen R. Curley to the boards of the "
    "Company and the Company's wholly owned subsidiary EagleBank (the \"Bank\"), "
    "effective July 6, 2026. Mr. Curley's appointment to the boards is in "
    "connection with his previously announced position as President and Chief "
    "Executive Officer of the Company and the Bank, also effective July 6."
)

# Yarrow Bioscience. One sentence appoints five officers at once — the case that
# makes cross-confirmation between them a live risk.
YARROW = (
    "Appointment of Directors and Certain Officers On July 27, 2026, the Board "
    "appointed Rebecca Frey, Pharm.D. as the Company's Chief Executive Officer, "
    "Tyler Zeronda as the Company's Chief Financial Officer, Steven Ryder, M.D. as "
    "the Company's Chief Medical Officer and Rachael Alford, Ph.D. as the "
    "Company's Chief Operating Officer, each to serve at the discretion of the Board."
)

# Columbus Circle Capital Corp II. A real appointment, correctly dated, in the
# sentence after the name — at a different company.
COLUMBUS = (
    "interests corresponding to 243,043 Founder Shares to Kevin Shannon. "
    "Effective June 26, 2026, Gary Quin resigned as Chairman and Chief Executive "
    "Officer of Inflection Point, and Michael Blitzer was appointed as director. "
    "Effective June 26, 2026, Kevin Shannon was appointed as Chief Executive "
    "Officer of Inflection Point."
)

# Meridian3 Industrials Acquisition Corp. The signature block carries a name, an
# office and a date in immediate proximity.
SIGNATURE_ONLY = (
    "Item 5.02. On July 1, 2026, in connection with the IPO, the Board appointed "
    "several directors. SIGNATURE Pursuant to the requirements of the Securities "
    "Exchange Act of 1934, the registrant has duly caused this report to be signed "
    "on its behalf by the undersigned hereunto duly authorized. MERIDIAN3 "
    "INDUSTRIALS ACQUISITION CORP By: /s/ Jeffrey H. Foster Name: Jeffrey H. "
    "Foster Title: Chief Financial Officer Dated: July 1, 2026"
)

# SharonAI Holdings Inc. Section 16's event date is the agreement date; the office
# begins a month later. Mr. Goel is also named earlier, away from the commencement
# clause — which is how a window-scoped check let this through.
SHARONAI = (
    "Item 1.01 On July 22, 2026, SharonAI Holdings Inc. entered into a guarantee "
    "with Anuj Goel. Appointment of Chief Financial Officer On July 22, 2026, "
    "SharonAI Holdings Inc. (the \"Company\") entered into an employment agreement "
    "between the Company's subsidiary and Anuj Goel as a guarantor, pursuant to "
    "which Mr. Goel will serve as Chief Financial Officer of the Company (the "
    "\"Employment Agreement\") commencing August 24, 2026."
)

# Kalaris Therapeutics. A later marked date exists in the filing but belongs to an
# unrelated compensation arrangement, so it must NOT refuse the appointment.
KALARIS = (
    "Appointment of Chief Financial Officer On July 16, 2026, the Board of "
    "Directors of Kalaris Therapeutics, Inc. (the \"Company\") appointed Liisa "
    "Bayko as Chief Financial Officer and Treasurer of the Company, effective as "
    "of July 20, 2026 (the \"Effective Date\"). The Company also adopted an "
    "inducement equity plan, effective as of September 1, 2026."
)

# NATIONAL HEALTH INVESTORS INC. An Item 5.02 about a severance agreement with a
# sitting officer, containing no appointment at all.
SEVERANCE_ONLY = (
    "Item 5.02. On July 1, 2026, National Health Investors, Inc. (the \"Company\") "
    "entered into a Change in Control Severance Agreement with Todd Siefert, the "
    "Company's Chief Financial Officer. The CIC Severance Agreement is effective "
    "as of July 1, 2026."
)

# Fortrea Holdings. An interim office announced next to a permanent one.
INTERIM = (
    "On July 6, 2026, the Board appointed David Smith to act as Interim Chief "
    "Financial Officer and principal financial officer of the Company."
)

# Freenome, Inc. The tightest mention of the appointee sits in a beneficial
# ownership table far from the announcement.
TABLE_THEN_ANNOUNCEMENT = (
    "Directors and Named Executive Officers: Aaron Elliott 278,596 Riley Ennis "
    "3,629,862 Linh H. Le 56,256 Carole Nuechterlein - Peter Kolchinsky - "
    + ("filler text about beneficial ownership percentages. " * 30)
    + "On July 20, 2026, the Board appointed Aaron M. Elliott as the Company's "
    "Chief Executive Officer, effective July 20, 2026."
)


class ConfirmationTest(unittest.TestCase):
    def test_office_attributed_in_a_later_sentence_still_confirms(self):
        # EAGLE BANCORP: the full name and the office are in different sentences,
        # bridged by "Mr. Curley". Refusing this cost a correct row.
        got = confirm(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 7, 6),
                      "EAGLE BANCORP INC")
        self.assertIsInstance(got, IssuerConfirmation)
        self.assertEqual(got.date_basis, "effective_cue")

    def test_the_issuers_spelling_is_captured_as_an_alias(self):
        got = confirm(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 7, 6),
                      "EAGLE BANCORP INC")
        self.assertEqual(got.issuer_spelling, "Stephen R. Curley")

    def test_an_on_date_appointment_confirms_without_the_word_effective(self):
        # YARROW: "On July 27, 2026, the Board appointed ... as Chief Executive
        # Officer". Requiring an effectiveness cue alone refused 15 filings in one
        # month, several of them correct like this one.
        got = confirm(YARROW, "Frey Rebecca", "ceo_of", date(2026, 7, 27))
        self.assertEqual(got.date_basis, "appointment_cue")

    def test_each_officer_in_a_multi_appointment_sentence_maps_to_their_own_office(self):
        self.assertIsInstance(
            confirm(YARROW, "Frey Rebecca", "ceo_of", date(2026, 7, 27)),
            IssuerConfirmation)
        self.assertIsInstance(
            confirm(YARROW, "Zeronda Tyler", "cfo_of", date(2026, 7, 27)),
            IssuerConfirmation)
        self.assertIsInstance(
            confirm(YARROW, "Alford Rachael", "coo_of", date(2026, 7, 27)),
            IssuerConfirmation)

    def test_officers_named_together_do_not_cross_confirm(self):
        # The CEO must not be confirmed as the CFO merely because both phrases sit
        # in one sentence. This is why the NEAREST attributed office is compared
        # rather than searching for the expected one.
        self.assertEqual(
            reason(YARROW, "Frey Rebecca", "cfo_of", date(2026, 7, 27)),
            "another_office_is_closer")
        self.assertEqual(
            reason(YARROW, "Zeronda Tyler", "ceo_of", date(2026, 7, 27)),
            "another_office_is_closer")

    def test_a_confirmation_reached_through_a_later_mention_is_accepted(self):
        # FREENOME: the tightest mention is in an ownership table with no office
        # near it. Checking only that mention refused a correct row.
        got = confirm(TABLE_THEN_ANNOUNCEMENT, "Elliott Aaron Matthew", "ceo_of",
                      date(2026, 7, 20))
        self.assertIsInstance(got, IssuerConfirmation)


class WrongCompanyTest(unittest.TestCase):
    def test_an_office_at_another_company_does_not_confirm(self):
        # COLUMBUS CIRCLE: correctly dated, attributed to the right person, in the
        # same sentence — but the office is at Inflection Point.
        self.assertNotEqual(
            reason(COLUMBUS, "Shannon Kevin George", "ceo_of", date(2026, 6, 26),
                   "Columbus Circle Capital Corp II"),
            "CONFIRMED")

    def test_an_unqualified_office_is_read_as_the_filers_own(self):
        # Filings routinely write "appointed X as Chief Financial Officer" with no
        # referent. In an Item 5.02 that is the filer's office.
        text = ("On July 22, 2026, Sidus Space, Inc. (the \"Company\") appointed "
                "Alan Khalili as Chief Financial Officer, effective July 22, 2026.")
        self.assertIsInstance(
            confirm(text, "Khalili Alan", "cfo_of", date(2026, 7, 22),
                    "Sidus Space Inc."),
            IssuerConfirmation)

    def test_self_referential_and_matching_referents_are_accepted(self):
        for referent, issuer in (("the Company", "OmniAb, Inc."),
                                 ("OmniAb, Inc.", "OmniAb, Inc."),
                                 ("the Bank", "EAGLE BANCORP INC")):
            text = (f"Effective July 13, 2026, Amechi Nwachuku was appointed as "
                    f"the Chief Operating Officer of {referent}.")
            with self.subTest(referent=referent):
                self.assertIsInstance(
                    confirm(text, "Nwachuku Amechi Ekeke", "coo_of",
                            date(2026, 7, 13), issuer),
                    IssuerConfirmation)

    def test_entity_tokens_ignore_corporate_suffixes(self):
        # Without this, every filer ending in "Inc" would match every referent
        # ending in "Inc".
        self.assertEqual(entity_tokens("OmniAb, Inc."), frozenset({"omniab"}))
        self.assertNotIn("inc", entity_tokens("Fortrea Holdings Inc."))
        self.assertFalse(entity_tokens("Holdings Group Inc.")
                         & entity_tokens("Capital Partners LLC"))


class SignatureBlockTest(unittest.TestCase):
    def test_a_signing_officer_is_not_an_appointed_officer(self):
        # MERIDIAN3: "/s/ Jeffrey H. Foster ... Title: Chief Financial Officer
        # Dated: July 1, 2026" satisfies name, office and date proximity, and
        # asserts no appointment.
        self.assertEqual(
            reason(SIGNATURE_ONLY, "FOSTER JEFFREY H", "cfo_of", date(2026, 7, 1)),
            "signature_block_only")


class IncumbencyTest(unittest.TestCase):
    def test_an_office_that_begins_later_is_refused(self):
        # SHARONAI: event date 2026-07-22 is when the agreement was signed; the
        # office begins 2026-08-24. Publishing the earlier date would assert a
        # present-tense answer that was not yet true.
        self.assertEqual(
            reason(SHARONAI, "Goel Anuj", "cfo_of", date(2026, 7, 22),
                   "SharonAI Holdings Inc."),
            "commences_after_event_date")

    def test_a_later_date_in_an_unrelated_sentence_does_not_refuse(self):
        # KALARIS: an inducement plan effective 2026-09-01 says nothing about when
        # the CFO takes office. A document-scoped check refused this row.
        got = confirm(KALARIS, "Bayko Liisa Ann", "cfo_of", date(2026, 7, 20),
                      "Kalaris Therapeutics, Inc.")
        self.assertIsInstance(got, IssuerConfirmation)

    def test_an_interim_office_never_corroborates_a_permanent_one(self):
        # officer_role_attribute refuses "Interim CFO" in the Section 16 title
        # field; the issuer's prose has to be judged by the same rule.
        self.assertNotEqual(
            reason(INTERIM, "Smith David", "cfo_of", date(2026, 7, 6)),
            "CONFIRMED")

    def test_a_filing_that_describes_no_appointment_is_refused(self):
        # NATIONAL HEALTH INVESTORS: Item 5.02 also covers severance and
        # compensation. Without this gate a severance agreement naming the sitting
        # CFO reads as an appointment of that CFO.
        self.assertEqual(
            reason(SEVERANCE_ONLY, "Siefert Todd Michael", "cfo_of",
                   date(2026, 7, 1)),
            "appointment_not_described")


class DateAgreementTest(unittest.TestCase):
    def test_the_board_action_date_does_not_stand_in_for_the_effective_date(self):
        # EAGLE BANCORP states both June 29 (board action) and July 6 (effective).
        # A row dated June 29 must not be confirmable from this filing.
        self.assertEqual(
            reason(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 6, 29),
                   "EAGLE BANCORP INC"),
            "date_not_marked_effective")

    def test_an_off_by_one_date_is_refused(self):
        self.assertEqual(
            reason(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 7, 7),
                   "EAGLE BANCORP INC"),
            "date_not_stated")

    def test_the_right_day_in_the_wrong_year_is_refused(self):
        # A year is always required, which is why the bare "July 6" second
        # reference in this filing cannot satisfy the check.
        self.assertEqual(
            reason(EAGLE, "Curley Stephen Russell", "ceo_of", date(2025, 7, 6),
                   "EAGLE BANCORP INC"),
            "date_not_stated")

    def test_documented_date_spellings_are_all_accepted(self):
        for spelling in ("July 6, 2026", "July 06, 2026", "6 July 2026",
                         "2026-07-06", "7/6/2026"):
            text = (f"On July 1, 2026 the Board appointed Jane A. Roe as Chief "
                    f"Executive Officer of the Company, effective {spelling}.")
            with self.subTest(spelling=spelling):
                self.assertIsInstance(
                    confirm(text, "Roe Jane Alice", "ceo_of", date(2026, 7, 6)),
                    IssuerConfirmation)


class NameMatchingTest(unittest.TestCase):
    def test_a_bare_surname_never_corroborates(self):
        # person_names_agree requires two substantive tokens: "Mr. Curley" appears
        # in filings that never name him.
        self.assertEqual(
            reason(EAGLE, "Curley", "ceo_of", date(2026, 7, 6)),
            "name_not_found")

    def test_a_different_person_does_not_corroborate(self):
        self.assertEqual(
            reason(EAGLE, "Smith Jane Alice", "ceo_of", date(2026, 7, 6)),
            "name_not_found")

    def test_a_middle_name_absent_from_the_filing_is_not_required(self):
        # EDGAR carries the full legal name, filings write an initial. Requiring
        # every token would reject the document that corroborates it.
        self.assertIsInstance(
            confirm(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 7, 6),
                    "EAGLE BANCORP INC"),
            IssuerConfirmation)

    def test_a_nickname_is_a_known_limitation_and_refuses_rather_than_guesses(self):
        # SECURITIZE CORP writes "Billy Miller" where EDGAR has "Miller William
        # Dawson". Nothing deterministic links the two, so the row is refused
        # rather than matched on surname alone.
        text = ("On July 1, 2026, the Board appointed Billy Miller as Chief "
                "Operating Officer of the Company, effective July 1, 2026.")
        self.assertEqual(
            reason(text, "Miller William Dawson", "coo_of", date(2026, 7, 1)),
            "name_not_found")

    def test_honorific_references_require_a_full_name_somewhere(self):
        # The Eagle bridge works only because the full name appears earlier. A
        # filing with nothing but "Mr. Curley" must not corroborate.
        text = ("On July 6, 2026 Mr. Curley was appointed Chief Executive Officer "
                "of the Company, effective July 6, 2026.")
        self.assertEqual(
            reason(text, "Curley Stephen Russell", "ceo_of", date(2026, 7, 6)),
            "name_not_found")
        self.assertEqual(find_name_spans(text, "Curley Stephen Russell"), [])


class FilingTextTest(unittest.TestCase):
    def test_tags_become_spaces_so_table_cells_do_not_fuse(self):
        # Dropping tags outright would produce "CurleyStephen", which no
        # word-boundary match can find.
        text = filing_text(b"<td>Curley</td><td>Stephen</td>")
        self.assertIn("Curley Stephen", text)

    def test_script_and_style_bodies_are_removed(self):
        text = filing_text(
            b"<style>.x{color:red}</style><script>var ceo=1</script><p>Hello</p>")
        self.assertEqual(text, "Hello")

    def test_entities_and_typographic_characters_are_normalized(self):
        text = filing_text(b"Mr.&nbsp;Curley&#8217;s &amp; Co.")
        self.assertEqual(text, "Mr. Curley's & Co.")


class CompliancePropertyTest(unittest.TestCase):
    def test_a_confirmation_carries_no_document_text(self):
        # The source policy excludes source_document_bodies from anything
        # published. Only a hash and offsets may leave this module; the one string
        # field is the appointee's name, which is the published answer anyway.
        got = confirm(EAGLE, "Curley Stephen Russell", "ceo_of", date(2026, 7, 6),
                      "EAGLE BANCORP INC")
        self.assertEqual(len(got.evidence_excerpt_sha256), 64)
        for value in vars(got).values():
            if isinstance(value, str) and len(value) > 64:
                self.fail(f"confirmation carries a long string: {value[:80]!r}")
        self.assertNotIn("Item 5.02", str(got))


if __name__ == "__main__":
    unittest.main()


class CuratedAliasScoringTest(unittest.TestCase):
    """The gold answer is EDGAR's legal name; the web writes the everyday form.

    Corvus-QA publishes the filing's own spelling in metadata["answer_aliases"].
    qa_answer_match ignored that field, so a correct answer using the real-world
    name form scored zero — 7 of 29 rows on the first fresh freeze, and the loss
    fell entirely on the retrieval arms, because those are the only arms that
    return real-world name forms at all. That is a 24-point understatement of
    exactly the effect the study exists to measure.
    """

    ROW = {
        "expected": "Steven Vincent Oroho Jr",
        "metadata": {"answer_aliases": ["Oroho Steven Vincent Jr", "Steven Oroho"]},
    }

    def _score(self, answer):
        import scorers
        got = scorers.qa_answer_match(
            {"question": "Who is the CFO of DLH Holdings Corp.?"},
            {"final_answer": answer, "trajectory": [], "decision_surface": "none",
             "used_searches": 0, "used_clicks": 0},
            self.ROW["expected"], metadata=self.ROW["metadata"])
        return got["score"]

    def test_a_curated_alias_counts_as_a_correct_answer(self):
        self.assertEqual(
            self._score("Steven Oroho is the Chief Financial Officer of DLH."), 1.0)

    def test_the_legal_name_still_counts(self):
        self.assertEqual(self._score("Steven Vincent Oroho Jr"), 1.0)

    def test_a_different_person_still_scores_zero(self):
        # The fix must widen the gold set, not weaken the matcher.
        self.assertEqual(self._score("Kathryn M. JohnBull is the CFO."), 0.0)

    def test_a_bare_surname_still_scores_zero(self):
        self.assertEqual(self._score("Oroho"), 0.0)

    def test_rows_without_curated_aliases_are_unaffected(self):
        import scorers
        got = scorers.qa_answer_match(
            {"question": "q"},
            {"final_answer": "Jane Doe", "trajectory": [],
             "decision_surface": "none", "used_searches": 0, "used_clicks": 0},
            "Jane Doe", metadata={})
        self.assertEqual(got["score"], 1.0)
