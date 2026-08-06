"""Corroborate a Section 16 appointment against the issuer's own Item 5.02 filing.

This closes the last gap that kept Corvus-QA from being buildable without a human
reading every filing. The Section 16 side was already deterministic: Form 3/4 is
structured XML, so the appointee's name, the office, and the "Date of Event
Requiring Statement" are read rather than parsed from prose. The issuer side was
not, because Item 5.02 states the appointment in free text — and
``build_rows(min_attesters=2)`` will not emit a row from the Section 16
observation alone.

The gap is closed by reframing the issuer side as VERIFICATION rather than
extraction. Nothing here reads a fact out of prose. Section 16 has already
supplied a complete candidate — this person, this office, this date — and the only
question asked of the issuer's filing is whether its own text asserts the same
three things. Three independent confirmations must all land in one window:

  1. the appointee's name, matched with the same subset rule
     ``person_names_agree`` uses, so "Stephen R. Curley" corroborates EDGAR's
     "Curley Stephen Russell";
  2. the office, and specifically the office being claimed — a role phrase for a
     DIFFERENT mapped office sitting closer to the name is a rejection, not a
     match, because an 8-K that announces a CEO and a CFO together would
     otherwise cross-confirm them;
  3. the Section 16 event date, rendered in the spellings US filings use.

Confirming the date rather than extracting it is what makes this tractable. The
Eagle Bancorp filing in the July window contains "On June 29, 2026, the Board ...
appointed Stephen R. Curley ... effective July 6, 2026": two dates, and the first
one is the board's action, not the effective date. Any "nearest date" heuristic
picks the wrong one. Asking instead whether the issuer states the date Section 16
already gave has no such failure mode, and it makes the two attesters agree on
``effective_ts`` exactly — which ``build_rows`` requires, since differing
timestamps are an ``effective_ts_disagreement`` rejection.

Compliance: filing bodies are read from local disk and NEVER returned or stored.
The only trace that leaves this module is a SHA-256 of the evidence window plus
its offsets, which is the shape ``OfficerTransition.evidence_excerpt_sha256``
already required. See config/corvus/source_compliance.json, which excludes
``source_document_bodies`` from anything published.
"""

from __future__ import annotations

import hashlib
import html as html_module
import re
from dataclasses import dataclass
from datetime import date

from corvus.section16 import (CEO_TITLE, CFO_TITLE, CHAIR_TITLE, COO_TITLE,
                              _SUFFIXES)

# Role phrase per attribute. Reusing the Section 16 title patterns is deliberate:
# the issuer's prose and the reporting owner's officerTitle must be judged by one
# definition of each office, or the two attesters could "agree" while meaning
# different things.
ROLE_PATTERNS: dict[str, re.Pattern] = {
    "ceo_of": CEO_TITLE,
    "chairperson_of": CHAIR_TITLE,
    "cfo_of": CFO_TITLE,
    "coo_of": COO_TITLE,
}

# The appointee's surname and one given name must occur within this many
# characters of each other to count as one mention. Wide enough for "Stephen R.
# Curley" and for a name split across markup; far too narrow to join two
# different people named in the same sentence.
NAME_SPAN_CHARS = 60

# How far from the name the office and date may sit. Measured against the July
# window: in the Eagle Bancorp filing the office phrase is ~250 characters from
# the name and the date ~110. A much larger window starts joining separate
# paragraphs about separate officers.
EVIDENCE_WINDOW_CHARS = 800

# A role phrase immediately followed by a subsidiary or unit reference is not the
# issuer-level office the question asks about. This is a lighter check than
# section16._is_scoped_below_issuer because prose says "of the Company" routinely
# and that is benign; only an explicit unit keyword disqualifies.
UNIT_TRAILER = re.compile(
    r"\b(?:subsidiary|subsidiaries|division|segment|unit|affiliate)\b",
    re.IGNORECASE,
)
UNIT_TRAILER_CHARS = 60

# officer_role_attribute refuses interim, acting and co-held offices in the
# Section 16 title field. The same refusals have to apply to the issuer's prose or
# the two sides would be judged by different rules: CFO_TITLE matches "Interim
# Chief Financial Officer", and a Fortrea filing announcing an Interim CFO
# alongside the permanent one would let the interim phrase stand in as evidence.
QUALIFIED_OFFICE = re.compile(
    r"\b(?:interim|acting|former|outgoing|deputy|co)[\s-]*$", re.IGNORECASE)
QUALIFIED_LOOKBEHIND_CHARS = 24

# "Chief Executive Officer OF WHAT" — the prose analogue of
# section16._is_scoped_below_issuer. A Columbus Circle Capital Corp II filing says
# "Effective June 26, 2026, Kevin Shannon was appointed as Chief Executive Officer
# of Inflection Point": a real appointment, correctly dated, attributed to the
# right person in the same sentence — at a DIFFERENT company. Only the referent
# distinguishes it from a valid row.
ROLE_REFERENT = re.compile(
    r"\s*(?:and|&|,)?\s*of\s+(?:the\s+)?([A-Za-z][\w'&.-]*(?:\s+[A-Za-z][\w'&.-]*){0,3})",
    re.IGNORECASE,
)
# Referents that just restate the filer. "the Bank" and "the Partnership" appear
# in bank-holding and REIT filings the same way "the Company" does.
BENIGN_REFERENTS = frozenset({
    "company", "registrant", "issuer", "corporation", "corp", "board",
    "directors", "bank", "partnership", "trust", "firm", "organization",
})
# Dropped before comparing an issuer name with a referent, so "OmniAb, Inc."
# matches "OmniAb" and does not match every other filing that ends in "Inc".
_ENTITY_STOPWORDS = frozenset({
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co", "co.",
    "company", "llc", "l.l.c.", "lp", "l.p.", "ltd", "ltd.", "limited", "plc",
    "holding", "holdings", "group", "the", "and", "of", "sa", "nv", "ag",
    "trust", "partners", "capital", "international", "industries", "technologies",
})


def entity_tokens(name: str) -> frozenset[str]:
    """Distinctive words in an issuer name, for referent comparison."""
    cleaned = re.sub(r"[^\w\s]", " ", name).casefold().split()
    return frozenset(t for t in cleaned if len(t) > 2 and t not in _ENTITY_STOPWORDS)


def _referent_is_the_issuer(window: str, role_end: int,
                            issuer_tokens: frozenset[str]) -> bool:
    """Whether an office phrase's "of X" refers to the filer rather than elsewhere.

    An office stated with no referent at all ("appointed Alan Khalili as Chief
    Financial Officer, effective July 27, 2026") is accepted: in an Item 5.02
    filing the unqualified office is the filer's own.
    """
    match = ROLE_REFERENT.match(window, role_end)
    if match is None:
        return True
    referent = match.group(1)
    tokens = [t for t in re.sub(r"[^\w\s]", " ", referent).casefold().split() if t]
    if not tokens:
        return True
    if tokens[0] in BENIGN_REFERENTS:
        return True
    distinctive = entity_tokens(referent)
    if not distinctive:
        # Only stopwords, e.g. "of the Holdings" — nothing to contradict.
        return True
    return bool(distinctive & issuer_tokens)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


class NotCorroborated(Exception):
    """The issuer's filing does not assert what Section 16 asserted.

    Carries a stable ``reason`` code so the rejection ledger reports WHY a
    candidate was dropped. A silent drop and a rejected drop have very different
    meanings for a yield number.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class IssuerConfirmation:
    """What the issuer's own filing independently asserts."""

    role_attribute: str
    effective_date: date
    # The name as the ISSUER spells it. Published as an alias, never as the
    # answer: the answer comes from the structured Section 16 field.
    issuer_spelling: str | None
    evidence_excerpt_sha256: str
    # Offsets into the normalized text, for re-audit against the same document.
    # The text itself is not carried.
    evidence_start: int
    evidence_end: int
    matched_date_form: str
    # "effective_cue" or "appointment_cue" — how the filing marks the date. Kept
    # per row so a later reader can restrict to the stricter basis without
    # rebuilding.
    date_basis: str


def filing_text(body: bytes) -> str:
    """Normalize a filing document to plain text for matching only.

    The result stays in memory. Callers must not persist or publish it.
    """
    raw = body.decode("utf-8", errors="replace")
    # Script and style bodies would otherwise contribute matchable words.
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    # Tags become spaces, not nothing: dropping them outright would fuse
    # "Curley</td><td>Stephen" into one token and break word-boundary matching.
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html_module.unescape(raw)
    # Typographic apostrophes and non-breaking spaces appear throughout EDGAR
    # HTML; normalizing them keeps one spelling of "Mr. Curley's".
    raw = raw.replace(" ", " ").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", raw).strip()


def name_tokens(edgar_owner_name: str) -> tuple[str, list[str]]:
    """Split EDGAR's surname-first ``rptOwnerName`` into (surname, given names).

    EDGAR stores the name surname-first with no delimiter, so the first token is
    the surname. That is the same single-token-surname assumption
    ``normalize_person_name`` documents, and it fails the same way on compound
    surnames — but harmlessly here: the first token of "Garcia Lopez Maria" is
    still part of the surname and still appears in the issuer's text.
    """
    cleaned = re.sub(r"[^\w\s]", " ", edgar_owner_name).casefold()
    tokens = [t for t in cleaned.split() if len(t) > 1 and t.rstrip(".") not in _SUFFIXES]
    if len(tokens) < 2:
        return ("", [])
    return (tokens[0], tokens[1:])


def full_name_spans(text: str, edgar_owner_name: str,
                    *, max_span: int = NAME_SPAN_CHARS) -> list[tuple[int, int]]:
    """Spans holding the surname AND a given name, tightest first.

    Kept separate from honorific references because only these can supply the
    published alias: "Mr. Curley" is a valid corroborating mention but a useless
    spelling of the person's name, and answer_aliases feeds the scorer.
    """
    surname, given = name_tokens(edgar_owner_name)
    if not surname or not given:
        return []
    lowered = text.casefold()
    surname_hits = [m.span() for m in re.finditer(rf"\b{re.escape(surname)}\b", lowered)]
    if not surname_hits:
        return []
    given_hits: list[tuple[int, int]] = []
    for token in set(given):
        given_hits.extend(
            m.span() for m in re.finditer(rf"\b{re.escape(token)}\b", lowered))
    if not given_hits:
        return []
    spans = set()
    for s_start, s_end in surname_hits:
        for g_start, g_end in given_hits:
            start, end = min(s_start, g_start), max(s_end, g_end)
            if end - start <= max_span:
                spans.add((start, end))
    return sorted(spans, key=lambda s: (s[1] - s[0], s[0]))


def find_name_spans(text: str, edgar_owner_name: str,
                    *, max_span: int = NAME_SPAN_CHARS) -> list[tuple[int, int]]:
    """Every span containing the surname and at least one given name, tightest first.

    Requiring a given name as well as the surname is what ``person_names_agree``
    means by "at least two substantive tokens": a bare surname never corroborates,
    because "Mr. Curley" appears in filings that never name him.

    Not every given name is required. EDGAR carries the full legal name while the
    filing usually writes an initial, so demanding all of "Curley Stephen Russell"
    would reject the very document that corroborates it.

    ALL mentions are returned, not just the tightest one. An 8-K names the same
    person several times — in the announcement, in a beneficial-ownership table, in
    a signature block — and the tightest match is frequently not the announcement.
    Checking only it rejected a Freenome filing whose tightest mention sat in an
    ownership table 4,000 characters from the appointment sentence.
    """
    surname, _given = name_tokens(edgar_owner_name)
    spans = full_name_spans(text, edgar_owner_name, max_span=max_span)
    if not spans:
        return []
    lowered = text.casefold()

    # Honorific references count too, but ONLY because a full-name mention was
    # found above. Filings introduce a person once and then refer back: Eagle
    # Bancorp appoints "Stephen R. Curley" in one sentence and states "Mr.
    # Curley's ... position as President and Chief Executive Officer" in the next,
    # so the office is never in the same sentence as the full name. Requiring the
    # full name to exist somewhere keeps the guarantee that a bare surname alone
    # never corroborates.
    honorific = set()
    for match in re.finditer(
            rf"\b(?:mr|ms|mrs|dr|prof|professor)\.?\s+{re.escape(surname)}\b", lowered):
        honorific.add(match.span())

    # Full-name mentions first (tightest first), honorific references after, so a
    # clean "Firstname Lastname" rendering is always preferred as the evidence.
    return (sorted(spans, key=lambda s: (s[1] - s[0], s[0]))
            + sorted(honorific))


# A name in a signature block must never confirm an appointment: the officer who
# signs an 8-K is usually not its subject. "By: /s/ Jeffrey H. Foster ... Title:
# Chief Financial Officer" sits next to both a role phrase and a date, so without
# this guard it reads as a textbook confirmation of an appointment the filing
# never announced.
SIGNATURE_CUE = re.compile(r"(?:/s/|\bBy:|\bSIGNATURE\b)", re.IGNORECASE)
SIGNATURE_LOOKBEHIND_CHARS = 80


def _is_signature_mention(text: str, start: int) -> bool:
    return bool(SIGNATURE_CUE.search(text[max(0, start - SIGNATURE_LOOKBEHIND_CHARS):start]))


def _date_regex(value: date) -> re.Pattern:
    """Spellings of one date that a US filing may use.

    A year is always required. "July 6" alone also appears in filings — the Eagle
    Bancorp text uses it for a second reference — but accepting it would let a
    filing about a different year corroborate this one.
    """
    month = _MONTHS[value.month - 1]
    return re.compile(
        "|".join([
            rf"{month}\s+0?{value.day}\s*,?\s*{value.year}",
            rf"0?{value.day}\s+{month}\s+{value.year}",
            rf"{value.year}-{value.month:02d}-{value.day:02d}",
            rf"0?{value.month}/0?{value.day}/{value.year}",
        ]),
        re.IGNORECASE,
    )


# An Item 5.02 filing routinely carries several dates: the board's action, the
# report date, and the date the appointment takes effect. The Eagle Bancorp text
# reads "On June 29, 2026, the Board ... appointed Stephen R. Curley ... effective
# July 6, 2026" — so merely finding the Section 16 date somewhere in the window
# would also have accepted the board-action date as corroboration. The date must
# be marked as the operative one.
EFFECTIVE_CUE = re.compile(
    r"\b(?:effective(?:\s+as\s+of)?|with\s+effect\s+(?:from|as\s+of)|"
    r"commencing(?:\s+on)?|beginning(?:\s+on)?|"
    r"(?:to|will)\s+(?:begin|commence)(?:\s+on)?|as\s+of)\s*$",
    re.IGNORECASE,
)
EFFECTIVE_CUE_CHARS = 48

# The other way a filing dates an appointment: "On July 27, 2026, the Board
# appointed Rebecca Frey ... as the Company's Chief Executive Officer". The date
# is the event date — exactly what Section 16's "Date of Event Requiring
# Statement" holds — but it is marked by the appointment itself rather than by the
# word "effective". Requiring an effectiveness cue alone rejected 15 filings in
# the July window, several of them clean confirmations like that one.
#
# This is safe to accept because the date under test always comes FROM Section 16.
# The risk an effectiveness cue guards against is a filing that mentions several
# dates and has the wrong one picked; here there is nothing to pick, only to
# confirm. A coincidental appearance of the Section 16 date next to an appointment
# verb, in a filing that also names the right person and the right office within
# the same window, is not a realistic failure mode.
APPOINTMENT_CUE = re.compile(
    r"\b(?:appoint(?:ed|s|ment)?|nam(?:ed|es)|elect(?:ed|s|ion)?|"
    r"promot(?:ed|es|ion)|hir(?:ed|es)|designat(?:ed|es)|"
    r"assum(?:ed|es)|succeed(?:ed|s)?|join(?:ed|s))\b",
    re.IGNORECASE,
)
APPOINTMENT_CUE_CHARS = 160


# Section 16's "Date of Event Requiring Statement" is not always the day the
# appointee takes office. A SharonAI filing pairs event date 2026-07-22 (when the
# employment agreement was signed) with "Mr. Goel will serve as Chief Financial
# Officer of the Company ... commencing August 24, 2026", and Sidus Space pairs
# 2026-07-22 with "effective July 27, 2026". Both would publish a present-tense
# answer that was not yet true.
#
# So when the issuer marks ANY date as the operative one and that date is later
# than the Section 16 event date, the office had not begun and the candidate is
# refused. Dates equal to or earlier than the event date are fine — filings
# routinely announce an appointment before it takes effect.
ANY_MARKED_DATE = re.compile(
    r"(?:effective(?:\s+as\s+of)?|commencing(?:\s+on)?|beginning(?:\s+on)?|"
    r"with\s+effect\s+(?:from|as\s+of)|as\s+of)\s+"
    r"(?:on\s+or\s+about\s+)?"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+0?(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE,
)


def _latest_marked_date(window: str) -> date | None:
    """The latest date the filing marks as operative, if any."""
    latest: date | None = None
    for match in ANY_MARKED_DATE.finditer(window):
        month = _MONTHS.index(match.group(1).capitalize()) + 1
        try:
            found = date(int(match.group(3)), month, int(match.group(2)))
        except ValueError:
            continue
        if latest is None or found > latest:
            latest = found
    return latest


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    """Start and end of the sentence containing ``position``."""
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text, 0, position):
        start = match.end()
    boundary = SENTENCE_BOUNDARY.search(text, position)
    return (start, boundary.start() + 1 if boundary else len(text))


def _office_sentences(text: str, spans: list[tuple[int, int]], role_attribute: str,
                      issuer_tokens: frozenset[str]) -> str:
    """Every sentence where this person is discussed in the office being claimed.

    Scoping the commencement check to these sentences is what makes it both safe
    and effective. Checking only the winning evidence window let a SharonAI filing
    through, because an earlier, weaker mention of Mr. Goel produced a window that
    excluded "commencing August 24, 2026". Checking the whole document instead
    refused Kalaris, Digimarc and OmniAb over later dates belonging to unrelated
    compensation arrangements. The sentences where the office is actually
    attributed to the person are the ones that can speak to when it begins.
    """
    pattern = ROLE_PATTERNS[role_attribute]
    collected: dict[int, str] = {}
    for name_start, name_end in spans:
        for match in pattern.finditer(text):
            if not _role_is_attributed(text, name_start, name_end,
                                       match.start(), match.end()):
                continue
            if not _referent_is_the_issuer(text, match.end(), issuer_tokens):
                continue
            start, end = _sentence_bounds(text, match.start())
            collected[start] = text[start:end]
    return " ".join(collected[key] for key in sorted(collected))


def _find_effective_date(window: str, value: date) -> tuple[str | None, str | None]:
    """(matched text, basis) for the Section 16 date in the window.

    ``basis`` is how the filing marks the date — "effective_cue" or
    "appointment_cue" — or None when the date appears but is marked as neither.
    Presence and marking are returned separately so the ledger can distinguish
    "the issuer never states this date" from "the issuer states it in an unrelated
    role", which are different findings.
    """
    found: str | None = None
    for match in _date_regex(value).finditer(window):
        found = match.group(0)
        preceding = window[max(0, match.start() - EFFECTIVE_CUE_CHARS):match.start()]
        if EFFECTIVE_CUE.search(preceding):
            return (found, "effective_cue")
        nearby = window[max(0, match.start() - APPOINTMENT_CUE_CHARS):
                        match.end() + APPOINTMENT_CUE_CHARS]
        if APPOINTMENT_CUE.search(nearby):
            return (found, "appointment_cue")
    return (found, None)


# A sentence boundary between the name and the office phrase means the office is
# not being attributed to that person. A Columbus Circle filing reads "...interests
# corresponding to 243,043 Founder Shares to Kevin Shannon. Effective June 26,
# 2026, Gary Quin resigned as Chairman and Chief Executive Officer of Inflection
# Point" — the nearest CEO phrase to Shannon's name describes a different person
# leaving a different company, and proximity alone accepted it.
#
# The lookbehind is what makes this usable on filing prose: an unconditional
# `\.\s+[A-Z]` would split "Rebecca Frey, Pharm.D. as the Company's Chief
# Executive Officer" and "Michael Severino, M.D. as Chief Executive Officer",
# both correct confirmations. Requiring the character before the period NOT to be
# an uppercase letter leaves initials and degree abbreviations intact.
SENTENCE_BOUNDARY = re.compile(r"(?<![A-Z])[.;]\s+(?=[A-Z])")

# Generous, because sentence containment is now the real constraint: filings write
# "appointed Andrew (Andy) Reding to serve as Executive Vice President, Chief
# Operating Officer ("COO") of the Company".
ROLE_DISTANCE_CHARS = 220


def _role_is_attributed(window: str, name_start: int, name_end: int,
                        role_start: int, role_end: int) -> bool:
    """Whether the office phrase and the name are in the same sentence, close by."""
    if role_end <= name_start:
        between = window[role_end:name_start]
    elif role_start >= name_end:
        between = window[name_end:role_start]
    else:
        return True
    if len(between) > ROLE_DISTANCE_CHARS:
        return False
    return not SENTENCE_BOUNDARY.search(between)


def _nearest_role(window: str, name_start: int, name_end: int,
                  issuer_tokens: frozenset[str] = frozenset()) -> tuple[str, int] | None:
    """The mapped office attributed to this name, nearest first.

    Returning the nearest office rather than merely searching for the expected one
    is what stops an 8-K that announces several appointments from confirming all
    of them against whichever name happens to be in range. Candidates that are not
    attributable to the name at all are skipped before that comparison, so an
    unrelated office in a neighbouring sentence neither confirms nor blocks.
    """
    following: tuple[str, int] | None = None
    preceding: tuple[str, int] | None = None
    for attribute, pattern in ROLE_PATTERNS.items():
        for match in pattern.finditer(window):
            trailer = window[match.end():match.end() + UNIT_TRAILER_CHARS]
            if UNIT_TRAILER.search(trailer):
                continue
            lead = window[max(0, match.start() - QUALIFIED_LOOKBEHIND_CHARS):match.start()]
            if QUALIFIED_OFFICE.search(lead):
                continue
            if not _role_is_attributed(window, name_start, name_end,
                                       match.start(), match.end()):
                continue
            if not _referent_is_the_issuer(window, match.end(), issuer_tokens):
                continue
            if match.start() >= name_end:
                distance = match.start() - name_end
                if following is None or distance < following[1]:
                    following = (attribute, distance)
            else:
                distance = max(0, name_start - match.end())
                if preceding is None or distance < preceding[1]:
                    preceding = (attribute, distance)
    # An office AFTER the name wins over a nearer one before it. English
    # appointment syntax is "appointed <name> as <office>", so in an enumerated
    # sentence — "appointed Rebecca Frey as Chief Executive Officer, Tyler Zeronda
    # as Chief Financial Officer" — every name but the first has the PREVIOUS
    # person's office sitting closer than its own. Pure proximity confirmed
    # Zeronda as CEO. Preceding offices still count when nothing follows, which is
    # how "Jeffrey F. Brotman, Chief Operating Officer" and "Mr. Curley's position
    # as President and Chief Executive Officer" resolve.
    return following or preceding


# How far a candidate mention got before failing. When no mention confirms, the
# ledger reports the FURTHEST failure across all of them, which is the informative
# one: "the issuer named a different office" says something about the filing,
# while "that particular mention was in a signature block" says nothing.
_REASON_PROGRESS = {
    "name_not_found": 0,
    "signature_block_only": 1,
    "role_phrase_absent": 2,
    "another_office_is_closer": 3,
    "appointment_not_described": 4,
    "date_not_stated": 5,
    "date_not_marked_effective": 6,
    "commences_after_event_date": 7,
}


def confirm_appointment(
    text: str,
    *,
    edgar_owner_name: str,
    role_attribute: str,
    effective_date: date,
    issuer_name: str = "",
    window_chars: int = EVIDENCE_WINDOW_CHARS,
) -> IssuerConfirmation:
    """Confirm the issuer's filing asserts this person, office, and date.

    Every mention of the person is tried, and the first that satisfies all four
    gates confirms. Raises NotCorroborated with a stable reason code otherwise.
    Rejections are reported rather than dropped, because "the issuer never named
    him" and "the issuer named a different office" are different findings and only
    the ledger distinguishes a strict pipeline from a broken one.
    """
    if role_attribute not in ROLE_PATTERNS:
        raise NotCorroborated("unsupported_attribute", role_attribute)

    issuer_tokens = entity_tokens(issuer_name)
    surname, _given = name_tokens(edgar_owner_name)
    spans = find_name_spans(text, edgar_owner_name)
    if not spans:
        raise NotCorroborated("name_not_found", edgar_owner_name)

    # Scoped to the sentences where this person is discussed in this office, and
    # evaluated before any single mention is allowed to confirm — otherwise a weak
    # mention elsewhere in the filing bypasses a gate the appointment paragraph
    # itself fails. See _office_sentences.
    latest_marked = _latest_marked_date(
        _office_sentences(text, spans, role_attribute, issuer_tokens))
    if latest_marked is not None and latest_marked > effective_date:
        raise NotCorroborated("commences_after_event_date", latest_marked.isoformat())

    failure = NotCorroborated("name_not_found", edgar_owner_name)

    def note(reason: str, detail: str = "") -> None:
        nonlocal failure
        if _REASON_PROGRESS[reason] > _REASON_PROGRESS[failure.reason]:
            failure = NotCorroborated(reason, detail)

    for name_start, name_end in spans:
        if _is_signature_mention(text, name_start):
            note("signature_block_only", text[name_start:name_end])
            continue

        start = max(0, name_start - window_chars)
        end = min(len(text), name_end + window_chars)
        window = text[start:end]
        rel_start, rel_end = name_start - start, name_end - start

        nearest = _nearest_role(window, rel_start, rel_end, issuer_tokens)
        if nearest is None:
            note("role_phrase_absent", role_attribute)
            continue
        nearest_attribute, _distance = nearest
        if nearest_attribute != role_attribute:
            # The issuer does discuss an office near this name — a different one.
            note("another_office_is_closer", nearest_attribute)
            continue

        # An Item 5.02 filing covers departures and compensation arrangements as
        # well as appointments. Without this gate a severance agreement naming the
        # sitting CFO reads as an appointment of that CFO.
        if not APPOINTMENT_CUE.search(window):
            note("appointment_not_described", role_attribute)
            continue

        matched_date, basis = _find_effective_date(window, effective_date)
        if matched_date is None:
            note("date_not_stated", effective_date.isoformat())
            continue
        if basis is None:
            note("date_not_marked_effective", matched_date)
            continue


        # The alias comes from the tightest FULL-NAME mention in the document, not
        # from whichever mention happened to confirm. An honorific bridge such as
        # Eagle Bancorp's "Mr. Curley" corroborates the office correctly but would
        # publish "Mr. Curley" as a spelling of the answer.
        #
        # It is recorded only when the issuer writes the name in natural order: a
        # surname-first table cell ("Curley Stephen") is a correct match and a
        # misleading alias, and answer_aliases is used for scoring.
        issuer_spelling = None
        for full_start, full_end in full_name_spans(text, edgar_owner_name):
            candidate = text[full_start:full_end]
            if not candidate.casefold().startswith(surname):
                issuer_spelling = candidate
                break

        return IssuerConfirmation(
            role_attribute=role_attribute,
            effective_date=effective_date,
            issuer_spelling=issuer_spelling,
            evidence_excerpt_sha256=hashlib.sha256(window.encode("utf-8")).hexdigest(),
            evidence_start=start,
            evidence_end=end,
            matched_date_form=matched_date,
            date_basis=basis,
        )

    raise failure
