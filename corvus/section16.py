"""Section 16 (Forms 3/4) corroboration for EDGAR officer transitions.

An Item 5.02 Form 8-K is the issuer's account of an officer change.  The
incoming officer's own Section 16 filing is a second, separately liable account
of the same event, and unlike the 8-K it is structured:

``periodOfReport``
    "Date of Event Requiring Statement" — the date the person actually became
    an officer or director.  This is the effective timestamp the 8-K only
    states in prose.
``reportingOwnerRelationship``
    ``isOfficer``/``isDirector`` flags plus ``officerTitle``, which is mandatory
    whenever ``isOfficer`` is set.

Form 3 is the initial statement, filed within ten days of becoming an officer or
director, and so covers external hires.  An internal promotion files no new
Form 3 — the person is already a reporting owner — so their next Form 4 carries
the updated ``officerTitle`` instead, and the promotion is detected by diffing
that title against the same person's prior filings for the same issuer.

Neither the reporting owner's address nor any other personal detail beyond the
name that answers the question is retained; see ``parse_ownership_document``.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, field_validator

from corvus.models import FactEvent
from corvus.sources import SEC_ARCHIVES, EdgarFilingCandidate, PolicyHttpClient


# Filed within 10 days of the event, but the 8-K may lag the event too, and
# amendments arrive later. A generous window costs only candidate pairs that
# the title and date checks then reject.
DEFAULT_PAIRING_WINDOW = (timedelta(days=14), timedelta(days=45))

OWNERSHIP_FORMS = ("3", "4")

# Section 16 titles are free text. Only map spellings that unambiguously denote
# the role the question asks about; anything else stays None so a deputy or
# divisional title never corroborates a top-office transition.
CEO_TITLE = re.compile(
    r"(?<!\bdeputy\s)(?<!\bvice\s)(?<!\bassistant\s)"
    r"\b(?:chief\s+executive\s+officer|chief\s+executive|c\.?e\.?o\.?)\b",
    re.IGNORECASE,
)
CHAIR_TITLE = re.compile(
    r"(?<!\bvice\s)(?<!\bdeputy\s)(?<!\bassistant\s)"
    r"\b(?:chair(?:person|man|woman)?)\b(?:\s+of\s+the\s+board)?",
    re.IGNORECASE,
)
# CFO and COO are mapped as well as CEO and chair. This is a yield decision with
# a measured basis: across one month's paired Form 3s, 28 filings carried a CEO
# title and 2 a chair title, while 41 carried a CFO title and 11 a COO title. A
# CEO-and-chair-only mapping discards roughly three quarters of the corroborated
# transitions in the window, which is what held the dev split below the size a
# freshness contrast needs.
#
# These are issuer-level named offices with the same properties that made CEO
# usable: a single incumbent, an unambiguous title, and a Section 16 filing by
# the appointee. The deputy/vice/assistant lookbehinds and the interim/acting/co-
# refusals in officer_role_attribute apply identically.
CFO_TITLE = re.compile(
    r"(?<!\bdeputy\s)(?<!\bvice\s)(?<!\bassistant\s)"
    r"\b(?:chief\s+financial\s+officer|c\.?f\.?o\.?)\b",
    re.IGNORECASE,
)
COO_TITLE = re.compile(
    r"(?<!\bdeputy\s)(?<!\bvice\s)(?<!\bassistant\s)"
    r"\b(?:chief\s+operat(?:ing|ions)\s+officer|c\.?o\.?o\.?)\b",
    re.IGNORECASE,
)
# `President` is deliberately NOT mapped. In the same window it would add only
# four filings, and its common spellings are ambiguous in a way the other titles
# are not: "President, CARFAX" and "President, CSE" name business units rather
# than the issuer, and neither is caught by _is_scoped_below_issuer because
# neither uses "of X" or a division keyword. The safe variants ("President and
# Chief Executive Officer") already map through CEO_TITLE, which is checked
# first, so nothing is lost that matters.
# "CEO of Acme Europe" and "President & CEO, Retail Division" are subsidiary or
# divisional offices, not the issuer-level office the question asks about.
# Phrases that merely restate the issuer are benign and stripped before the
# residual check, so "Chairman of the Board of Directors" still qualifies.
BENIGN_SCOPE = re.compile(
    r"\bof\s+(?:the\s+)?(?:board(?:\s+of\s+directors)?|directors|company|issuer|registrant|corporation)\b",
    re.IGNORECASE,
)
RESIDUAL_SCOPE = re.compile(r"\bof\s+\S", re.IGNORECASE)
DIVISION_SCOPE = re.compile(
    r"[,\-]\s*[\w&.\- ]*"
    r"\b(?:division|segment|subsidiary|group|unit|region|bank|holdings?)\b",
    re.IGNORECASE,
)


def _is_scoped_below_issuer(text: str) -> bool:
    return bool(RESIDUAL_SCOPE.search(BENIGN_SCOPE.sub(" ", text))) or bool(
        DIVISION_SCOPE.search(text)
    )

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "md", "phd", "esq"}


def normalize_person_name(raw: str) -> str:
    """Convert EDGAR's ``Last First Middle`` owner name to natural order.

    EDGAR stores ``rptOwnerName`` surname-first with no delimiter, so
    "Cole Amanda Marie" and "Smith Jr John" both need reordering before they can
    be compared with a name written out in a filing.

    The format is lossy for compound surnames: "Garcia Lopez Maria" could be
    Maria Garcia Lopez or Lopez Maria Garcia, and nothing in the filing
    disambiguates it.  This function assumes a single-token surname, which is
    right for the common case and wrong for the rest — so agreement between
    attesters is checked with :func:`person_agreement_key`, which is
    order-insensitive and therefore unaffected by a wrong guess here.
    """
    cleaned = re.sub(r"\s+", " ", raw.replace(",", " ")).strip()
    if not cleaned:
        raise ValueError("reporting owner name must not be blank")
    parts = cleaned.split(" ")
    if len(parts) == 1:
        return parts[0]
    suffixes = [p for p in parts if p.casefold().rstrip(".") in _SUFFIXES]
    core = [p for p in parts if p not in suffixes]
    if len(core) < 2:
        return " ".join(parts)
    surname, *given = core
    return " ".join([*given, surname, *suffixes])


def person_agreement_key(name: str) -> tuple[str, ...]:
    """Order- and initial-insensitive key for comparing two spellings of a name.

    Middle names and initials are dropped rather than compared, because the 8-K
    prose and the Section 16 record disagree about them routinely ("Amanda M.
    Cole" versus "Cole Amanda Marie") without disagreeing about the person.
    Word order is discarded so a compound surname cannot cause a false reject.

    This deliberately makes two different people with the same first and last
    name indistinguishable; within a single issuer over a 60-day window that
    collision is not a realistic failure mode, but it is the reason this key is
    used for corroboration only and never as the published answer.
    """
    cleaned = re.sub(r"[^\w\s]", " ", name).casefold()
    tokens = [t for t in cleaned.split() if t and t.rstrip(".") not in _SUFFIXES]
    substantive = [t for t in tokens if len(t) > 1]
    return tuple(sorted(substantive))


def person_names_agree(left: str, right: str) -> bool:
    """Whether two spellings plausibly name the same person.

    One name's tokens must be a subset of the other's, so "Amanda M. Cole"
    agrees with "Cole Amanda Marie" (the initial drops out, "Marie" is extra on
    one side only) while "Amanda Cole" and "Jane Cole" do not.  At least two
    substantive tokens are required, so a bare surname never corroborates.
    """
    a, b = set(person_agreement_key(left)), set(person_agreement_key(right))
    if len(a) < 2 or len(b) < 2:
        return False
    return a <= b or b <= a


def officer_role_attribute(title: str | None) -> str | None:
    """Map a Section 16 ``officerTitle`` to a Corvus attribute, or None.

    Returns None for any title that is scoped to a subsidiary or division, and
    for deputy/vice/interim variants, because those are different offices from
    the one the benchmark question asks about.
    """
    if not title:
        return None
    text = re.sub(r"\s+", " ", title).strip()
    if not text:
        return None
    if re.search(r"\b(?:interim|acting|co[- ]|former|outgoing)\b", text, re.IGNORECASE):
        return None
    if _is_scoped_below_issuer(text):
        return None
    # Order is significant: a combined title like "President & CEO" or
    # "EVP, CFO & Treasurer" names more than one office, and the FIRST match wins.
    # CEO is checked first so a chief executive who is also chair or CFO is
    # recorded as the chief executive, which is the office the question asks about.
    if CEO_TITLE.search(text):
        return "ceo_of"
    # A board chair is often reported as a director rather than an officer. The
    # relationship flags are not consulted: a director-only filing carries no
    # officerTitle and has already returned None above, so the title alone is
    # what distinguishes the office.
    if CHAIR_TITLE.search(text):
        return "chairperson_of"
    if CFO_TITLE.search(text):
        return "cfo_of"
    if COO_TITLE.search(text):
        return "coo_of"
    return None


class Section16Filing(BaseModel):
    """One parsed ownership document, stripped of non-answer personal data."""

    model_config = ConfigDict(extra="forbid")

    form_type: str = Field(pattern=r"^[34](?:/A)?$")
    issuer_cik: str = Field(min_length=1)
    issuer_name: str = Field(min_length=1)
    owner_cik: str = Field(min_length=1)
    owner_name: str = Field(min_length=1)
    period_of_report: date
    is_officer: bool
    is_director: bool
    officer_title: str | None = None
    accession_number: str = Field(min_length=1)
    filing_url: str = Field(min_length=1)
    document_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("issuer_cik", "owner_cik")
    @classmethod
    def pad_cik(cls, value: str) -> str:
        return value.strip().zfill(10)

    @property
    def natural_owner_name(self) -> str:
        return normalize_person_name(self.owner_name)

    @property
    def role_attribute(self) -> str | None:
        return officer_role_attribute(self.officer_title)

    @property
    def effective_ts(self) -> datetime:
        """Event date at UTC midnight.

        Section 16 reports a date, not a time. Midnight is a declared
        convention, not a measurement — recency rungs finer than a day must not
        be read off a Section 16-corroborated row.
        """
        return datetime.combine(self.period_of_report, time.min, tzinfo=timezone.utc)


def _text(node: ElementTree.Element | None) -> str | None:
    if node is None:
        return None
    # Ownership XML wraps some fields in a <value> child.
    value = node.find("value")
    source = value if value is not None else node
    return (source.text or "").strip() or None


def _flag(root: ElementTree.Element, path: str) -> bool:
    raw = _text(root.find(path))
    return raw in {"1", "true", "TRUE", "True", "Y", "y"}


# Some filing agents append a local offset to the calendar date, e.g.
# "2026-07-20-05:00". The offset describes the agent's timezone, not a time of
# day, so the date is taken verbatim and the offset discarded.
_PERIOD_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_period_of_report(raw: str) -> date:
    match = _PERIOD_DATE.match(raw.strip())
    if not match:
        raise ValueError(f"unparseable periodOfReport {raw!r}")
    return date.fromisoformat(match.group(1))


def parse_ownership_document(
    body: bytes,
    *,
    accession_number: str,
    filing_url: str,
    document_sha256: str,
) -> Section16Filing:
    """Parse a Form 3/4 ownership XML into the fields Corvus is allowed to keep.

    ``reportingOwnerAddress`` and the signature block are deliberately never
    read: the source policy permits derived facts only, and the officer's home
    address is personal information the benchmark question does not need.
    """
    root = ElementTree.fromstring(body)
    form_type = _text(root.find("documentType"))
    if form_type not in {"3", "4", "3/A", "4/A"}:
        raise ValueError(f"{accession_number}: unsupported ownership form {form_type!r}")
    period = _text(root.find("periodOfReport"))
    if not period:
        raise ValueError(f"{accession_number}: missing periodOfReport")
    owner = root.find("reportingOwner")
    if owner is None:
        raise ValueError(f"{accession_number}: missing reportingOwner")
    relationship = owner.find("reportingOwnerRelationship")
    if relationship is None:
        raise ValueError(f"{accession_number}: missing reportingOwnerRelationship")

    is_officer = _flag(relationship, "isOfficer")
    officer_title = _text(relationship.find("officerTitle"))
    if is_officer and not officer_title:
        raise ValueError(
            f"{accession_number}: officerTitle is mandatory when isOfficer is set"
        )
    return Section16Filing(
        form_type=form_type,
        issuer_cik=_text(root.find("issuer/issuerCik")) or "",
        issuer_name=_text(root.find("issuer/issuerName")) or "",
        owner_cik=_text(owner.find("reportingOwnerId/rptOwnerCik")) or "",
        owner_name=_text(owner.find("reportingOwnerId/rptOwnerName")) or "",
        period_of_report=parse_period_of_report(period),
        is_officer=is_officer,
        is_director=_flag(relationship, "isDirector"),
        officer_title=officer_title,
        accession_number=accession_number,
        filing_url=filing_url,
        document_sha256=document_sha256,
    )


class OwnershipFilingRef(BaseModel):
    """A Form 3/4 located in submissions metadata, before its XML is fetched."""

    model_config = ConfigDict(extra="forbid")

    issuer_cik: str
    form_type: str
    accession_number: str
    filing_date: date
    period_of_report: date | None
    primary_document: str
    filing_url: str
    paired_item_502_accessions: list[str] = Field(default_factory=list)


def ownership_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Raw ownership XML URL.

    ``primaryDocument`` in submissions metadata points at the XSL-rendered view
    (``xslF345X06/primary_doc.xml``); the machine-readable original sits at the
    same accession path without that prefix.
    """
    return SEC_ARCHIVES.format(
        cik=int(cik),
        accession=accession_number.replace("-", ""),
        document=primary_document.split("/")[-1],
    )


def pair_item_502_with_ownership(
    candidates: Iterable[EdgarFilingCandidate],
    submissions: dict[str, dict[str, Any]],
    *,
    window: tuple[timedelta, timedelta] = DEFAULT_PAIRING_WINDOW,
) -> list[OwnershipFilingRef]:
    """Find Form 3/4 filings near each issuer's Item 5.02 filings.

    ``submissions`` maps a zero-padded CIK to that filer's submissions JSON, so
    pairing runs entirely against the bulk archive already on disk — no
    additional SEC requests are spent narrowing the candidate set.

    The window is anchored on **each** Item 5.02 filing, not on the issuer's
    earliest one.  Over a multi-month collection an issuer commonly files
    several, and anchoring on the earliest would both miss ownership filings
    near the later ones and attach every one of the issuer's candidates to each
    reference.  ``paired_item_502_accessions`` therefore lists only the
    candidates this ownership filing is actually near.
    """
    before, after = window
    by_cik: dict[str, list[EdgarFilingCandidate]] = {}
    for candidate in candidates:
        by_cik.setdefault(candidate.cik.zfill(10), []).append(candidate)

    refs: list[OwnershipFilingRef] = []
    for cik, group in sorted(by_cik.items()):
        payload = submissions.get(cik)
        if not payload:
            continue
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for index, form in enumerate(forms):
            if form not in OWNERSHIP_FORMS:
                continue
            filed = date.fromisoformat(recent["filingDate"][index])
            near = sorted(
                item.accession_number
                for item in group
                if item.filing_date - before <= filed <= item.filing_date + after
            )
            if not near:
                continue
            report_raw = recent.get("reportDate", [])[index] or ""
            accession = recent["accessionNumber"][index]
            primary = recent["primaryDocument"][index]
            refs.append(
                OwnershipFilingRef(
                    issuer_cik=cik,
                    form_type=form,
                    accession_number=accession,
                    filing_date=filed,
                    period_of_report=(
                        parse_period_of_report(report_raw) if report_raw else None
                    ),
                    primary_document=primary,
                    filing_url=ownership_document_url(cik, accession, primary),
                    paired_item_502_accessions=near,
                )
            )
    return refs


def iter_submissions(archive_path: Path, ciks: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Read only the requested filers out of the bulk submissions archive."""
    wanted = {cik.zfill(10) for cik in ciks}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for cik in sorted(wanted):
            member = f"CIK{cik}.json"
            if member in names:
                with archive.open(member) as source:
                    yield cik, json.load(source)


def detect_promotion(
    current: Section16Filing, priors: Iterable[Section16Filing]
) -> str | None:
    """Return the prior title when a Form 4 shows a title change, else None.

    A Form 4 is routine and most carry an unchanged title.  Only a filing whose
    role attribute differs from the same person's most recent earlier filing for
    the same issuer evidences a promotion, and only then is it corroboration of
    a transition rather than of the status quo.
    """
    if current.role_attribute is None:
        return None
    earlier = sorted(
        (
            p
            for p in priors
            if p.owner_cik == current.owner_cik
            and p.issuer_cik == current.issuer_cik
            and p.period_of_report < current.period_of_report
        ),
        key=lambda p: p.period_of_report,
    )
    if not earlier:
        return None
    previous = earlier[-1]
    if previous.role_attribute == current.role_attribute:
        return None
    return previous.officer_title


class Section16Adapter:
    """Fetch and interpret Section 16 filings as officer-transition attesters."""

    resolver_id = "edgar-section16"

    def __init__(self, client: PolicyHttpClient):
        self.client = client

    def fetch_ownership_documents(
        self,
        refs: Iterable[OwnershipFilingRef],
        *,
        output_dir: Path,
        max_document_bytes: int = 5 * 1024 * 1024,
    ) -> tuple[list[Section16Filing], list[dict[str, Any]]]:
        """Download each ownership XML once and parse it; report per-ref errors."""
        parsed: list[Section16Filing] = []
        failures: list[dict[str, Any]] = []
        seen: dict[str, OwnershipFilingRef] = {}
        for ref in refs:
            seen.setdefault(ref.filing_url, ref)
        for index, ref in enumerate(
            sorted(seen.values(), key=lambda item: item.filing_url), start=1
        ):
            destination = (
                output_dir / ref.issuer_cik / f"{ref.accession_number}_ownership.xml"
            )
            try:
                if destination.is_file():
                    body = destination.read_bytes()
                    digest = hashlib.sha256(body).hexdigest()
                else:
                    _size, digest = self.client.download(
                        ref.filing_url,
                        destination,
                        max_bytes=max_document_bytes,
                        accept="application/xml, text/xml;q=0.9",
                    )
                    body = destination.read_bytes()
                parsed.append(
                    parse_ownership_document(
                        body,
                        accession_number=ref.accession_number,
                        filing_url=ref.filing_url,
                        document_sha256=digest,
                    )
                )
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                failures.append(
                    {
                        "filing_url": ref.filing_url,
                        "accession_number": ref.accession_number,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if index % 25 == 0:
                print(
                    f"Fetched or verified {index:,}/{len(seen):,} ownership documents",
                    flush=True,
                )
        return parsed, failures

    @staticmethod
    def emit_fact(
        filing: Section16Filing,
        *,
        entity_name: str,
        old_value: str | None = None,
        observed_ts: datetime | None = None,
    ) -> FactEvent:
        """Build the corroborating observation for one ownership filing.

        The attester is the reporting owner, not the issuer, which is what makes
        this independent of the Item 5.02 filing for the same event.

        ``old_value`` is **not attested by this filing**. A Form 3 is an initial
        statement and says nothing about a predecessor; a Form 4 reports the
        owner's own holdings, not the outgoing officer.  It is accepted here
        only so the observation can be grouped with the issuer's account, which
        is where the previous answer actually comes from — and it is recorded as
        unattested in provenance.  So Section 16 corroborates the new value and
        the effective date; ``previous_answer`` stays single-attested.
        """
        attribute = filing.role_attribute
        if attribute is None:
            raise ValueError(
                f"{filing.accession_number}: officerTitle "
                f"{filing.officer_title!r} does not map to a Corvus attribute"
            )
        return FactEvent(
            entity_id=f"CIK{filing.issuer_cik}",
            entity_name=entity_name,
            entity_type="company",
            attribute=attribute,
            old_value=old_value,
            new_value=filing.natural_owner_name,
            effective_ts=filing.effective_ts,
            observed_ts=observed_ts or filing.effective_ts,
            source_url=filing.filing_url,
            source_type=f"sec_form_{filing.form_type.replace('/', '_').lower()}",
            resolver_id=Section16Adapter.resolver_id,
            authority_family="sec",
            attester_id=f"CIK{filing.owner_cik}",
            attester_role="reporting_owner",
            compliance_source_id="sec_edgar",
            distribution_rights_confirmed=True,
            aliases=[filing.owner_name] if filing.owner_name != filing.natural_owner_name else [],
            provenance={
                "accession_number": filing.accession_number,
                "form_type": filing.form_type,
                "officer_title": filing.officer_title,
                "is_officer": filing.is_officer,
                "is_director": filing.is_director,
                "period_of_report": filing.period_of_report.isoformat(),
                "document_sha256": filing.document_sha256,
                "effective_ts_precision": "day",
                # Guards against a later reader mistaking this observation for
                # independent confirmation of the predecessor.
                "old_value_attested": False,
            },
        )
