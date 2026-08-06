"""Core schemas and deterministic row builder for the Corvus-QA dataset.

Source adapters or explicit curation emit one ``FactEvent`` per observation.
The builder consumes only those normalized observations; it never consumes
source-specific candidates. It turns an event group into a benchmark row only
after independent attesters agree on the new value. Keeping collection,
curation, and question generation separate makes the eligibility rules testable
without source APIs.

Independence is measured on three separate axes, because conflating them either
admits correlated errors or rejects genuinely independent evidence:

``resolver_id``
    The detector that produced the observation.  Two resolvers reading the same
    document are not independent evidence.
``attester_id``
    The legal person who asserted the fact and is accountable for it.  This is
    the gate.  An issuer's Form 8-K and the incoming officer's Section 16
    Form 3 are separately prepared and separately liable, so a transcription
    error in one does not propagate to the other.
``authority_family``
    The publisher that disseminated the observation.  Recorded for provenance
    and available as an optional gate, but publisher diversity is not what
    protects against a wrong name or a wrong date.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class DatasetSplit(str, Enum):
    DEV = "dev"
    TEST = "test"


class AnswerClass(str, Enum):
    CHANGED = "changed"
    POST_CUTOFF_NOVEL = "post_cutoff_novel"
    UNANSWERABLE_TRAP = "unanswerable_trap"


class FactEvent(BaseModel):
    """One resolver's observation of a fact transition."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1)
    entity_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    old_value: str | None = None
    new_value: str = Field(min_length=1)
    effective_ts: datetime
    observed_ts: datetime
    source_url: HttpUrl
    source_type: str = Field(min_length=1)
    resolver_id: str = Field(min_length=1)
    authority_family: str = Field(
        min_length=1,
        description=(
            "Publisher that disseminated the observation, not the detector. A "
            "Wikidata claim citing an SEC filing should use the SEC authority "
            "family."
        ),
    )
    attester_id: str | None = Field(
        default=None,
        description=(
            "Stable identifier for the legal person accountable for the "
            "assertion — an issuer CIK for a company filing, a reporting-owner "
            "CIK for a Section 16 filing. Defaults to authority_family, which "
            "keeps adapters that do not distinguish attesters exactly as strict "
            "as publisher-level agreement."
        ),
    )
    attester_role: str | None = Field(
        default=None,
        description="How the attester relates to the entity, e.g. issuer or reporting_owner.",
    )
    compliance_source_id: str = Field(min_length=1)
    distribution_rights_confirmed: bool
    aliases: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "entity_id",
        "entity_name",
        "entity_type",
        "attribute",
        "new_value",
        "source_type",
        "resolver_id",
        "authority_family",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("old_value", "attester_id", "attester_role")
    @classmethod
    def normalize_old_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("effective_ts", "observed_ts")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_transition(self) -> "FactEvent":
        if canonical_value(self.old_value) == canonical_value(self.new_value):
            raise ValueError("old_value and new_value must differ")
        if not self.distribution_rights_confirmed:
            raise ValueError("distribution rights must be confirmed before ingestion")
        return self

    @property
    def answer_class(self) -> AnswerClass:
        return (
            AnswerClass.CHANGED
            if self.old_value is not None
            else AnswerClass.POST_CUTOFF_NOVEL
        )

    @property
    def accountable_attester(self) -> str:
        """The attesting party, falling back to the publisher when unset.

        The fallback is the conservative direction: an adapter that does not
        name its attester can never manufacture independence, because every one
        of its observations collapses onto the same publisher identity.
        """
        return self.attester_id or self.authority_family


class CorvusRow(BaseModel):
    """Braintrust-compatible, locally serializable Corvus-QA row."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: dict[str, str]
    expected: str
    metadata: dict[str, Any]
    tags: list[str]


class CoverageAssessment(BaseModel):
    """Pinned top-k reference-search assessment; no snippets are retained."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    attribute: str
    new_value: str
    reference_engine: str
    compliance_source_id: str = Field(min_length=1)
    query: str
    queried_ts: datetime
    top_k: int = Field(default=20, ge=1)
    answer_bearing_domains: list[str] = Field(default_factory=list)
    storage_rights_confirmed: bool = False
    evidence_artifact_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("queried_ts")
    @classmethod
    def coverage_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("queried_ts must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def require_storage_rights(self) -> "CoverageAssessment":
        if not self.storage_rights_confirmed:
            raise ValueError(
                "coverage evidence cannot be used without confirmed result-storage rights"
            )
        return self


class TrapObservation(BaseModel):
    """One authority's observation of an event that resolves after the run."""

    model_config = ConfigDict(extra="forbid")

    trap_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    entity_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    scheduled_resolution_ts: datetime
    observed_ts: datetime
    source_url: HttpUrl
    source_type: str = Field(min_length=1)
    resolver_id: str = Field(min_length=1)
    authority_family: str = Field(min_length=1)
    attester_id: str | None = None
    attester_role: str | None = None
    compliance_source_id: str = Field(min_length=1)
    distribution_rights_confirmed: bool

    @property
    def accountable_attester(self) -> str:
        return self.attester_id or self.authority_family

    @field_validator("scheduled_resolution_ts", "observed_ts")
    @classmethod
    def trap_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def trap_distribution_rights_required(self) -> "TrapObservation":
        if not self.distribution_rights_confirmed:
            raise ValueError("distribution rights must be confirmed before ingestion")
        return self


QUESTION_TEMPLATES: dict[str, str] = {
    "ceo_of": "Who is the CEO of {entity_name}?",
    "chairperson_of": "Who is the chairperson of {entity_name}?",
    "cfo_of": "Who is the CFO of {entity_name}?",
    "coo_of": "Who is the COO of {entity_name}?",
    "head_coach_of": "Who is the head coach of {entity_name}?",
    "latest_stable_version": "What is the latest stable release of {entity_name}?",
    "cvss_score": "What is the current CVSS score for {entity_name}?",
    "kev_status": "Is {entity_name} currently in CISA's Known Exploited Vulnerabilities catalog?",
    "final_score": "What was the final score of {entity_name}?",
}


def canonical_value(value: str | None) -> str:
    if value is None:
        return ""
    value = value.casefold().strip()
    value = re.sub(r"^v(?=\d)", "", value)
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    return re.sub(r"\s+", " ", value)


def _agreement_key(event: FactEvent) -> tuple[str, str, str]:
    return (
        event.entity_id.casefold(),
        event.attribute.casefold(),
        canonical_value(event.new_value),
    )


def _row_id(event: FactEvent, freeze_id: str) -> str:
    identity = json.dumps(
        [
            freeze_id,
            event.entity_id,
            event.attribute,
            canonical_value(event.new_value),
            event.effective_ts.isoformat(),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def recency_rung(effective_ts: datetime, as_of_ts: datetime) -> str:
    age = as_of_ts - effective_ts
    if age.total_seconds() < 0:
        return "future"
    hours = age.total_seconds() / 3600
    if hours < 24:
        return "lt_24h"
    if hours < 72:
        return "24h_72h"
    if hours < 24 * 7:
        return "3d_7d"
    if hours < 24 * 30:
        return "7d_30d"
    return "gte_30d"


def coverage_tier(assessment: CoverageAssessment | None) -> str:
    if assessment is None:
        return "unmeasured"
    count = len({domain.casefold() for domain in assessment.answer_bearing_domains})
    if count == 0:
        return "uncovered"
    if count <= 2:
        return "tail"
    if count <= 5:
        return "torso"
    return "head"


def build_rows(
    events: Iterable[FactEvent],
    *,
    split: DatasetSplit,
    freeze_id: str,
    min_resolvers: int = 2,
    min_attesters: int = 2,
    min_authorities: int = 1,
    templates: dict[str, str] | None = None,
    as_of_ts: datetime | None = None,
    coverage: dict[tuple[str, str, str], CoverageAssessment] | None = None,
) -> tuple[list[CorvusRow], list[dict[str, Any]]]:
    """Resolve observations and return (eligible rows, rejection records).

    ``min_attesters`` is the independence gate: two separately accountable
    parties must assert the same transition.  ``min_authorities`` defaults to 1
    because publisher diversity is provenance, not corroboration — raise it when
    a specific study needs distribution-channel independence as well.
    """

    if not freeze_id.strip():
        raise ValueError("freeze_id must not be blank")
    if min_resolvers < 1 or min_attesters < 1 or min_authorities < 1:
        raise ValueError("minimum agreement counts must be positive")
    if as_of_ts is not None:
        if as_of_ts.tzinfo is None or as_of_ts.utcoffset() is None:
            raise ValueError("as_of_ts must include a timezone")
        as_of_ts = as_of_ts.astimezone(timezone.utc)

    template_map = templates or QUESTION_TEMPLATES
    coverage = coverage or {}
    grouped: dict[tuple[str, str, str], list[FactEvent]] = defaultdict(list)
    for event in events:
        grouped[_agreement_key(event)].append(event)

    rows: list[CorvusRow] = []
    rejected: list[dict[str, Any]] = []
    for key, observations in sorted(grouped.items()):
        exemplar = min(observations, key=lambda item: item.observed_ts)
        resolvers = sorted({item.resolver_id for item in observations})
        attesters = sorted({item.accountable_attester for item in observations})
        authorities = sorted({item.authority_family for item in observations})
        reasons = []
        if len(resolvers) < min_resolvers:
            reasons.append("insufficient_resolvers")
        if len(attesters) < min_attesters:
            reasons.append("insufficient_attesters")
        if len(authorities) < min_authorities:
            reasons.append("insufficient_authorities")
        if exemplar.attribute not in template_map:
            reasons.append("unsupported_attribute")
        if as_of_ts is not None and exemplar.effective_ts > as_of_ts:
            reasons.append("not_yet_effective")

        old_values = {
            canonical_value(item.old_value)
            for item in observations
            if item.old_value is not None
        }
        old_value_presence = {item.old_value is not None for item in observations}
        if len(old_values) > 1 or len(old_value_presence) > 1:
            reasons.append("old_value_disagreement")
        effective_times = {item.effective_ts for item in observations}
        if len(effective_times) > 1:
            reasons.append("effective_ts_disagreement")

        if reasons:
            rejected.append(
                {
                    "agreement_key": list(key),
                    "reasons": reasons,
                    "resolver_ids": resolvers,
                    "attester_ids": attesters,
                    "authority_families": authorities,
                    "observation_count": len(observations),
                }
            )
            continue

        question = template_map[exemplar.attribute].format(
            entity_name=exemplar.entity_name
        )
        source_urls = sorted({str(item.source_url) for item in observations})
        observed_ts = max(item.observed_ts for item in observations)
        aliases = sorted(
            {
                alias.strip()
                for item in observations
                for alias in item.aliases
                if alias.strip()
            }
            | {
                item.new_value
                for item in observations
                if item.new_value != exemplar.new_value
            }
        )
        assessment = coverage.get(_agreement_key(exemplar))
        tier = coverage_tier(assessment)
        rung = (
            recency_rung(exemplar.effective_ts, as_of_ts)
            if as_of_ts is not None
            else "unmeasured"
        )
        metadata = {
            "dataset": "Corvus-QA",
            "corvus_freeze_id": freeze_id,
            "corvus_split": split.value,
            "entity_id": exemplar.entity_id,
            "entity_name": exemplar.entity_name,
            "entity_type": exemplar.entity_type,
            "attribute": exemplar.attribute,
            "answer_class": exemplar.answer_class.value,
            "previous_answer": exemplar.old_value,
            "answer_aliases": aliases,
            "effective_ts": exemplar.effective_ts.isoformat(),
            "event_date": exemplar.effective_ts.date().isoformat(),
            "observed_ts": observed_ts.isoformat(),
            "as_of_ts": as_of_ts.isoformat() if as_of_ts else None,
            "recency_rung": rung,
            "coverage_tier": tier,
            "coverage": (
                assessment.model_dump(mode="json") if assessment is not None else None
            ),
            "resolver_ids": resolvers,
            "attester_ids": attesters,
            "attester_roles": sorted(
                {item.attester_role for item in observations if item.attester_role}
            ),
            "authority_families": authorities,
            "compliance_source_ids": sorted(
                {item.compliance_source_id for item in observations}
                | (
                    {assessment.compliance_source_id}
                    if assessment is not None
                    else set()
                )
            ),
            "source_types": sorted({item.source_type for item in observations}),
            "source_urls": source_urls,
            # Existing leakage code consumes articles[*].url.
            "articles": [{"url": url} for url in source_urls],
        }
        rows.append(
            CorvusRow(
                id=_row_id(exemplar, freeze_id),
                input={"question": question},
                expected=exemplar.new_value,
                metadata=metadata,
                tags=[
                    split.value,
                    exemplar.answer_class.value,
                    exemplar.attribute,
                    rung,
                    tier,
                ],
            )
        )

    return rows, rejected


def build_trap_rows(
    observations: Iterable[TrapObservation],
    *,
    split: DatasetSplit,
    freeze_id: str,
    run_end: datetime,
    safety_buffer: timedelta = timedelta(days=7),
    min_resolvers: int = 2,
    min_attesters: int = 2,
    min_authorities: int = 1,
) -> tuple[list[CorvusRow], list[dict[str, Any]]]:
    """Build traps whose scheduled resolution is safely after the run window."""

    if run_end.tzinfo is None or run_end.utcoffset() is None:
        raise ValueError("run_end must include a timezone")
    run_end = run_end.astimezone(timezone.utc)
    grouped: dict[str, list[TrapObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.trap_id].append(observation)

    rows, rejected = [], []
    for trap_id, group in sorted(grouped.items()):
        exemplar = min(group, key=lambda item: item.observed_ts)
        resolvers = sorted({item.resolver_id for item in group})
        attesters = sorted({item.accountable_attester for item in group})
        authorities = sorted({item.authority_family for item in group})
        questions = {item.question for item in group}
        resolution_times = {item.scheduled_resolution_ts for item in group}
        reasons = []
        if len(resolvers) < min_resolvers:
            reasons.append("insufficient_resolvers")
        if len(attesters) < min_attesters:
            reasons.append("insufficient_attesters")
        if len(authorities) < min_authorities:
            reasons.append("insufficient_authorities")
        if len(questions) != 1:
            reasons.append("question_disagreement")
        if len(resolution_times) != 1:
            reasons.append("resolution_time_disagreement")
        if exemplar.scheduled_resolution_ts <= run_end + safety_buffer:
            reasons.append("resolves_inside_safety_window")
        if reasons:
            rejected.append({"trap_id": trap_id, "reasons": reasons})
            continue

        source_urls = sorted({str(item.source_url) for item in group})
        identity = json.dumps(
            [freeze_id, trap_id, exemplar.scheduled_resolution_ts.isoformat()],
            separators=(",", ":"),
        )
        row_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        metadata = {
            "dataset": "Corvus-QA",
            "corvus_freeze_id": freeze_id,
            "corvus_split": split.value,
            "answer_class": AnswerClass.UNANSWERABLE_TRAP.value,
            "trap_id": trap_id,
            "entity_id": exemplar.entity_id,
            "entity_name": exemplar.entity_name,
            "entity_type": exemplar.entity_type,
            "attribute": exemplar.attribute,
            "scheduled_resolution_ts": exemplar.scheduled_resolution_ts.isoformat(),
            "run_end": run_end.isoformat(),
            "safety_buffer_seconds": int(safety_buffer.total_seconds()),
            "resolver_ids": resolvers,
            "attester_ids": attesters,
            "authority_families": authorities,
            "compliance_source_ids": sorted(
                {item.compliance_source_id for item in group}
            ),
            "source_types": sorted({item.source_type for item in group}),
            "source_urls": source_urls,
            "articles": [{"url": url} for url in source_urls],
            "recency_rung": "future",
            "coverage_tier": "not_applicable",
        }
        rows.append(
            CorvusRow(
                id=row_id,
                input={"question": exemplar.question},
                expected="I could not find this.",
                metadata=metadata,
                tags=[
                    split.value,
                    AnswerClass.UNANSWERABLE_TRAP.value,
                    exemplar.attribute,
                ],
            )
        )
    return rows, rejected
