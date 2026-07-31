"""Schemas for separating claim preparation from factual verification."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


SCHEMA_VERSION = "corvus.review.v2"
CLAIM_PREPARATION_TASK = "Prepare one explicit atomic claim for later verification."
FACT_VERIFICATION_TASK = "Verify whether the stated atomic claim is true."

FORBIDDEN_SOURCE_FIELD_PARTS = {
    "article_body",
    "description",
    "evidence_context",
    "excerpt",
    "headline",
    "image",
    "local_path",
    "page_text",
    "raw_response",
    "thumbnail",
    "transcript",
    "video",
}


class FactVerification(str, Enum):
    VERIFIED = "verified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTED = "contradicted"


class ClaimPreparationStatus(str, Enum):
    PREPARED = "prepared"
    NO_ATOMIC_FACT = "no_atomic_fact"
    REJECTED = "rejected"


class AtomicClaim(BaseModel):
    """One explicit proposition that a reviewer can verify or contradict."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    subject_name: str = Field(min_length=1)
    subject_type: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_value: str = Field(min_length=1)
    previous_value: str | None = None
    asserted_effective_ts: datetime | None = None
    time_basis: str = Field(min_length=1)

    @field_validator("asserted_effective_ts")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asserted_effective_ts must include a timezone")
        return value.astimezone(timezone.utc)


class EvidenceReference(BaseModel):
    """A link and attribution, never copied source prose."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    source_role: Literal["candidate", "context", "independent_verification"]
    compliance_source_id: str = Field(min_length=1)
    authority_family: str = Field(min_length=1)
    license: str | None = None
    attribution: str | None = None


class FactVerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["corvus.review.v2"] = SCHEMA_VERSION
    task: Literal["Verify whether the stated atomic claim is true."]
    claim: AtomicClaim
    evidence: list[EvidenceReference] = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    instructions: list[str] = Field(min_length=1)


class FactVerificationExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_verification: FactVerification | None = None
    verified_effective_ts: datetime | None = None
    verification_source_url: HttpUrl | None = None
    correction: str | None = None

    @field_validator("verified_effective_ts")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("verified_effective_ts must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def correction_only_for_contradiction(self) -> "FactVerificationExpected":
        if self.correction and self.fact_verification not in (
            FactVerification.CONTRADICTED,
            None,
        ):
            raise ValueError("correction is only valid for a contradicted claim")
        return self


class ClaimDraft(BaseModel):
    """Fields completed during preparation before verification is allowed."""

    model_config = ConfigDict(extra="forbid")

    statement: str | None = None
    subject_id: str | None = None
    subject_name: str | None = None
    subject_type: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    previous_value: str | None = None
    asserted_effective_ts: datetime | None = None
    time_basis: str | None = None

    @field_validator("asserted_effective_ts")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asserted_effective_ts must include a timezone")
        return value.astimezone(timezone.utc)


class ClaimPreparationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["corvus.review.v2"] = SCHEMA_VERSION
    task: Literal["Prepare one explicit atomic claim for later verification."]
    source_kind: Literal["sec_8k_item_5_02", "wikipedia_current_events_citation"]
    source_record_id: str = Field(min_length=1)
    subject_hint: str | None = None
    source_urls: list[HttpUrl] = Field(min_length=1)
    candidate_context: dict[str, Any] = Field(default_factory=dict)
    instructions: list[str] = Field(min_length=1)


class ClaimPreparationExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preparation_status: ClaimPreparationStatus | None = None
    claim: ClaimDraft = Field(default_factory=ClaimDraft)


def braintrust_schemas(
    input_model: type[BaseModel], expected_model: type[BaseModel]
) -> dict[str, Any]:
    """Return Braintrust's metadata.__schemas payload with enforcement enabled."""

    input_schema = input_model.model_json_schema(mode="validation")
    expected_schema = expected_model.model_json_schema(mode="validation")
    input_schema["enforce"] = True
    expected_schema["enforce"] = True
    return {"input": input_schema, "expected": expected_schema}


def validate_review_row(row: dict[str, Any], *, stage: str) -> None:
    if stage == "claim_preparation":
        ClaimPreparationInput.model_validate(row["input"])
        ClaimPreparationExpected.model_validate(row["expected"])
    elif stage == "fact_verification":
        FactVerificationInput.model_validate(row["input"])
        FactVerificationExpected.model_validate(row["expected"])
    else:
        raise ValueError(f"unknown review stage {stage!r}")
    metadata = row.get("metadata") or {}
    if metadata.get("review_stage") != stage:
        raise ValueError("row metadata review_stage does not match its schema")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("row metadata schema_version does not match")
    if metadata.get("contains_source_text") is not False:
        raise ValueError("review rows must explicitly exclude source text")
    assert_metadata_only(row)


def assert_metadata_only(value: object, path: str = "row") -> None:
    """Reject fields that could accidentally carry copied source material."""

    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.casefold()
            if any(part in lowered for part in FORBIDDEN_SOURCE_FIELD_PARTS):
                raise ValueError(f"{path}.{key}: forbidden source-content field")
            assert_metadata_only(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_metadata_only(child, f"{path}[{index}]")
