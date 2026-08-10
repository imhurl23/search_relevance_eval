"""Auditable reconciliation of provider-specific sports result candidates.

Collectors deliberately stop at :class:`SportsResultCandidate`.  This module
is the missing curation boundary: a reviewed decision maps provider event/team
identities to one canonical match and supplies the match-completion timestamp.
Only then are normalized ``FactEvent`` observations emitted for the ordinary
Corvus builder.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from corvus.models import FactEvent
from corvus.news_sports_sources import SportsResultCandidate


class SportsCandidateRef(BaseModel):
    """Stable reference to one collected provider result."""

    model_config = ConfigDict(extra="forbid")

    resolver_id: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    reverse_teams: bool = False


class SportsMatchDecision(BaseModel):
    """Human-reviewed mapping from source results to one canonical match."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    approved: bool
    canonical_event_id: str = Field(min_length=1)
    canonical_home_team: str = Field(min_length=1)
    canonical_away_team: str = Field(min_length=1)
    effective_ts: datetime
    effective_time_evidence: str = Field(
        min_length=1,
        description=(
            "Reviewer-supplied basis for the match-completion timestamp; a "
            "scheduled start time alone is not sufficient."
        ),
    )
    observations: list[SportsCandidateRef] = Field(min_length=2)
    review_notes: str | None = None

    @field_validator("effective_ts")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("effective_ts must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def canonical_teams_must_differ(self) -> "SportsMatchDecision":
        if self.canonical_home_team.casefold() == self.canonical_away_team.casefold():
            raise ValueError("canonical home and away teams must differ")
        refs = {(item.resolver_id, item.source_event_id) for item in self.observations}
        if len(refs) != len(self.observations):
            raise ValueError("decision contains duplicate candidate references")
        return self


def load_candidates(paths: Iterable[Path]) -> dict[tuple[str, str], SportsResultCandidate]:
    """Load candidate files and reject ambiguous provider identifiers."""

    candidates: dict[tuple[str, str], SportsResultCandidate] = {}
    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    candidate = SportsResultCandidate.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid sports candidate at {path}:{line_number}: {exc}"
                    ) from exc
                key = (candidate.resolver_id, candidate.source_event_id)
                if key in candidates:
                    raise ValueError(f"duplicate sports candidate key: {key!r}")
                candidates[key] = candidate
    return candidates


def load_decisions(path: Path) -> list[SportsMatchDecision]:
    decisions = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                decisions.append(SportsMatchDecision.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid sports decision at {path}:{line_number}: {exc}"
                ) from exc
    ids = [decision.decision_id for decision in decisions]
    if len(set(ids)) != len(ids):
        raise ValueError("sports decisions contain duplicate decision_id values")
    return decisions


def reconcile_sports(
    candidates: dict[tuple[str, str], SportsResultCandidate],
    decisions: Iterable[SportsMatchDecision],
    *,
    min_resolvers: int = 2,
    min_authorities: int = 2,
    max_start_delta: timedelta = timedelta(hours=6),
) -> tuple[list[FactEvent], list[str]]:
    """Apply reviewed decisions and emit independently sourced observations.

    Team names may differ across providers because the reviewed decision owns
    the canonical identity. Scores, sport, start time, and source independence
    remain mechanically enforced.
    """

    if min_resolvers < 1 or min_authorities < 1:
        raise ValueError("minimum agreement counts must be positive")
    if max_start_delta < timedelta(0):
        raise ValueError("max_start_delta must not be negative")

    facts: list[FactEvent] = []
    skipped: list[str] = []
    consumed: set[tuple[str, str]] = set()
    for decision in decisions:
        if not decision.approved:
            skipped.append(decision.decision_id)
            continue

        selected: list[tuple[SportsCandidateRef, SportsResultCandidate]] = []
        for ref in decision.observations:
            key = (ref.resolver_id, ref.source_event_id)
            if key not in candidates:
                raise ValueError(
                    f"{decision.decision_id}: missing candidate {key!r}"
                )
            if key in consumed:
                raise ValueError(
                    f"{decision.decision_id}: candidate {key!r} is reused"
                )
            selected.append((ref, candidates[key]))

        resolvers = {candidate.resolver_id for _, candidate in selected}
        authorities = {candidate.authority_family for _, candidate in selected}
        if len(resolvers) < min_resolvers:
            raise ValueError(
                f"{decision.decision_id}: insufficient independent resolvers"
            )
        if len(authorities) < min_authorities:
            raise ValueError(
                f"{decision.decision_id}: insufficient independent authorities"
            )
        sports = {candidate.sport.casefold() for _, candidate in selected}
        if len(sports) != 1:
            raise ValueError(f"{decision.decision_id}: sport disagreement")

        starts = [candidate.event_start_ts for _, candidate in selected]
        if max(starts) - min(starts) > max_start_delta:
            raise ValueError(
                f"{decision.decision_id}: event start times exceed allowed delta"
            )
        if decision.effective_ts < max(starts):
            raise ValueError(
                f"{decision.decision_id}: effective_ts precedes an event start"
            )
        if decision.effective_ts > min(
            candidate.observed_ts for _, candidate in selected
        ):
            raise ValueError(
                f"{decision.decision_id}: effective_ts is after a source observation"
            )

        canonical_scores = set()
        for ref, candidate in selected:
            score = (candidate.home_score, candidate.away_score)
            canonical_scores.add(tuple(reversed(score)) if ref.reverse_teams else score)
        if len(canonical_scores) != 1:
            raise ValueError(f"{decision.decision_id}: final score disagreement")

        for ref, candidate in selected:
            home = decision.canonical_away_team if ref.reverse_teams else decision.canonical_home_team
            away = decision.canonical_home_team if ref.reverse_teams else decision.canonical_away_team
            fact = candidate.emit_fact(
                canonical_event_id=decision.canonical_event_id,
                canonical_home_team=home,
                canonical_away_team=away,
                effective_ts=decision.effective_ts,
            )
            if ref.reverse_teams:
                # emit_fact uses the candidate's home/away scores. Restore the
                # canonical orientation when a provider listed the teams backwards.
                home_score, away_score = next(iter(canonical_scores))
                fact.new_value = (
                    f"{decision.canonical_home_team} {home_score}-"
                    f"{away_score} {decision.canonical_away_team}"
                )
                fact.entity_name = (
                    f"{decision.canonical_home_team} vs "
                    f"{decision.canonical_away_team}"
                )
            fact.attester_id = candidate.authority_family
            fact.attester_role = "result_provider"
            fact.provenance.update(
                {
                    "curation_decision_id": decision.decision_id,
                    "effective_time_evidence": decision.effective_time_evidence,
                    "review_notes": decision.review_notes,
                    "reverse_teams": ref.reverse_teams,
                }
            )
            facts.append(fact)
            consumed.add((candidate.resolver_id, candidate.source_event_id))

    return facts, skipped


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for record in records:
            output.write(
                json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
            )
