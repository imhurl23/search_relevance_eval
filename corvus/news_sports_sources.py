"""Metadata-only news and derived-result sports source adapters."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Protocol
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from corvus.models import FactEvent


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
OPENLIGADB_MATCHES = "https://api.openligadb.de/getmatchdata/{league}/{season}"
THESPORTSDB_EVENTS = "https://www.thesportsdb.com/api/v1/json/123/eventsday.php"


class JsonSourceClient(Protocol):
    def get_json(
        self, url: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    def get_json_value(
        self, url: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]: ...


class NewsPageCandidate(BaseModel):
    """Metadata-only review candidate for a Wikipedia Current Events page."""

    model_config = ConfigDict(extra="forbid")

    event_date: date
    page_title: str
    page_url: str
    permanent_url: str
    revision_id: int
    revision_ts: datetime
    section_titles: list[str]
    external_source_urls: list[str]
    compliance_source_id: str = "wikipedia_current_events"
    license: str = "CC-BY-SA-4.0"
    contains_page_text: bool = False

    @field_validator("revision_ts")
    @classmethod
    def revision_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("revision_ts must include a timezone")
        return value.astimezone(timezone.utc)


class SportsResultCandidate(BaseModel):
    """Normalized completed match result awaiting cross-source reconciliation."""

    model_config = ConfigDict(extra="forbid")

    source_event_id: str
    sport: str
    competition: str
    season: str | None = None
    event_start_ts: datetime
    observed_ts: datetime
    home_team: str
    away_team: str
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    source_url: str
    source_type: str
    resolver_id: str
    authority_family: str
    compliance_source_id: str
    license: str | None = None
    attribution: str

    @field_validator("event_start_ts", "observed_ts")
    @classmethod
    def sports_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sports timestamps must include a timezone")
        return value.astimezone(timezone.utc)

    def emit_fact(
        self,
        *,
        canonical_event_id: str,
        canonical_home_team: str,
        canonical_away_team: str,
        effective_ts: datetime,
    ) -> FactEvent:
        """Emit after team reconciliation and effective-time review."""

        score = (
            f"{canonical_home_team} {self.home_score}-"
            f"{self.away_score} {canonical_away_team}"
        )
        return FactEvent(
            entity_id=canonical_event_id,
            entity_name=f"{canonical_home_team} vs {canonical_away_team}",
            entity_type="sports_match",
            attribute="final_score",
            old_value=None,
            new_value=score,
            effective_ts=effective_ts,
            observed_ts=self.observed_ts,
            source_url=self.source_url,
            source_type=self.source_type,
            resolver_id=self.resolver_id,
            authority_family=self.authority_family,
            compliance_source_id=self.compliance_source_id,
            distribution_rights_confirmed=True,
            provenance={
                "source_event_id": self.source_event_id,
                "competition": self.competition,
                "season": self.season,
                "source_home_team": self.home_team,
                "source_away_team": self.away_team,
                "event_start_ts": self.event_start_ts.isoformat(),
                "effective_time_basis": "reviewer_confirmed_match_completion",
                "license": self.license,
                "attribution": self.attribution,
            },
        )


class WikipediaCurrentEventsAdapter:
    """Collect revision and section metadata without copying event prose."""

    def __init__(self, client: JsonSourceClient):
        self.client = client

    def page_candidate(self, event_date: date) -> NewsPageCandidate:
        title = (
            f"Portal:Current events/{event_date.year} "
            f"{event_date.strftime('%B')} {event_date.day}"
        )
        revision_payload = self.client.get_json(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "revisions",
                "titles": title,
                "rvprop": "ids|timestamp",
                "rvlimit": 1,
                "maxlag": 1,
            },
        )
        pages = revision_payload.get("query", {}).get("pages", [])
        if len(pages) != 1 or pages[0].get("missing"):
            raise ValueError(f"{title}: current-events page was not found")
        revisions = pages[0].get("revisions") or []
        if len(revisions) != 1:
            raise ValueError(f"{title}: expected one current revision")
        revision = revisions[0]

        section_payload = self.client.get_json(
            WIKIPEDIA_API,
            params={
                "action": "parse",
                "format": "json",
                "formatversion": 2,
                "page": title,
                "prop": "sections|externallinks",
                "maxlag": 1,
            },
        )
        sections = section_payload.get("parse", {}).get("sections") or []
        section_titles = [
            str(section["line"]).strip()
            for section in sections
            if section.get("line") and str(section.get("level")) in {"2", "3"}
        ]
        external_source_urls = sorted(
            {
                str(url)
                for url in section_payload.get("parse", {}).get(
                    "externallinks", []
                )
                if isinstance(url, str)
                and url.startswith(("http://", "https://"))
            }
        )
        revision_ts = datetime.fromisoformat(
            str(revision["timestamp"]).replace("Z", "+00:00")
        )
        encoded_title = quote(title.replace(" ", "_"), safe=":/")
        page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        return NewsPageCandidate(
            event_date=event_date,
            page_title=title,
            page_url=page_url,
            permanent_url=f"{page_url}?oldid={int(revision['revid'])}",
            revision_id=int(revision["revid"]),
            revision_ts=revision_ts,
            section_titles=section_titles,
            external_source_urls=external_source_urls,
        )


def _parse_utc_timestamp(value: str, *, naive_is_utc: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if not naive_is_utc:
            raise ValueError("source timestamp must include a timezone")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _observed_timestamp(value: datetime | None) -> datetime:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("observed_ts must include a timezone")
    return observed.astimezone(timezone.utc)


class OpenLigaDbAdapter:
    """Collect completed football results from the ODbL OpenLigaDB API."""

    def __init__(self, client: JsonSourceClient):
        self.client = client

    def completed_results(
        self,
        league: str,
        season: str,
        *,
        group_order: int | None = None,
        observed_ts: datetime | None = None,
    ) -> list[SportsResultCandidate]:
        url = OPENLIGADB_MATCHES.format(
            league=quote(league, safe=""),
            season=quote(season, safe=""),
        )
        if group_order is not None:
            if group_order < 1:
                raise ValueError("group_order must be positive")
            url = f"{url}/{group_order}"
        payload = self.client.get_json_value(url)
        if not isinstance(payload, list):
            raise ValueError("OpenLigaDB match response must be an array")
        observed = _observed_timestamp(observed_ts)
        candidates = []
        for row in payload:
            if not isinstance(row, dict) or row.get("matchIsFinished") is not True:
                continue
            results = [
                result
                for result in (row.get("matchResults") or [])
                if isinstance(result, dict)
                and result.get("pointsTeam1") is not None
                and result.get("pointsTeam2") is not None
            ]
            if not results or not row.get("matchDateTimeUTC"):
                continue
            final = max(
                results,
                key=lambda result: (
                    int(result.get("resultOrderID") or 0),
                    int(result.get("resultTypeID") or 0),
                ),
            )
            team1 = row.get("team1") or {}
            team2 = row.get("team2") or {}
            if not team1.get("teamName") or not team2.get("teamName"):
                continue
            match_id = str(row["matchID"])
            candidates.append(
                SportsResultCandidate(
                    source_event_id=match_id,
                    sport="Soccer",
                    competition=str(row.get("leagueName") or league),
                    season=str(row.get("leagueSeason") or season),
                    event_start_ts=_parse_utc_timestamp(
                        str(row["matchDateTimeUTC"]),
                        naive_is_utc=True,
                    ),
                    observed_ts=observed,
                    home_team=str(team1["teamName"]),
                    away_team=str(team2["teamName"]),
                    home_score=int(final["pointsTeam1"]),
                    away_score=int(final["pointsTeam2"]),
                    source_url=f"https://api.openligadb.de/getmatchdata/{match_id}",
                    source_type="openligadb_completed_match",
                    resolver_id="openligadb-results",
                    authority_family="openligadb",
                    compliance_source_id="openligadb",
                    license="ODbL-1.0",
                    attribution=(
                        "Contains information from OpenLigaDB under ODbL 1.0."
                    ),
                )
            )
        return candidates


class TheSportsDbAdapter:
    """Collect a small date slice from the official TheSportsDB v1 API."""

    def __init__(self, client: JsonSourceClient):
        self.client = client

    def completed_results(
        self,
        event_date: date,
        *,
        sport: str | None = None,
        league_id: str | None = None,
        observed_ts: datetime | None = None,
    ) -> list[SportsResultCandidate]:
        params: dict[str, Any] = {"d": event_date.isoformat()}
        if sport:
            params["s"] = sport
        if league_id:
            params["l"] = league_id
        payload = self.client.get_json(THESPORTSDB_EVENTS, params=params)
        observed = _observed_timestamp(observed_ts)
        candidates = []
        for row in payload.get("events") or []:
            if not isinstance(row, dict):
                continue
            if (
                row.get("intHomeScore") in (None, "")
                or row.get("intAwayScore") in (None, "")
                or not row.get("strTimestamp")
                or not row.get("strHomeTeam")
                or not row.get("strAwayTeam")
            ):
                continue
            event_id = str(row["idEvent"])
            candidates.append(
                SportsResultCandidate(
                    source_event_id=event_id,
                    sport=str(row.get("strSport") or sport or "unknown"),
                    competition=str(row.get("strLeague") or league_id or "unknown"),
                    season=str(row["strSeason"]) if row.get("strSeason") else None,
                    event_start_ts=_parse_utc_timestamp(
                        str(row["strTimestamp"]),
                        # TheSportsDB's data guide declares API timestamps UTC.
                        naive_is_utc=True,
                    ),
                    observed_ts=observed,
                    home_team=str(row["strHomeTeam"]),
                    away_team=str(row["strAwayTeam"]),
                    home_score=int(row["intHomeScore"]),
                    away_score=int(row["intAwayScore"]),
                    source_url=f"https://www.thesportsdb.com/event/{event_id}",
                    source_type="thesportsdb_completed_match",
                    resolver_id="thesportsdb-results",
                    authority_family="thesportsdb",
                    compliance_source_id="thesportsdb",
                    attribution="Sports result data sourced from TheSportsDB.",
                )
            )
        return candidates
