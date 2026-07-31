"""Compliance-aware EDGAR and Wikidata source adapters for Corvus-QA."""

from __future__ import annotations

import json
import random
import time
import fcntl
import hashlib
import zipfile
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from corvus.compliance import (
    REPOSITORY_ROOT,
    SOURCE_POLICY_PATH,
    load_source_policy,
    require_approved_sources,
)
from corvus.models import FactEvent
from corvus.news_sports_sources import (
    OpenLigaDbAdapter,
    TheSportsDbAdapter,
    WikipediaCurrentEventsAdapter,
)
from import_livenewsbench import load_env


SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
SEC_BULK_SUBMISSIONS = (
    "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
)
COMPLIANCE_POLICY_PATH = SOURCE_POLICY_PATH
RATE_LIMIT_DIR = REPOSITORY_ROOT / ".corvus" / "rate-limits"


def load_compliance_policy(path: Path = COMPLIANCE_POLICY_PATH) -> dict[str, Any]:
    return load_source_policy(path)


def assert_source_approved(source_id: str, *, path: Path = COMPLIANCE_POLICY_PATH) -> None:
    require_approved_sources([source_id], path=path)


def _host_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    return any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts)


def retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


class SharedHostLimiter:
    """Serialize requests across local processes using an advisory file lock."""

    def __init__(self, host_key: str, interval: float, state_dir: Path = RATE_LIMIT_DIR):
        self.interval = interval
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        safe_key = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in host_key)
        self.path = state_dir / f"{safe_key}.state"

    def wait(self) -> None:
        with self.path.open("a+") as state:
            fcntl.flock(state, fcntl.LOCK_EX)
            state.seek(0)
            raw = state.read().strip()
            next_allowed = float(raw) if raw else 0.0
            delay = next_allowed - time.time()
            if delay > 0:
                time.sleep(delay)
            state.seek(0)
            state.truncate()
            state.write(str(time.time() + self.interval))
            state.flush()
            fcntl.flock(state, fcntl.LOCK_UN)


class PolicyHttpClient:
    """Small synchronous client with source-specific pacing and retry handling."""

    def __init__(
        self,
        *,
        user_agent: str,
        min_interval_seconds: float,
        allowed_hosts: tuple[str, ...],
        rate_limit_key: str,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 5,
    ):
        if "@" not in user_agent and "http" not in user_agent:
            raise ValueError("User-Agent must contain contact information")
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=False,
            transport=transport,
        )
        self._allowed_hosts = tuple(host.lower() for host in allowed_hosts)
        self._limiter = SharedHostLimiter(rate_limit_key, min_interval_seconds)
        self._max_attempts = max_attempts

    def get_json_value(
        self, url: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        request_host = (urlparse(url).hostname or "").lower()
        if not _host_allowed(request_host, self._allowed_hosts):
            raise ValueError(f"request host is not approved: {request_host!r}")
        for attempt in range(self._max_attempts):
            self._limiter.wait()
            response = self._client.get(url, params=params)
            if 300 <= response.status_code < 400:
                raise ValueError(
                    f"redirect refused for approved API request: "
                    f"{response.headers.get('Location', '<missing location>')}"
                )
            final_host = (response.url.host or "").lower()
            if not _host_allowed(final_host, self._allowed_hosts):
                raise ValueError(f"redirected to unapproved host: {final_host!r}")
            retryable_http = response.status_code in (429, 500, 502, 503, 504)
            if (
                request_host.endswith("wikidata.org")
                and response.status_code == 503
                and "Retry-After" not in response.headers
                and "X-Database-Lag" not in response.headers
            ):
                response.raise_for_status()
            value: Any = None
            if not retryable_http:
                response.raise_for_status()
                value = response.json()
                retryable_api = (
                    isinstance(value, dict)
                    and isinstance(value.get("error"), dict)
                    and value["error"].get("code") in {"maxlag", "ratelimited"}
                )
                if not retryable_api:
                    if not isinstance(value, (dict, list)):
                        raise ValueError(f"Expected JSON object or array from {url}")
                    return value
            if attempt == self._max_attempts - 1:
                if retryable_http:
                    response.raise_for_status()
                raise RuntimeError(f"{url}: API remained rate-limited after retries")
            retry_after = retry_after_seconds(response.headers.get("Retry-After"))
            exponential = min(60.0, 2 ** attempt * 5.0)
            time.sleep(max(retry_after, exponential) + random.uniform(0, 0.5))
        raise AssertionError("unreachable")

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict:
        value = self.get_json_value(url, params=params)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object response from {url}")
        return value

    def download(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        accept: str = "application/octet-stream",
    ) -> tuple[int, str]:
        """Stream one approved bulk file atomically, with a hard size ceiling."""
        request_host = (urlparse(url).hostname or "").lower()
        if not _host_allowed(request_host, self._allowed_hosts):
            raise ValueError(f"download host is not approved: {request_host!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(self._max_attempts):
            self._limiter.wait()
            try:
                with self._client.stream(
                    "GET", url, headers={"Accept": accept}
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ValueError("redirect refused for approved bulk download")
                    if response.status_code in (429, 500, 502, 503, 504):
                        if attempt == self._max_attempts - 1:
                            response.raise_for_status()
                        retry_after = retry_after_seconds(
                            response.headers.get("Retry-After")
                        )
                        time.sleep(max(retry_after, min(60.0, 5.0 * 2 ** attempt)))
                        continue
                    response.raise_for_status()
                    declared = int(response.headers.get("Content-Length", "0") or 0)
                    if declared > max_bytes:
                        raise ValueError(
                            f"bulk download declares {declared:,} bytes, above "
                            f"the {max_bytes:,}-byte safety cap"
                        )
                    digest = hashlib.sha256()
                    size = 0
                    with partial.open("wb") as output:
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise ValueError(
                                    f"bulk download exceeded {max_bytes:,}-byte safety cap"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                    partial.replace(destination)
                    return size, digest.hexdigest()
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == self._max_attempts - 1:
                    raise
                time.sleep(min(60.0, 5.0 * 2 ** attempt))
        raise AssertionError("unreachable")


class EdgarFilingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cik: str
    entity_name: str
    accession_number: str
    filing_date: date
    report_date: date | None = None
    acceptance_ts: datetime
    primary_document: str
    filing_url: str
    items: list[str]


class OfficerTransition(BaseModel):
    """Reviewed Item 5.02 extraction; effective date must come from filing text."""

    model_config = ConfigDict(extra="forbid")

    role_attribute: str = Field(pattern=r"^(ceo_of|chairperson_of)$")
    old_value: str | None = None
    new_value: str = Field(min_length=1)
    effective_ts: datetime
    aliases: list[str] = Field(default_factory=list)
    evidence_excerpt_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hash of the reviewed filing excerpt; the excerpt is not distributed.",
    )

    @field_validator("effective_ts")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("effective_ts must include a timezone")
        return value.astimezone(timezone.utc)


class EdgarAdapter:
    """Collect Item 5.02 candidates and emit reviewed officer transitions."""

    def __init__(self, client: PolicyHttpClient):
        self.client = client

    def list_item_502(
        self,
        cik: str,
        *,
        since: date,
        until: date,
    ) -> list[EdgarFilingCandidate]:
        padded_cik = cik.zfill(10)
        payload = self.client.get_json(SEC_SUBMISSIONS.format(cik=padded_cik))
        return self._item_502_from_payload(
            payload, padded_cik=padded_cik, since=since, until=until
        )

    @staticmethod
    def _item_502_from_payload(
        payload: dict[str, Any],
        *,
        padded_cik: str,
        since: date,
        until: date,
    ) -> list[EdgarFilingCandidate]:
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            raise ValueError("SEC submissions response is missing filings.recent")

        fields = (
            "accessionNumber",
            "filingDate",
            "reportDate",
            "acceptanceDateTime",
            "form",
            "primaryDocument",
            "items",
        )
        columns = {field: recent.get(field, []) for field in fields}
        lengths = {len(value) for value in columns.values() if isinstance(value, list)}
        if len(lengths) != 1:
            raise ValueError("SEC submissions columns have inconsistent lengths")

        candidates = []
        for values in zip(*(columns[field] for field in fields), strict=True):
            row = dict(zip(fields, values, strict=True))
            if row["form"] != "8-K":
                continue
            filing_date = date.fromisoformat(row["filingDate"])
            item_codes = [part.strip() for part in str(row["items"] or "").split(",")]
            if not (since <= filing_date <= until and "5.02" in item_codes):
                continue
            accession_compact = row["accessionNumber"].replace("-", "")
            filing_url = SEC_ARCHIVES.format(
                cik=int(padded_cik),
                accession=accession_compact,
                document=row["primaryDocument"],
            )
            report_date = (
                date.fromisoformat(row["reportDate"]) if row["reportDate"] else None
            )
            candidates.append(
                EdgarFilingCandidate(
                    cik=padded_cik,
                    entity_name=payload["name"],
                    accession_number=row["accessionNumber"],
                    filing_date=filing_date,
                    report_date=report_date,
                    acceptance_ts=_parse_sec_datetime(row["acceptanceDateTime"]),
                    primary_document=row["primaryDocument"],
                    filing_url=filing_url,
                    items=item_codes,
                )
            )
        return candidates

    def collect_bulk_item_502(
        self,
        archive_path: Path,
        *,
        since: date,
        until: date,
    ) -> tuple[list[EdgarFilingCandidate], int]:
        """Filter the official nightly all-filer archive without extracting it."""
        candidates: list[EdgarFilingCandidate] = []
        filer_count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith(".json"):
                    continue
                # Reject pathological members before decompression.
                if member.file_size > 100 * 1024 * 1024:
                    raise ValueError(
                        f"{member.filename}: unexpectedly large JSON member"
                    )
                with archive.open(member) as source:
                    payload = json.load(source)
                if not isinstance(payload, dict) or "cik" not in payload:
                    continue
                padded_cik = str(payload["cik"]).zfill(10)
                candidates.extend(
                    self._item_502_from_payload(
                        payload,
                        padded_cik=padded_cik,
                        since=since,
                        until=until,
                    )
                )
                filer_count += 1
        return candidates, filer_count

    def fetch_candidate_filings(
        self,
        candidates: list[EdgarFilingCandidate],
        *,
        output_dir: Path,
        max_document_bytes: int = 30 * 1024 * 1024,
    ) -> list[dict[str, Any]]:
        """Download each unique approved SEC primary document for local review."""
        unique = {candidate.filing_url: candidate for candidate in candidates}
        ledger: list[dict[str, Any]] = []
        for index, candidate in enumerate(
            sorted(unique.values(), key=lambda item: item.filing_url), start=1
        ):
            destination = (
                output_dir
                / candidate.cik
                / f"{candidate.accession_number}_{candidate.primary_document}"
            )
            if destination.is_file():
                body = destination.read_bytes()
                size = len(body)
                digest = hashlib.sha256(body).hexdigest()
                state = "cached"
            else:
                size, digest = self.client.download(
                    candidate.filing_url,
                    destination,
                    max_bytes=max_document_bytes,
                    accept="text/html, text/plain;q=0.9",
                )
                state = "downloaded"
            ledger.append(
                {
                    "cik": candidate.cik,
                    "entity_name": candidate.entity_name,
                    "accession_number": candidate.accession_number,
                    "filing_date": candidate.filing_date.isoformat(),
                    "filing_url": candidate.filing_url,
                    "local_path": str(destination),
                    "bytes": size,
                    "sha256": digest,
                    "state": state,
                }
            )
            if index % 25 == 0:
                print(
                    f"Fetched or verified {index:,}/{len(unique):,} filing documents",
                    flush=True,
                )
        return ledger

    @staticmethod
    def emit_fact(
        candidate: EdgarFilingCandidate,
        transition: OfficerTransition,
    ) -> FactEvent:
        return FactEvent(
            entity_id=f"CIK{candidate.cik}",
            entity_name=candidate.entity_name,
            entity_type="company",
            attribute=transition.role_attribute,
            old_value=transition.old_value,
            new_value=transition.new_value,
            effective_ts=transition.effective_ts,
            observed_ts=candidate.acceptance_ts,
            source_url=candidate.filing_url,
            source_type="sec_8k_item_5_02",
            resolver_id="edgar-8k",
            authority_family="sec",
            compliance_source_id="sec_edgar",
            distribution_rights_confirmed=True,
            aliases=transition.aliases,
            provenance={
                "accession_number": candidate.accession_number,
                "filing_date": candidate.filing_date.isoformat(),
                "report_date": (
                    candidate.report_date.isoformat() if candidate.report_date else None
                ),
                "items": candidate.items,
                "evidence_excerpt_sha256": transition.evidence_excerpt_sha256,
            },
        )


def _parse_sec_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # SEC acceptanceDateTime is Eastern time when an offset is omitted.
        raise ValueError("SEC acceptanceDateTime must include an offset")
    return parsed.astimezone(timezone.utc)


def authority_family_for_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host == "sec.gov" or host.endswith(".sec.gov"):
        return "sec"
    if host == "cisa.gov" or host.endswith(".cisa.gov"):
        return "cisa"
    if host == "nist.gov" or host.endswith(".nist.gov"):
        return "nist"
    return f"publisher:{host.removeprefix('www.')}"


def _claim_value(claim: dict[str, Any]) -> str | None:
    datavalue = claim.get("mainsnak", {}).get("datavalue")
    if not isinstance(datavalue, dict):
        return None
    value = datavalue.get("value")
    if isinstance(value, dict):
        if value.get("id"):
            return str(value["id"])
        if "amount" in value:
            return str(value["amount"]).removeprefix("+")
        if "time" in value:
            return str(value["time"])
    return str(value) if value is not None else None


def _reference_urls(claim: dict[str, Any]) -> list[str]:
    urls = []
    for reference in claim.get("references") or []:
        for snak in reference.get("snaks", {}).get("P854", []):
            value = snak.get("datavalue", {}).get("value")
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                urls.append(value)
    return urls


class WikidataAdapter:
    """Read revision pairs and turn single-valued claim changes into FactEvents."""

    def __init__(self, client: PolicyHttpClient):
        self.client = client

    def fetch_latest_revision_pair(
        self, qid: str
    ) -> tuple[dict[str, Any], dict[str, Any], datetime]:
        payload = self.client.get_json(
            WIKIDATA_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "revisions",
                "titles": qid,
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "rvlimit": 2,
                "maxlag": 1,
            },
        )
        pages = payload.get("query", {}).get("pages", [])
        revisions = pages[0].get("revisions", []) if pages else []
        if len(revisions) != 2:
            raise ValueError(f"{qid}: expected two revisions")
        after, before = revisions
        return (
            _revision_content(before),
            _revision_content(after),
            datetime.fromisoformat(after["timestamp"].replace("Z", "+00:00")),
        )

    def fetch_english_labels(self, ids: set[str]) -> dict[str, str]:
        ids = {item for item in ids if item.startswith(("P", "Q"))}
        if not ids:
            return {}
        payload = self.client.get_json(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(sorted(ids)),
                "props": "labels",
                "languages": "en",
                "languagefallback": 1,
                "maxlag": 1,
            },
        )
        labels = {}
        for entity_id, entity in payload.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                labels[entity_id] = label
        return labels

    def fetch_current_ceos_by_cik(
        self, ciks: set[str], *, batch_size: int = 40
    ) -> list[dict[str, Any]]:
        """Return current Wikidata CEO statements for SEC CIK crosswalks."""
        normalized = sorted({str(cik).zfill(10) for cik in ciks})
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(normalized), batch_size):
            batch = normalized[offset : offset + batch_size]
            values = " ".join(json.dumps(cik) for cik in batch)
            query = f"""
SELECT ?cik ?company ?companyLabel ?ceo ?ceoLabel ?referenceUrl WHERE {{
  VALUES ?cik {{ {values} }}
  ?company wdt:P5531 ?cik;
           p:P169 ?statement.
  ?statement ps:P169 ?ceo.
  FILTER NOT EXISTS {{ ?statement pq:P582 ?endTime }}
  OPTIONAL {{
    ?statement prov:wasDerivedFrom ?reference.
    ?reference pr:P854 ?referenceUrl.
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
            payload = self.client.get_json(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
            )
            for binding in payload.get("results", {}).get("bindings", []):
                rows.append(
                    {
                        "cik": binding.get("cik", {}).get("value"),
                        "company_qid": binding.get("company", {})
                        .get("value", "")
                        .rsplit("/", 1)[-1],
                        "company_label": binding.get("companyLabel", {}).get("value"),
                        "ceo_qid": binding.get("ceo", {})
                        .get("value", "")
                        .rsplit("/", 1)[-1],
                        "ceo_label": binding.get("ceoLabel", {}).get("value"),
                        "reference_url": binding.get("referenceUrl", {}).get("value"),
                        "compliance_source_id": "wikidata",
                        "license": "CC0-1.0",
                    }
                )
        return rows

    def events_from_revision_pair(
        self,
        *,
        qid: str,
        property_id: str,
        before: dict[str, Any],
        after: dict[str, Any],
        observed_ts: datetime,
        effective_ts: datetime,
        labels: dict[str, str],
        attribute: str,
        canonical_entity_id: str | None = None,
        entity_type: str = "wikidata_item",
    ) -> list[FactEvent]:
        before_claims = before.get("claims", {}).get(property_id, [])
        after_claims = after.get("claims", {}).get(property_id, [])
        before_values = {_claim_value(claim) for claim in before_claims}
        after_values = {_claim_value(claim) for claim in after_claims}
        before_values.discard(None)
        after_values.discard(None)
        removed = before_values - after_values
        added = after_values - before_values
        if len(removed) > 1 or len(added) != 1:
            return []

        added_id = next(iter(added))
        matching_claim = next(
            claim for claim in after_claims if _claim_value(claim) == added_id
        )
        urls = _reference_urls(matching_claim)
        events = []
        for url in urls:
            events.append(
                FactEvent(
                    entity_id=canonical_entity_id or qid,
                    entity_name=labels.get(qid, qid),
                    entity_type=entity_type,
                    attribute=attribute,
                    old_value=labels.get(next(iter(removed)), next(iter(removed)))
                    if removed
                    else None,
                    new_value=labels.get(added_id, added_id),
                    effective_ts=effective_ts,
                    observed_ts=observed_ts,
                    source_url=url,
                    source_type="wikidata_claim_revision",
                    resolver_id="wikidata-revision",
                    authority_family=authority_family_for_url(url),
                    compliance_source_id="wikidata",
                    distribution_rights_confirmed=True,
                    provenance={
                        "qid": qid,
                        "property_id": property_id,
                        "license": "CC0-1.0",
                    },
                )
            )
        return events


def _revision_content(revision: dict[str, Any]) -> dict[str, Any]:
    content = revision.get("slots", {}).get("main", {}).get("content")
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise ValueError("Wikidata revision has no main-slot content")


def claim_entity_ids(entity: dict[str, Any], property_id: str) -> set[str]:
    values = {
        _claim_value(claim)
        for claim in entity.get("claims", {}).get(property_id, [])
    }
    return {value for value in values if value and value.startswith(("P", "Q"))}


def _operator_context(
    env_path: Path, *, require_single_deployment: bool
) -> tuple[dict[str, str], str]:
    env = load_env(env_path)
    contact = env.get("CORVUS_CONTACT_EMAIL")
    if (
        not contact
        or contact.count("@") != 1
        or any(character.isspace() for character in contact)
    ):
        raise ValueError("a valid CORVUS_CONTACT_EMAIL is required in the env file")
    if env.get("CORVUS_CONTACT_EMAIL_IS_ROLE_ACCOUNT") != "yes":
        raise ValueError(
            "set CORVUS_CONTACT_EMAIL_IS_ROLE_ACCOUNT=yes in .env only after "
            "confirming the address is a monitored organizational role inbox"
        )
    if (
        require_single_deployment
        and env.get("CORVUS_SINGLE_DEPLOYMENT_CONFIRMED") != "yes"
    ):
        raise ValueError(
            "set CORVUS_SINGLE_DEPLOYMENT_CONFIRMED=yes in .env only after "
            "confirming no other machine or service uses this SEC request budget"
        )
    user_agent = f"CorvusQABot/0.1 ({contact}) httpx/{httpx.__version__}"
    return env, user_agent


def make_official_source_adapters(
    env_path: Path,
    *,
    sec_transport: httpx.BaseTransport | None = None,
    wikidata_transport: httpx.BaseTransport | None = None,
) -> tuple[EdgarAdapter, WikidataAdapter]:
    """Create policy-compliant clients from .env, overriding ambient settings."""

    assert_source_approved("sec_edgar")
    assert_source_approved("wikidata")
    _env, user_agent = _operator_context(
        env_path, require_single_deployment=True
    )
    sec = PolicyHttpClient(
        user_agent=user_agent,
        # Deliberately 10x below SEC's aggregate ceiling. The local file lock
        # shares this budget across processes on this workspace.
        min_interval_seconds=1.0,
        allowed_hosts=("sec.gov",),
        rate_limit_key="sec.gov",
        transport=sec_transport,
    )
    wikidata = PolicyHttpClient(
        user_agent=user_agent,
        # 60/minute, well below the current identified-client limit.
        min_interval_seconds=1.0,
        allowed_hosts=("wikidata.org",),
        rate_limit_key="wikidata.org",
        transport=wikidata_transport,
    )
    return EdgarAdapter(sec), WikidataAdapter(wikidata)


def make_news_sports_source_adapters(
    env_path: Path,
    *,
    enabled_sources: set[str] | None = None,
    wikipedia_transport: httpx.BaseTransport | None = None,
    openligadb_transport: httpx.BaseTransport | None = None,
    thesportsdb_transport: httpx.BaseTransport | None = None,
) -> tuple[
    WikipediaCurrentEventsAdapter,
    OpenLigaDbAdapter,
    TheSportsDbAdapter,
]:
    """Create news/sports clients after each requested source is attested."""

    supported = {
        "wikipedia_current_events",
        "openligadb",
        "thesportsdb",
    }
    requested = enabled_sources or supported
    unknown = requested - supported
    if unknown:
        raise ValueError(f"unsupported news/sports sources: {sorted(unknown)}")
    for source_id in sorted(requested):
        assert_source_approved(source_id)
    env, user_agent = _operator_context(
        env_path, require_single_deployment=False
    )
    attestations = {
        "wikipedia_current_events": "CORVUS_WIKIPEDIA_TERMS_CONFIRMED",
        "openligadb": "CORVUS_OPENLIGADB_LICENSE_CONFIRMED",
        "thesportsdb": "CORVUS_THESPORTSDB_TERMS_CONFIRMED",
    }
    for source_id in sorted(requested):
        name = attestations[source_id]
        if env.get(name) != "yes":
            raise ValueError(
                f"set {name}=yes in .env only after confirming the documented "
                f"terms and attribution requirements for {source_id}"
            )
    wikipedia = PolicyHttpClient(
        user_agent=user_agent,
        min_interval_seconds=1.0,
        allowed_hosts=("wikipedia.org",),
        rate_limit_key="en.wikipedia.org",
        transport=wikipedia_transport,
    )
    openligadb = PolicyHttpClient(
        user_agent=user_agent,
        # OpenLigaDB publishes no numeric limit; remain deliberately serial.
        min_interval_seconds=2.0,
        allowed_hosts=("api.openligadb.de",),
        rate_limit_key="api.openligadb.de",
        transport=openligadb_transport,
    )
    thesportsdb = PolicyHttpClient(
        user_agent=user_agent,
        # The free tier allows 30/minute. 2.1 seconds stays below that ceiling.
        min_interval_seconds=2.1,
        allowed_hosts=("thesportsdb.com",),
        rate_limit_key="www.thesportsdb.com",
        transport=thesportsdb_transport,
    )
    return (
        WikipediaCurrentEventsAdapter(wikipedia),
        OpenLigaDbAdapter(openligadb),
        TheSportsDbAdapter(thesportsdb),
    )
