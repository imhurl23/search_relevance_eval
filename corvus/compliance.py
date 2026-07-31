"""Fail-closed compliance gates shared by Corvus collection and import tools."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPOSITORY_ROOT / "config" / "corvus"
SOURCE_POLICY_PATH = CONFIG_DIR / "source_compliance.json"
PROVIDER_POLICY_PATH = CONFIG_DIR / "provider_permissions.json"


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_policy(path: Path = SOURCE_POLICY_PATH) -> dict[str, Any]:
    policy = load_json_object(path)
    if not isinstance(policy.get("sources"), dict):
        raise ValueError(f"{path}: missing sources object")
    valid_until = policy.get("review_valid_until")
    if not valid_until:
        raise ValueError(f"{path}: missing review_valid_until")
    if date.today() > date.fromisoformat(valid_until):
        raise ValueError(
            f"{path}: compliance review expired on {valid_until}; re-review official terms"
        )
    return policy


def require_approved_sources(
    source_ids: Iterable[str], *, path: Path = SOURCE_POLICY_PATH
) -> None:
    policy = load_source_policy(path)
    for source_id in sorted(set(source_ids)):
        source = policy["sources"].get(source_id)
        if not source:
            raise ValueError(f"source {source_id!r} has no compliance review")
        if source.get("status") != "approved_with_controls":
            raise ValueError(
                f"source {source_id!r} is not approved: {source.get('status')}"
            )


def require_provider_permission(
    provider: str, *, path: Path = PROVIDER_POLICY_PATH
) -> None:
    policy = load_json_object(path)
    grant = (policy.get("providers") or {}).get(provider)
    if not grant or grant.get("status") != "approved":
        status = grant.get("status") if grant else "missing"
        raise ValueError(
            f"provider {provider!r} is blocked ({status}); record written permission "
            f"in {path} before benchmarking"
        )
    required_true = (
        "benchmarking_allowed",
        "braintrust_storage_allowed",
        "result_storage_allowed",
    )
    provider_specific = {
        "exa": ("output_copy_and_distribution_allowed",),
        "parallel": ("third_party_benchmark_results_allowed",),
        "youdotcom": ("repeated_queries_without_cache_allowed",),
    }
    required_true += provider_specific.get(provider, ())
    missing = [field for field in required_true if grant.get(field) is not True]
    if missing:
        raise ValueError(f"provider {provider!r} permission lacks {missing}")
    for field in ("written_permission_reference", "approved_by", "valid_until"):
        if not grant.get(field):
            raise ValueError(f"provider {provider!r} permission lacks {field}")
    if date.today() > date.fromisoformat(grant["valid_until"]):
        raise ValueError(f"provider {provider!r} written permission has expired")


def require_import_approval(
    approval_path: Path,
    *,
    artifact_path: Path,
    source_policy_path: Path = SOURCE_POLICY_PATH,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    approval = load_json_object(approval_path)
    required = (
        "approval_id",
        "approved_by",
        "approved_at",
        "scope",
        "written_basis_reference",
        "artifact_sha256",
        "source_policy_sha256",
    )
    missing = [field for field in required if not approval.get(field)]
    if missing:
        raise ValueError(f"{approval_path}: missing approval fields {missing}")
    approved_at = datetime.fromisoformat(
        str(approval["approved_at"]).replace("Z", "+00:00")
    )
    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        raise ValueError(f"{approval_path}: approved_at must include a timezone")
    if approved_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError(f"{approval_path}: approved_at is in the future")
    if approval["scope"] != "braintrust_private_dataset_import":
        raise ValueError(f"{approval_path}: approval scope does not permit this import")
    for field in (
        "braintrust_dpa_confirmed",
        "braintrust_retention_policy_confirmed",
        "source_distribution_rights_confirmed",
    ):
        if approval.get(field) is not True:
            raise ValueError(f"{approval_path}: {field} must be explicitly true")
    if approval["artifact_sha256"] != sha256_file(artifact_path):
        raise ValueError(f"{approval_path}: approval does not match dataset artifact")
    if approval["source_policy_sha256"] != sha256_file(source_policy_path):
        raise ValueError(f"{approval_path}: approval does not match source policy")
    if schema_path is not None:
        if not approval.get("schema_sha256"):
            raise ValueError(f"{approval_path}: approval is missing schema_sha256")
        if approval["schema_sha256"] != sha256_file(schema_path):
            raise ValueError(f"{approval_path}: approval does not match dataset schema")
    return approval
