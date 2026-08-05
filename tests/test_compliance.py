from datetime import date, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from corvus.compliance import (
    load_source_policy,
    require_import_approval,
    sha256_file,
)
from corvus.sources import PolicyHttpClient


class ComplianceGateTests(unittest.TestCase):
    def test_expired_source_review_fails_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(
                json.dumps(
                    {
                        "review_valid_until": (
                            date.today() - timedelta(days=1)
                        ).isoformat(),
                        "sources": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "expired"):
                load_source_policy(path)

    def test_retry_after_and_transient_status_are_retried(self):
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )

        def handler(request):
            response = next(responses)
            response.request = request
            return response

        client = PolicyHttpClient(
            user_agent="Corvus-QA/0.1 (research@example.com)",
            min_interval_seconds=0,
            allowed_hosts=("sec.gov",),
            rate_limit_key="test-retry",
            transport=httpx.MockTransport(handler),
        )
        with patch("corvus.sources.time.sleep") as sleep:
            self.assertEqual(
                client.get_json("https://data.sec.gov/example.json"),
                {"ok": True},
            )
        sleep.assert_called_once()

    def test_schema_bound_approval_fails_when_schema_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.jsonl"
            schema = root / "schema.json"
            policy = root / "policy.json"
            approval = root / "approval.json"
            artifact.write_text("{}\n")
            schema.write_text('{"input": {}}\n')
            policy.write_text(
                json.dumps(
                    {
                        "review_valid_until": (
                            date.today() + timedelta(days=1)
                        ).isoformat(),
                        "sources": {},
                    }
                )
            )
            approval.write_text(
                json.dumps(
                    {
                        "approval_id": "approval-1",
                        "approved_by": "owner",
                        "approved_at": "2026-01-01T00:00:00Z",
                        "scope": "braintrust_private_dataset_import",
                        "written_basis_reference": "test",
                        "artifact_sha256": sha256_file(artifact),
                        "source_policy_sha256": sha256_file(policy),
                        "schema_sha256": sha256_file(schema),
                        "braintrust_dpa_confirmed": True,
                        "braintrust_retention_policy_confirmed": True,
                        "source_distribution_rights_confirmed": True,
                    }
                )
            )
            schema.write_text('{"input": {"enforce": true}}\n')
            with self.assertRaisesRegex(ValueError, "does not match dataset schema"):
                require_import_approval(
                    approval,
                    artifact_path=artifact,
                    source_policy_path=policy,
                    schema_path=schema,
                )


if __name__ == "__main__":
    unittest.main()
