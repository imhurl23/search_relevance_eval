from datetime import date, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from corvus.compliance import load_source_policy, require_provider_permission
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

    def test_unapproved_provider_benchmark_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "blocked"):
            require_provider_permission("parallel")

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


if __name__ == "__main__":
    unittest.main()
