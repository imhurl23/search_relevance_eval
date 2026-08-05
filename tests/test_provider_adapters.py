"""Pin all three provider request shapes to their published specs.

These adapters are the only place the eval touches a paid third-party API, and a
silently-wrong parameter does not raise — it changes the retrieval condition and
therefore the result. Each assertion below corresponds to a documented rule:

Exa (https://exa.ai/docs/reference/search, /reference/pricing)
Parallel (https://docs.parallel.ai/search/search-quickstart,
          /search/search-migration-guide, /getting-started/pricing)
You.com (https://you.com/docs/api-reference/search)
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("EXA_API_KEY", "test-exa-key")
os.environ.setdefault("PARALLEL_API_KEY", "test-parallel-key")
os.environ.setdefault("YDC_API_KEY", "test-ydc-key")

import run_eval


EXA_RESPONSE = {
    "requestId": "exa-request-id",
    "results": [
        {
            "url": "https://publisher.example/story",
            "title": "Story",
            "highlights": ["h" * 5000],
            "publishedDate": "2026-07-29T00:00:00Z",
        }
    ]
}

PARALLEL_RESPONSE = {
    "search_id": "parallel-search-id",
    "results": [
        {
            "url": "https://publisher.example/story",
            "title": "Story",
            "excerpts": ["p" * 5000],
            "publish_date": "2026-07-29T00:00:00Z",
        }
    ],
}

YDC_RESPONSE = {
    "results": {
        "web": [
            {
                "url": "https://publisher.example/story",
                "title": "Story",
                "description": "desc",
                "snippets": ["s" * 5000],
                "page_age": "2026-07-29T00:00:00Z",
            }
        ],
        "news": [],
    },
    "metadata": {"search_uuid": "uuid", "query": "q", "latency": 0.1},
}


def _capture(response):
    """Patch the shared provider transport and record the outgoing request."""
    calls = []

    def fake_provider_json(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return response

    return calls, patch.object(run_eval, "_provider_json", fake_provider_json)


class ExaAdapterTests(unittest.TestCase):
    def _call(self, arm, exclude_domains=("gold.example",)):
        calls, patcher = _capture(EXA_RESPONSE)
        with patcher:
            results, _ = run_eval.exa_search("q", arm, list(exclude_domains))
        return calls[0], results

    def test_documented_endpoint_and_auth_header(self):
        call, _ = self._call("normalized")
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://api.exa.ai/search")
        self.assertEqual(call["headers"]["x-api-key"], os.environ["EXA_API_KEY"])

    def test_search_tier_is_pinned_not_left_to_per_query_routing(self):
        call, _ = self._call("normalized")
        self.assertEqual(call["json"]["type"], run_eval.EXA_SEARCH_TYPE)
        self.assertIn(
            run_eval.EXA_SEARCH_TYPE,
            {"instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"},
        )

    def test_highlights_use_maxcharacters_not_deprecated_knobs(self):
        call, _ = self._call("normalized")
        highlights = call["json"]["contents"]["highlights"]
        self.assertEqual(highlights["maxCharacters"], run_eval.SNIPPET_CHARS)
        self.assertNotIn("numSentences", highlights)
        self.assertNotIn("highlightsPerUrl", highlights)

    def test_normalized_arm_sends_no_freshness_parameters(self):
        call, _ = self._call("normalized")
        self.assertNotIn("maxAgeHours", call["json"]["contents"])
        self.assertNotIn("livecrawl", call["json"]["contents"])

    def test_fresh_arm_uses_maxagehours_and_never_deprecated_livecrawl(self):
        call, _ = self._call("native_fresh")
        contents = call["json"]["contents"]
        # livecrawl (the string parameter, all values) was deprecated in favor of
        # maxAgeHours in Feb 2026, and the two together contradict each other.
        self.assertNotIn("livecrawl", contents)
        self.assertEqual(contents["maxAgeHours"], run_eval.EXA_MAX_AGE_HOURS)
        self.assertTrue(-1 <= contents["maxAgeHours"] <= 720)
        self.assertLessEqual(
            contents["livecrawlTimeout"] / 1000.0,
            30.0,
            "livecrawlTimeout must stay under the client read timeout",
        )
        self.assertLessEqual(contents["livecrawlTimeout"], 90_000)

    def test_request_stays_inside_documented_limits(self):
        call, _ = self._call("normalized", ["d%d.example" % i for i in range(5)])
        self.assertTrue(1 <= call["json"]["numResults"] <= 100)
        self.assertLessEqual(
            len(call["json"]["excludeDomains"]), run_eval.EXA_MAX_DOMAINS
        )
        self.assertEqual(call["json"]["category"], "news")

    def test_normalized_result_shape_and_snippet_cap(self):
        _, results = self._call("normalized")
        self.assertEqual(
            sorted(results[0]),
            ["published_date", "rank", "snippet", "title", "url"],
        )
        self.assertEqual(len(results[0]["snippet"]), run_eval.SNIPPET_CHARS)
        self.assertEqual(results[0]["published_date"], "2026-07-29T00:00:00Z")


class YouDotComAdapterTests(unittest.TestCase):
    def _call(self, arm, exclude_domains=("gold.example", "web.archive.org")):
        calls, patcher = _capture(YDC_RESPONSE)
        with patcher:
            results, _ = run_eval.youdotcom_search("q", arm, list(exclude_domains))
        return calls[0], results

    def test_documented_endpoint_method_and_auth_header(self):
        call, _ = self._call("normalized")
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://ydc-index.io/v1/search")
        self.assertEqual(call["headers"]["X-API-Key"], os.environ["YDC_API_KEY"])
        self.assertIsNone(call.get("json"), "documented shape is GET query params")

    def test_get_requests_opt_out_of_intermediary_caching(self):
        # You.com documents GET responses as CDN/proxy cacheable; freshness is the
        # measured quantity, so a stale cache hit would corrupt the result.
        call, _ = self._call("native_fresh")
        self.assertEqual(call["headers"]["Cache-Control"], "no-cache")

    def test_exclude_domains_is_comma_separated_and_never_paired_with_include(self):
        call, _ = self._call("normalized")
        self.assertEqual(
            call["params"]["exclude_domains"], "gold.example,web.archive.org"
        )
        # Sending both include_domains and exclude_domains returns 422.
        self.assertNotIn("include_domains", call["params"])

    def test_empty_exclude_list_omits_the_parameter(self):
        call, _ = self._call("normalized", [])
        self.assertNotIn("exclude_domains", call["params"])

    def test_freshness_only_in_fresh_arm_and_a_documented_value(self):
        normalized, _ = self._call("normalized")
        self.assertNotIn("freshness", normalized["params"])
        fresh, _ = self._call("native_fresh")
        self.assertEqual(fresh["params"]["freshness"], "day")

    def test_livecrawl_stays_off_in_both_arms(self):
        for arm in ("normalized", "native_fresh"):
            call, _ = self._call(arm)
            self.assertNotIn("livecrawl", call["params"])
            self.assertNotIn("livecrawl_formats", call["params"])

    def test_normalized_result_shape_matches_exa(self):
        _, results = self._call("normalized")
        self.assertEqual(
            sorted(results[0]),
            ["published_date", "rank", "snippet", "title", "url"],
        )
        self.assertEqual(len(results[0]["snippet"]), run_eval.SNIPPET_CHARS)
        # page_age is the documented timestamp on a web result.
        self.assertEqual(results[0]["published_date"], "2026-07-29T00:00:00Z")

    def test_news_section_is_not_read(self):
        # results.news[] carries no `snippets`, so mixing it in would hand the
        # agent a different decision surface than the other providers get.
        response = {"results": {"web": [], "news": [{"url": "https://n.example",
                                                     "title": "N"}]}}
        calls, patcher = _capture(response)
        with patcher:
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(results, [])


class ParallelAdapterTests(unittest.TestCase):
    def _call(self, arm, exclude_domains=("gold.example", "web.archive.org")):
        calls, patcher = _capture(PARALLEL_RESPONSE)
        with patcher:
            results, _ = run_eval.parallel_search("q", arm, list(exclude_domains))
        return calls[0], results

    def test_ga_endpoint_and_auth_header(self):
        call, _ = self._call("normalized")
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://api.parallel.ai/v1/search")
        self.assertEqual(
            call["headers"]["x-api-key"], os.environ["PARALLEL_API_KEY"])
        self.assertNotIn("parallel-beta", call["headers"])

    def test_mode_is_pinned_and_processor_removed(self):
        call, _ = self._call("normalized")
        self.assertEqual(call["json"]["mode"], run_eval.PARALLEL_MODE)
        self.assertIn(run_eval.PARALLEL_MODE, {"turbo", "basic", "advanced"})
        self.assertNotIn("processor", call["json"])

    def test_ga_settings_are_nested_and_equivalent(self):
        call, _ = self._call("normalized")
        body = call["json"]
        settings = body["advanced_settings"]
        self.assertNotIn("max_results", body)
        self.assertNotIn("source_policy", body)
        self.assertNotIn("excerpts", body)
        self.assertEqual(settings["max_results"], run_eval.N_RESULTS)
        self.assertEqual(
            settings["excerpt_settings"]["max_chars_per_result"],
            run_eval.SNIPPET_CHARS,
        )
        self.assertEqual(
            settings["source_policy"]["exclude_domains"],
            ["gold.example", "web.archive.org"],
        )

    def test_freshness_is_the_only_arm_specific_request_setting(self):
        normalized, _ = self._call("normalized")
        fresh, _ = self._call("native_fresh")
        normalized_settings = normalized["json"]["advanced_settings"]
        fresh_settings = fresh["json"]["advanced_settings"]
        self.assertNotIn("fetch_policy", normalized_settings)
        self.assertEqual(
            fresh_settings["fetch_policy"]["max_age_seconds"],
            run_eval.PARALLEL_MAX_AGE_SECONDS,
        )
        without_freshness = dict(fresh_settings)
        without_freshness.pop("fetch_policy")
        self.assertEqual(without_freshness, normalized_settings)

    def test_normalized_result_shape_and_snippet_cap_match_other_providers(self):
        _, results = self._call("native_fresh")
        self.assertEqual(
            sorted(results[0]),
            ["published_date", "rank", "snippet", "title", "url"],
        )
        self.assertEqual(len(results[0]["snippet"]), run_eval.SNIPPET_CHARS)
        self.assertEqual(results[0]["published_date"], "2026-07-29T00:00:00Z")

    def test_domain_limit_is_enforced(self):
        domains = ["d%d.example" % i for i in range(250)]
        call, _ = self._call("normalized", domains)
        excluded = call["json"]["advanced_settings"]["source_policy"][
            "exclude_domains"]
        self.assertEqual(len(excluded), run_eval.PARALLEL_MAX_DOMAINS)


class SearchCostTests(unittest.TestCase):
    def test_exa_bills_per_call_plus_one_content_type_per_page(self):
        # $7/1k requests + $1/1k pages for highlights.
        self.assertAlmostEqual(
            run_eval.search_cost_usd("exa", "normalized", 8), 0.007 + 8 * 0.001
        )

    def test_exa_charges_for_results_beyond_ten(self):
        # $1/1k results past the first 10, on top of the content charge.
        self.assertAlmostEqual(
            run_eval.search_cost_usd("exa", "normalized", 20),
            0.007 + 20 * 0.001 + 10 * 0.001,
        )

    def test_youdotcom_is_flat_per_call_regardless_of_count(self):
        self.assertAlmostEqual(
            run_eval.search_cost_usd("youdotcom", "normalized", 8), 0.005
        )
        self.assertAlmostEqual(
            run_eval.search_cost_usd("youdotcom", "native_fresh", 20), 0.005
        )

    def test_parallel_uses_current_mode_pricing(self):
        self.assertAlmostEqual(
            run_eval.search_cost_usd("parallel", "normalized", 8), 0.005
        )
        self.assertAlmostEqual(
            run_eval.search_cost_usd("parallel", "native_fresh", 20),
            0.005 + 10 * 0.001,
        )


class InstrumentationTests(unittest.TestCase):
    def test_each_provider_emits_the_same_trace_fields(self):
        responses = {
            "exa": EXA_RESPONSE,
            "parallel": PARALLEL_RESPONSE,
            "youdotcom": YDC_RESPONSE,
        }
        expected_ids = {
            "exa": "exa-request-id",
            "parallel": "parallel-search-id",
            "youdotcom": "uuid",
        }
        expected_log_keys = {"input", "output", "metadata", "metrics"}
        expected_metric_keys = {
            "tokens", "latency_s", "search_cost_usd", "n_results"}
        for provider, response in responses.items():
            span = unittest.mock.Mock()
            calls, transport = _capture(response)
            with transport, patch.object(run_eval, "current_span", return_value=span):
                run_eval.run_search.__wrapped__(
                    provider, "normalized", "q", ["gold.example"])
            self.assertEqual(len(calls), 1)
            logged = span.log.call_args.kwargs
            self.assertEqual(set(logged), expected_log_keys)
            self.assertEqual(set(logged["metrics"]), expected_metric_keys)
            self.assertEqual(logged["input"]["provider"], provider)
            self.assertEqual(logged["metadata"]["provider"], provider)
            self.assertEqual(
                logged["metadata"]["provider_request_id"], expected_ids[provider])
            self.assertFalse(logged["metadata"]["raw_payload_retained"])

    def test_every_provider_arm_pair_is_priced(self):
        for provider in run_eval.PROVIDERS:
            for arm in ("normalized", "native_fresh"):
                self.assertIn((provider, arm), run_eval.SEARCH_PRICING)
                self.assertGreater(
                    run_eval.search_cost_usd(provider, arm, run_eval.N_RESULTS), 0.0
                )

    def test_unknown_pair_costs_zero_rather_than_raising(self):
        self.assertEqual(run_eval.search_cost_usd("unknown", "normalized", 8), 0.0)


if __name__ == "__main__":
    unittest.main()
