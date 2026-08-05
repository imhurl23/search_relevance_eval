"""Pin the You.com request shape to its published spec, and the setups to theirs.

You.com is the only search API in this study, which makes the SETUP the treatment
variable rather than the provider. That raises the stakes on these assertions: a
silently-wrong parameter does not raise, it changes the retrieval condition, and
with one provider there is no second arm whose behavior would look anomalous by
comparison. If `freshness` were dropped on the fresh_day setup, that arm would
quietly become a duplicate of `normalized` and the comparison would report a null
effect that is really a bug.

Spec: https://you.com/docs/api-reference/search (checked 2026-07-30)
  * base host is ydc-index.io, no api. prefix
  * exclude_domains is one comma-separated string, mutually exclusive with
    include_domains (sending both returns 422)
  * freshness accepts day | week | month | year | YYYY-MM-DDtoYYYY-MM-DD
  * count does not affect price
  * results.web[] carries `snippets`; results.news[] does not
  * `page_age` is the timestamp field; there is no publication-date field
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("YDC_API_KEY", "test-ydc-key")

import run_eval


YDC_RESPONSE = {
    "metadata": {"search_uuid": "ydc-search-uuid"},
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
}


class YouComRequestShapeTest(unittest.TestCase):
    def _call(self, arm, excludes=None):
        with patch.object(run_eval, "_provider_json",
                          return_value=YDC_RESPONSE) as sent:
            results, raw = run_eval.youdotcom_search(
                "who won", arm, excludes if excludes is not None else [])
        return sent.call_args, results, raw

    def test_uses_the_documented_host_and_method(self):
        (args, kwargs), _, _ = self._call("normalized")
        self.assertEqual(args[0], "GET")
        self.assertEqual(args[1], "https://ydc-index.io/v1/search")

    def test_sends_the_api_key_and_asks_intermediaries_not_to_cache(self):
        # Freshness is the quantity under measurement, so a CDN hit would be a
        # silent confound.
        (_, kwargs), _, _ = self._call("normalized")
        self.assertEqual(kwargs["headers"]["X-API-Key"], os.environ["YDC_API_KEY"])
        self.assertEqual(kwargs["headers"]["Cache-Control"], "no-cache")

    def test_exclude_domains_is_one_comma_separated_string(self):
        (_, kwargs), _, _ = self._call(
            "normalized", ["source.example", "web.archive.org"])
        self.assertEqual(kwargs["params"]["exclude_domains"],
                         "source.example,web.archive.org")

    def test_empty_exclude_list_is_omitted_not_sent_blank(self):
        # An empty string is not a documented value for this parameter.
        (_, kwargs), _, _ = self._call("normalized", [])
        self.assertNotIn("exclude_domains", kwargs["params"])

    def test_include_domains_is_never_sent(self):
        # Mutually exclusive with exclude_domains; sending both returns 422.
        (_, kwargs), _, _ = self._call("normalized", ["source.example"])
        self.assertNotIn("include_domains", kwargs["params"])

    def test_snippet_is_truncated_to_the_declared_budget(self):
        _, results, _ = self._call("normalized")
        self.assertEqual(len(results[0]["snippet"]), run_eval.SNIPPET_CHARS)

    def test_page_age_maps_to_published_date(self):
        # There is no publication-date field in this response shape, so
        # published_date carries last-modified. temporal_grounding's construct
        # limitation traces to exactly this line.
        _, results, _ = self._call("normalized")
        self.assertEqual(results[0]["published_date"], "2026-07-29T00:00:00Z")

    def test_only_the_web_shape_is_read(self):
        # results.news[] carries no snippets, so it yields no decision surface.
        response = {"metadata": {}, "results": {"web": [], "news": [
            {"url": "https://n.example/x", "title": "N"}]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(results, [])

    def test_request_id_is_captured_without_retaining_the_payload(self):
        _, _, raw = self._call("normalized")
        self.assertEqual(run_eval._provider_request_id(raw), "ydc-search-uuid")


class YouComSetupTest(unittest.TestCase):
    """Each setup must send exactly the parameters that define it — no more."""

    def _params(self, arm):
        with patch.object(run_eval, "_provider_json",
                          return_value=YDC_RESPONSE) as sent:
            run_eval.youdotcom_search("who won", arm, [])
        return sent.call_args[1]["params"]

    def test_normalized_sends_no_freshness_filter(self):
        params = self._params("normalized")
        self.assertNotIn("freshness", params)
        self.assertEqual(params["count"], 8)

    def test_native_fresh_sends_one_day(self):
        self.assertEqual(self._params("native_fresh")["freshness"], "day")

    def test_fresh_week_sends_one_week(self):
        # If this silently matched native_fresh, the window-width comparison
        # would report a null effect that is really a duplicated arm.
        self.assertEqual(self._params("fresh_week")["freshness"], "week")

    def test_wide_raises_count_without_adding_a_freshness_filter(self):
        params = self._params("wide")
        self.assertEqual(params["count"], 20)
        self.assertNotIn("freshness", params)

    def test_setups_differ_from_each_other(self):
        # The treatment axis is only real if the requests actually differ.
        sent = {name: tuple(sorted(self._params(name).items()))
                for name in run_eval.YDC_SETUPS}
        self.assertEqual(len(set(sent.values())), len(run_eval.YDC_SETUPS))

    def test_freshness_values_are_documented_ones(self):
        allowed = {None, "day", "week", "month", "year"}
        for name, cfg in run_eval.YDC_SETUPS.items():
            with self.subTest(setup=name):
                self.assertIn(cfg["freshness"], allowed)

    def test_unknown_setup_fails_loudly_naming_the_valid_ones(self):
        with self.assertRaises(SystemExit) as caught:
            run_eval.ydc_setup("livecrawl")
        self.assertIn("normalized", str(caught.exception))


class YouComPricingTest(unittest.TestCase):
    def test_price_is_per_call_and_independent_of_result_count(self):
        # This is why `wide` is free: 20 results cost the same as 8, so a recall
        # gain there carries no cost penalty.
        self.assertEqual(run_eval.search_cost_usd("normalized", 8),
                         run_eval.search_cost_usd("wide", 20))
        self.assertEqual(run_eval.search_cost_usd("wide", 20),
                         run_eval.YDC_USD_PER_CALL)

    def test_every_harness_setup_costs_the_same(self):
        # Identical search spend across setups isolates the cost comparison to
        # the native arms and the model tokens.
        costs = {run_eval.search_cost_usd(name, cfg["count"])
                 for name, cfg in run_eval.YDC_SETUPS.items()}
        self.assertEqual(len(costs), 1)


class ApprovedHostTest(unittest.TestCase):
    def test_only_youcom_is_an_approved_search_host(self):
        # The removed providers' hosts must not remain reachable: an accidental
        # call would spend money on an API no longer part of the design.
        for url in ("https://api.exa.ai/search",
                    "https://api.parallel.ai/v1/search"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_eval._provider_json("POST", url, json={})


if __name__ == "__main__":
    unittest.main()
