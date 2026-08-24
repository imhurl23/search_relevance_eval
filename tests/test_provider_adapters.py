"""Pin the You.com request shape to its published spec, and the setups to theirs.

You.com is the only search API in this study, which makes the SETUP the treatment
variable rather than the provider. That raises the stakes on these assertions: a
silently-wrong parameter does not raise, it changes the retrieval condition, and
with one provider there is no second arm whose behavior would look anomalous by
comparison. If `freshness` were dropped on the fresh_day setup, that arm would
quietly become a duplicate of `normalized` and the comparison would report a null
effect that is really a bug.

Spec: https://you.com/docs/api-reference/search (checked 2026-08-17)
  * base host is ydc-index.io, no api. prefix
  * POST is the documented path; extraction_mode: "highlights" is POST-only
  * exclude_domains is a JSON array on POST, mutually exclusive with
    include_domains (sending both returns 422)
  * freshness accepts day | week | month | year | YYYY-MM-DDtoYYYY-MM-DD
  * count does not affect price
  * results.web[] carries contents.highlights when extraction is requested, and
    NO `snippets` key at all in that case (verified live 2026-08-18)
  * results.news[] carries NO `contents` key, so news snippets come from
    `description` — roughly 150 chars against ~2900 for a web highlight set
  * `page_age` is the timestamp field; for news results it is a publication date
  * `count` is applied PER SECTION, so the registered baseline requests up to
    5 web + 5 news results

Merge policy is ours, not You.com's: the two sections are interleaved by
within-section rank, and news is additive rather than capped into `count`.
Concatenating instead would pin every news result below every web result and
bury the freshest coverage on a benchmark that is mostly news.
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
                "snippets": ["first snippet", "second snippet"],
                "page_age": "2026-07-29T00:00:00Z",
                "contents": {
                    "highlights": ["highlight one", "highlight two"],
                },
            }
        ],
        "news": [
            {
                "url": "https://news.example/breaking",
                "title": "Breaking News",
                "description": "News article summary",
                "page_age": "2026-08-17T10:00:00Z",
                "contents": {
                    "highlights": ["news highlight passage"],
                },
            }
        ],
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
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "https://ydc-index.io/v1/search")

    def test_sends_the_api_key_and_asks_intermediaries_not_to_cache(self):
        # Freshness is the quantity under measurement, so a CDN hit would be a
        # silent confound.
        (_, kwargs), _, _ = self._call("normalized")
        self.assertEqual(kwargs["headers"]["X-API-Key"], os.environ["YDC_API_KEY"])
        self.assertEqual(kwargs["headers"]["Cache-Control"], "no-cache")

    def test_sends_extraction_mode_highlights(self):
        # Highlights are query-aware passages purpose-built for agent grounding
        # and are only available on POST.
        (_, kwargs), _, _ = self._call("normalized")
        self.assertEqual(kwargs["json"]["extraction"],
                         {"extraction_mode": "highlights"})

    def test_pins_location_language_and_safesearch(self):
        (_, kwargs), _, _ = self._call("normalized")
        self.assertEqual(kwargs["json"]["country"], "US")
        self.assertEqual(kwargs["json"]["language"], "en")
        self.assertEqual(kwargs["json"]["safesearch"], "moderate")

    def test_exclude_domains_is_a_json_array(self):
        (_, kwargs), _, _ = self._call(
            "normalized", ["source.example", "web.archive.org"])
        self.assertEqual(kwargs["json"]["exclude_domains"],
                         ["source.example", "web.archive.org"])

    def test_empty_exclude_list_is_omitted_not_sent_blank(self):
        # An empty list is not a documented value for this parameter.
        (_, kwargs), _, _ = self._call("normalized", [])
        self.assertNotIn("exclude_domains", kwargs["json"])

    def test_include_domains_is_never_sent(self):
        # Mutually exclusive with exclude_domains; sending both returns 422.
        (_, kwargs), _, _ = self._call("normalized", ["source.example"])
        self.assertNotIn("include_domains", kwargs["json"])

    def test_snippet_is_not_truncated(self):
        # The full highlight passage is passed to the agent without truncation.
        _, results, _ = self._call("normalized")
        self.assertEqual(results[0]["snippet"], "highlight one\nhighlight two")

    def test_all_highlights_are_used_not_just_the_first(self):
        _, results, _ = self._call("normalized")
        self.assertIn("highlight two", results[0]["snippet"])

    def test_page_age_is_retained_with_explicit_semantics(self):
        _, results, _ = self._call("normalized")
        self.assertEqual(results[0]["published_date"], "2026-07-29T00:00:00Z")
        self.assertEqual(results[0]["date_semantics"],
                         "provider_page_age_unverified")

    def test_news_results_are_read_alongside_web(self):
        # results.news[] is automatically returned for news-intent queries.
        _, results, _ = self._call("normalized")
        self.assertEqual(len(results), 2)  # 1 web + 1 news
        self.assertEqual(results[1]["url"], "https://news.example/breaking")
        self.assertEqual(results[1]["published_date"], "2026-08-17T10:00:00Z")
        self.assertEqual(results[1]["date_semantics"], "publication")
        self.assertEqual(results[1]["snippet"], "news highlight passage")

    def test_each_result_records_which_section_it_came_from(self):
        # Sections have independent ranking and date semantics, so the section
        # must survive into the result dict.
        _, results, _ = self._call("normalized")
        self.assertEqual([r["source"] for r in results], ["web", "news"])
        self.assertEqual([r["section_rank"] for r in results], [1, 1])

    def test_sections_are_interleaved_not_concatenated(self):
        # Concatenation would pin every news result below every web result. On
        # a freshness study that buries the freshest coverage at the bottom of
        # the surface, confounding section with rank.
        response = {"metadata": {}, "results": {
            "web": [{"url": f"https://w.example/{i}", "title": f"W{i}",
                     "description": "d"} for i in range(3)],
            "news": [{"url": f"https://n.example/{i}", "title": f"N{i}",
                      "description": "d"} for i in range(3)]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual([r["source"] for r in results],
                         ["web", "news", "web", "news", "web", "news"])
        self.assertEqual([r["rank"] for r in results], [1, 2, 3, 4, 5, 6])

    def test_uneven_sections_append_the_longer_tail(self):
        response = {"metadata": {}, "results": {
            "web": [{"url": f"https://w.example/{i}", "title": f"W{i}",
                     "description": "d"} for i in range(3)],
            "news": [{"url": "https://n.example/0", "title": "N0",
                      "description": "d"}]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual([r["source"] for r in results],
                         ["web", "news", "web", "web"])

    def test_news_is_additive_and_not_capped_into_count(self):
        # `count` is per section. Two of the three datasets are news benchmarks,
        # so news is on-target retrieval; capping it would displace fresh
        # coverage to hold a number. A news-intent query legitimately surfaces
        # up to 2x count.
        response = {"metadata": {}, "results": {
            "web": [{"url": f"https://w.example/{i}", "title": f"W{i}",
                     "description": "d"} for i in range(5)],
            "news": [{"url": f"https://n.example/{i}", "title": f"N{i}",
                      "description": "d"} for i in range(5)]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(len(results), 10)
        self.assertEqual([r["rank"] for r in results], list(range(1, 11)))
        self.assertEqual(sum(1 for r in results if r["source"] == "web"), 5)
        self.assertEqual(sum(1 for r in results if r["source"] == "news"), 5)

    def test_no_news_section_leaves_the_web_surface_untouched(self):
        # An evergreen query returns no news section. The surface must then be
        # exactly the web results -- news being additive must not change the
        # non-news case.
        response = {"metadata": {}, "results": {
            "web": [{"url": f"https://w.example/{i}", "title": f"W{i}",
                     "description": "d"} for i in range(8)], "news": []}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(len(results), 8)
        self.assertEqual({r["source"] for r in results}, {"web"})

    def test_results_without_text_are_surfaced_for_their_date(self):
        # A news result with an empty description still carries a publication
        # timestamp, which is the construct temporal_grounding wants. Keep it.
        response = {"metadata": {}, "results": {"web": [], "news": [
            {"url": "https://n.example/x", "title": "N", "description": "",
             "page_age": "2026-08-17T10:00:00Z"}]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["snippet"], "")
        self.assertEqual(results[0]["published_date"], "2026-08-17T10:00:00Z")

    def test_news_results_without_highlights_use_description(self):
        # News results carry no snippets; without highlights, description is
        # the fallback.
        response = {"metadata": {}, "results": {"web": [], "news": [
            {"url": "https://n.example/x", "title": "N",
             "description": "summary"}]}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["snippet"], "summary")

    def test_web_results_without_highlights_use_all_snippets(self):
        # When highlights are not available, all snippets are joined, not just
        # the first.
        response = {"metadata": {}, "results": {"web": [
            {"url": "https://w.example/x", "title": "W",
             "snippets": ["first", "second"]}], "news": []}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(results[0]["snippet"], "first\nsecond")

    def test_empty_response_yields_no_results(self):
        response = {"metadata": {}, "results": {"web": [], "news": []}}
        with patch.object(run_eval, "_provider_json", return_value=response):
            results, _ = run_eval.youdotcom_search("q", "normalized", [])
        self.assertEqual(results, [])

    def test_request_id_is_captured_without_retaining_the_payload(self):
        _, _, raw = self._call("normalized")
        self.assertEqual(run_eval._provider_request_id(raw), "ydc-search-uuid")


class YouComSetupTest(unittest.TestCase):
    """Each setup must send exactly the parameters that define it — no more."""

    def _body(self, arm):
        with patch.object(run_eval, "_provider_json",
                          return_value=YDC_RESPONSE) as sent:
            run_eval.youdotcom_search("who won", arm, [])
        return sent.call_args[1]["json"]

    def test_normalized_sends_no_freshness_filter(self):
        body = self._body("normalized")
        self.assertNotIn("freshness", body)
        self.assertEqual(body["count"], 5)

    def test_native_fresh_sends_one_day(self):
        self.assertEqual(self._body("native_fresh")["freshness"], "day")

    def test_fresh_week_sends_one_week(self):
        # If this silently matched native_fresh, the window-width comparison
        # would report a null effect that is really a duplicated arm.
        self.assertEqual(self._body("fresh_week")["freshness"], "week")

    def test_historical_setup_sends_an_absolute_date_range(self):
        setup = run_eval.ydc_setup("fresh_week", "2024-02-01")
        with patch.object(run_eval, "_provider_json",
                          return_value=YDC_RESPONSE) as sent:
            run_eval.youdotcom_search("who won in 2024", "fresh_week", [], setup)
        self.assertEqual(
            sent.call_args[1]["json"]["freshness"],
            "2024-01-26to2024-02-01",
        )

    def test_wide_raises_count_without_adding_a_freshness_filter(self):
        body = self._body("wide")
        self.assertEqual(body["count"], 20)
        self.assertNotIn("freshness", body)

    def test_setups_differ_from_each_other(self):
        # The treatment axis is only real if the requests actually differ.
        # json.dumps with sort_keys makes the body hashable for set comparison.
        import json as _json
        sent = {name: _json.dumps(self._body(name), sort_keys=True)
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
        # The per-section count does not change per-call price.
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
