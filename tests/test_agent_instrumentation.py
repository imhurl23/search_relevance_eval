"""Pin the four-cell matrix instrumentation: axes, adapters, and scorer gating.

The failure this suite exists to prevent is silent, not loud. A native-search
arm produces no `search_web` tool spans, so every trajectory-reading scorer used
to return a PASSING score for the wrong reason — leakage_guard 1.0 because it
saw no URLs, budget_economy 1.0 because it saw no searches, search_cost $0.00
because no priced call was made. The native arm would then top the leaderboard
on compliance and cost by construction. Nothing raises; the numbers are just
wrong. So the assertions below are mostly about None, not about values.

Native adapter shapes are asserted against the vendors' published response
schemas:
  Anthropic https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
  OpenAI    https://developers.openai.com/api/docs/guides/tools-web-search
"""

import os
import unittest

os.environ.setdefault("EXA_API_KEY", "test-exa-key")
os.environ.setdefault("PARALLEL_API_KEY", "test-parallel-key")
os.environ.setdefault("YDC_API_KEY", "test-ydc-key")

import agents
import run_eval
import scorers


# --- fixtures ---------------------------------------------------------------

# Shape from the Anthropic web search docs: server_tool_use carries the query,
# web_search_tool_result carries a LIST of web_search_result, and citations ride
# on the text block. Note there is no snippet field anywhere.
ANTHROPIC_RESPONSE = {
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "server_tool_use": {"web_search_requests": 1},
    },
    "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
         "input": {"query": "who won"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
         "content": [
             {"type": "web_search_result", "url": "https://a.example/x",
              "title": "A", "page_age": "2026-07-30T00:00:00Z",
              "encrypted_content": "enc"},
             {"type": "web_search_result", "url": "https://b.example/y",
              "title": "B", "page_age": "2026-07-31T00:00:00Z",
              "encrypted_content": "enc"},
         ]},
        {"type": "text", "text": "Team Alpha",
         "citations": [{"type": "web_search_result_location",
                        "url": "https://a.example/x", "title": "A",
                        "cited_text": "Team Alpha won"}]},
    ],
}

# On an error, `content` is a single object rather than a list.
ANTHROPIC_ERROR_RESPONSE = {
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5,
              "server_tool_use": {"web_search_requests": 1}},
    "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
         "input": {"query": "who won"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
         "content": {"type": "web_search_tool_result_error",
                     "error_code": "max_uses_exceeded"}},
        {"type": "text", "text": "I could not find this."},
    ],
}

# Shape from the OpenAI Responses web search docs: action.sources holds the URLs
# consulted (requires the include flag), url_citation annotations hold titles.
OPENAI_RESPONSE = {
    "usage": {"input_tokens": 200, "output_tokens": 30},
    "output": [
        {"type": "web_search_call", "status": "completed",
         "action": {"type": "search", "query": "who won",
                    "sources": [{"type": "url", "url": "https://a.example/x"},
                                {"type": "url", "url": "https://b.example/y"}]}},
        {"type": "message", "content": [
            {"type": "output_text", "text": "Team Alpha",
             "annotations": [{"type": "url_citation", "url": "https://a.example/x",
                              "title": "A", "start_index": 0, "end_index": 5}]},
        ]},
    ],
}


class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeAnthropicMessages(response)


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def _row_output(surface, results=None, used_searches=0):
    trajectory = []
    if results is not None:
        trajectory = [{"type": "search", "query": "q", "tokens": 0,
                       "results": results}]
    return {"final_answer": "Team Alpha", "trajectory": trajectory,
            "decision_surface": surface, "used_searches": used_searches,
            "used_clicks": 0}


LNB_METADATA = {
    "livenewsbench_release": "2026-07",
    "link": "https://source.example/article",
    "event_date": "2026-07-29",
    "search_budget": 5,
}


# --- axes -------------------------------------------------------------------

class VendorRegistryTest(unittest.TestCase):
    def test_model_class_split_matches_the_matrix(self):
        self.assertEqual(agents.VENDORS["baseten"].model_class, "oss")
        self.assertEqual(agents.VENDORS["openai"].model_class, "frontier")
        self.assertEqual(agents.VENDORS["anthropic"].model_class, "frontier")

    def test_only_frontier_vendors_offer_native_search(self):
        # The oss x native cell does not exist: no OSS serving vendor runs
        # server-side search. Asserted so the matrix cannot silently gain a cell
        # that would be filled by something else.
        self.assertFalse(agents.VENDORS["baseten"].supports_native_search)
        self.assertTrue(agents.VENDORS["openai"].supports_native_search)
        self.assertTrue(agents.VENDORS["anthropic"].supports_native_search)

    def test_no_frontier_vendor_sends_sampling_params(self):
        # Both frontier vendors reject them: Opus 5 400s on
        # temperature/top_p/top_k, and gpt-5-family models 400 on `temperature`
        # ("only the default (1) is supported") and support no `seed`. Sending
        # temperature=0 to either would fail every row of those arms, so
        # "temp 0, seed 42" is unavailable, not merely unused.
        for vendor in ("openai", "anthropic"):
            with self.subTest(vendor=vendor):
                self.assertEqual(agents.VENDORS[vendor].sampling, {})
                self.assertFalse(agents.VENDORS[vendor].seed_supported)

    def test_only_the_oss_arm_can_pin_sampling(self):
        self.assertEqual(agents.VENDORS["baseten"].sampling, {"temperature": 0})
        self.assertFalse(agents.VENDORS["baseten"].seed_supported)

    def test_frontier_models_are_pinned_snapshots_not_moving_aliases(self):
        # `gpt-5.6` is an alias for gpt-5.6-sol and will move to whatever sol
        # becomes; an eval condition has to name the snapshot. The two frontier
        # models are NOT asserted to be capability-equivalent — see the vendor
        # registry comment; no primary contrast depends on that.
        self.assertEqual(agents.VENDORS["openai"].default_model, "gpt-5.6-sol")
        self.assertNotEqual(agents.VENDORS["openai"].default_model, "gpt-5.6")

    def test_anthropic_default_is_the_flagship_pairing(self):
        # Paired with gpt-5.6-sol on within-lineup position: each is its vendor's
        # most capable widely released model. claude-opus-5 stays selectable via
        # --agent-model when the extra capability is not worth 2x.
        self.assertEqual(agents.VENDORS["anthropic"].default_model,
                         "claude-fable-5")
        self.assertIn("claude-opus-5", agents.MODEL_USD_PER_MTOK)

    def test_effort_is_pinned_only_where_it_coexists_with_tools(self):
        # Anthropic can pin effort on both its arms. OpenAI cannot: reasoning
        # models reject reasoning_effort alongside function tools on chat
        # completions, so pinning it on the native arm alone would make effort
        # differ between OpenAI's own native and harness arms.
        self.assertEqual(agents.VENDORS["anthropic"].reasoning_effort, "high")
        self.assertIsNone(agents.VENDORS["openai"].reasoning_effort)

    def test_openai_native_search_budget_is_not_api_enforced(self):
        # Anthropic takes max_uses; OpenAI's hosted web_search publishes no
        # equivalent, so its native arm can exceed the 5-search cap every other
        # arm is held to. Recorded, not assumed away.
        self.assertTrue(agents.NATIVE_BUDGET_ENFORCED["anthropic"])
        self.assertFalse(agents.NATIVE_BUDGET_ENFORCED["openai"])

    def test_date_field_semantics_separate_publication_from_last_modified(self):
        # temporal_grounding reads `published_date`, but two surfaces report
        # last-modified instead — a re-rendered page looks fresh without
        # carrying new information. Pooling the two would inflate freshness.
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["exa"], "publication")
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["parallel"], "publication")
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["youdotcom"], "last_modified")
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["anthropic_native"],
                         "last_modified")
        self.assertIsNone(agents.DATE_FIELD_SEMANTICS["openai_native"])

    def test_baseten_uses_its_documented_openai_compatible_base_url(self):
        self.assertEqual(agents.VENDORS["baseten"].base_url,
                         "https://inference.baseten.co/v1")
        self.assertEqual(agents.VENDORS["baseten"].api_key_env, "BASETEN_API_KEY")


class SurfaceConstantsTest(unittest.TestCase):
    def test_agents_and_scorers_agree_on_the_tier_names(self):
        # agents.py sets the tier and scorers.py gates on it, but scorers.py
        # deliberately does not import agents (that would pull the openai SDK
        # into the deployed scorer bundle). The strings are therefore duplicated,
        # and a rename on one side would silently send every gated scorer down
        # its fallback path. This test is the seam.
        self.assertEqual(
            (agents.SURFACE_FULL, agents.SURFACE_NO_SNIPPET,
             agents.SURFACE_URLS_ONLY, agents.SURFACE_NONE),
            (scorers.SURFACE_FULL, scorers.SURFACE_NO_SNIPPET,
             scorers.SURFACE_URLS_ONLY, scorers.SURFACE_NONE))

    def test_every_tier_is_known_to_the_scorers(self):
        for surface in (agents.SURFACE_FULL, agents.SURFACE_NO_SNIPPET,
                        agents.SURFACE_URLS_ONLY, agents.SURFACE_NONE):
            with self.subTest(surface=surface):
                self.assertIn(surface, scorers._KNOWN_SURFACES)


class ConditionLabelTest(unittest.TestCase):
    def test_native_search_and_native_fresh_get_distinct_labels(self):
        # The one naming collision that would corrupt every slice: `native_fresh`
        # is a SEARCH API's freshness parameter; `native` is the MODEL vendor's
        # own search. They must never produce the same condition.
        harness = run_eval.condition_label(
            agents.SEARCH_MODE_HARNESS, "exa", "native_fresh", "openai")
        native = run_eval.condition_label(
            agents.SEARCH_MODE_NATIVE, "exa", "normalized", "openai")
        self.assertEqual(harness, "harness-exa-native_fresh")
        self.assertEqual(native, "native-openai")
        self.assertNotEqual(harness, native)

    def test_native_label_names_the_vendor(self):
        # "native" alone is not a condition — it is vendor-specific, and the two
        # native arms are not interchangeable evidence.
        self.assertNotEqual(
            run_eval.condition_label(agents.SEARCH_MODE_NATIVE, "exa",
                                     "normalized", "openai"),
            run_eval.condition_label(agents.SEARCH_MODE_NATIVE, "exa",
                                     "normalized", "anthropic"))


# --- native adapters --------------------------------------------------------

class AnthropicNativeSearchTest(unittest.TestCase):
    def test_normalizes_results_onto_the_harness_trajectory_schema(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(len(run.trajectory), 1)
        results = run.trajectory[0]["results"]
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertEqual(results[0]["url"], "https://a.example/x")
        self.assertEqual(results[0]["published_date"], "2026-07-30T00:00:00Z")
        self.assertEqual(run.final_answer, "Team Alpha")

    def test_declares_a_no_snippet_surface_and_leaves_snippets_empty(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(run.surface, agents.SURFACE_NO_SNIPPET)
        for result in run.trajectory[0]["results"]:
            self.assertEqual(result["snippet"], "")

    def test_cited_text_is_not_promoted_into_a_snippet(self):
        # cited_text exists and contains the gold answer, but it is selected
        # BECAUSE it supports the answer. Mapping it to `snippet` would make
        # snippet_sufficiency score ~1.0 on this arm by construction.
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(run.citations[0]["cited_text"], "Team Alpha won")
        blob = " ".join(r["snippet"] for r in run.trajectory[0]["results"])
        self.assertNotIn("Team Alpha", blob)

    def test_sends_max_uses_and_blocked_domains(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?",
            ["source.example", "web.archive.org"], 5)
        tool = client.messages.calls[0]["tools"][0]
        self.assertEqual(tool["type"], agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE)
        self.assertEqual(tool["max_uses"], 5)
        self.assertEqual(tool["blocked_domains"],
                         ["source.example", "web.archive.org"])

    def test_uses_the_basic_tool_not_dynamic_filtering(self):
        # Dynamic filtering (_20260209+) drops results before they reach the
        # context window, so the model would answer from a surface we cannot
        # observe and that does not match the harness arms.
        self.assertEqual(agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
                         "web_search_20250305")

    def test_thinking_stays_enabled(self):
        # Disabling thinking on Opus 5 can emit a tool call as visible text: the
        # search silently never runs and nothing errors, which on this eval
        # manufactures no-search rows inside a search arm.
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(client.messages.calls[0]["thinking"]["type"], "adaptive")

    def test_prefers_the_vendor_reported_search_count(self):
        # A search whose results were dropped from the response still spent
        # budget and still costs money, so trust usage over parsed blocks.
        client = FakeAnthropicClient(ANTHROPIC_ERROR_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(run.trajectory, [])
        self.assertEqual(run.n_searches, 1)

    def test_records_search_errors_instead_of_reading_them_as_empty_results(self):
        client = FakeAnthropicClient(ANTHROPIC_ERROR_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertEqual(run.search_errors,
                         [{"query": "who won", "error_code": "max_uses_exceeded"}])

    def test_refusal_is_flagged_rather_than_scored_as_a_wrong_answer(self):
        client = FakeAnthropicClient(dict(ANTHROPIC_RESPONSE,
                                          stop_reason="refusal"))
        run = agents.anthropic_native_search(
            client, "claude-opus-5", "sys", "who won?", [], 5)
        self.assertTrue(run.refused)


class OpenAINativeSearchTest(unittest.TestCase):
    def test_normalizes_sources_onto_the_trajectory_schema(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        run = agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", [])
        self.assertEqual(len(run.trajectory), 1)
        results = run.trajectory[0]["results"]
        self.assertEqual([r["url"] for r in results],
                         ["https://a.example/x", "https://b.example/y"])
        self.assertEqual(run.final_answer, "Team Alpha")

    def test_declares_urls_only_because_there_are_no_dates(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        run = agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", [])
        self.assertEqual(run.surface, agents.SURFACE_URLS_ONLY)
        for result in run.trajectory[0]["results"]:
            self.assertIsNone(result["published_date"])

    def test_requests_sources_so_leakage_is_not_scored_off_citations_alone(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(client, "gpt-5.6", "sys", "who won?", [])
        call = client.responses.calls[0]
        self.assertIn("web_search_call.action.sources", call["include"])

    def test_sends_blocked_domains_through_the_filters_object(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", ["source.example"])
        tool = client.responses.calls[0]["tools"][0]
        self.assertEqual(tool["type"], "web_search")
        self.assertEqual(tool["filters"]["blocked_domains"], ["source.example"])
        self.assertEqual(tool["search_context_size"],
                         agents.OPENAI_SEARCH_CONTEXT_SIZE)

    def test_backfills_titles_from_citations_only(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        run = agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", [])
        results = run.trajectory[0]["results"]
        self.assertEqual(results[0]["title"], "A")
        # Uncited results stay untitled — the reason this arm is urls_only.
        self.assertEqual(results[1]["title"], "")


class NativeSearchPricingTest(unittest.TestCase):
    def test_both_native_arms_price_at_the_published_ten_dollars_per_thousand(self):
        anthropic_rate, anthropic_ok = agents.native_search_rate_usd(
            "anthropic", "claude-opus-5")
        openai_rate, openai_ok = agents.native_search_rate_usd(
            "openai", "gpt-5.6")
        self.assertEqual(anthropic_rate, 0.010)
        self.assertEqual(openai_rate, 0.010)
        self.assertTrue(anthropic_ok and openai_ok)

    def test_unrecognized_openai_model_is_priced_high_not_cheap(self):
        # A non-reasoning model routes through web_search_preview at $25/1k. An
        # unrecognized model must not default to the cheaper rate, which would
        # understate the native arm's cost by 2.5x.
        rate, confirmed = agents.native_search_rate_usd("openai", "gpt-4o")
        self.assertEqual(rate, 0.025)
        self.assertFalse(confirmed)


# --- scorer gating ----------------------------------------------------------

class ScorerSurfaceGatingTest(unittest.TestCase):
    """Every one of these asserted None used to be a passing or failing score."""

    def test_leakage_guard_is_unmeasurable_on_the_no_search_arm(self):
        result = scorers.leakage_guard(
            {}, _row_output(scorers.SURFACE_NONE), "Team Alpha",
            metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

    def test_leakage_guard_still_measures_native_arms(self):
        # Both native vendors accept a blocked-domain list and return URLs, so
        # the leakage rule genuinely applies to them.
        clean = scorers.leakage_guard(
            {}, _row_output(scorers.SURFACE_URLS_ONLY,
                            [{"rank": 1, "url": "https://other.example/a"}],
                            used_searches=1),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertEqual(clean["score"], 1.0)
        leaked = scorers.leakage_guard(
            {}, _row_output(scorers.SURFACE_NO_SNIPPET,
                            [{"rank": 1, "url": "https://source.example/article"}],
                            used_searches=1),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertEqual(leaked["score"], 0.0)

    def test_leakage_guard_does_not_pass_a_search_that_surfaced_nothing(self):
        # Searches ran but no URLs came back to inspect. That is a dropped or
        # filtered response, not a clean SERP.
        result = scorers.leakage_guard(
            {}, _row_output(scorers.SURFACE_URLS_ONLY, [], used_searches=3),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

    def test_budget_economy_is_unmeasurable_without_tools(self):
        result = scorers.budget_economy(
            {}, _row_output(scorers.SURFACE_NONE), "Team Alpha",
            metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

    def test_budget_economy_reads_the_cap_from_the_run(self):
        output = _row_output(scorers.SURFACE_FULL, [], used_searches=4)
        self.assertEqual(
            scorers.budget_economy({}, output, "x",
                                   metadata={"search_budget": 3})["score"], 0.0)
        self.assertEqual(
            scorers.budget_economy({}, output, "x",
                                   metadata={"search_budget": 5})["score"], 1.0)

    def test_temporal_grounding_is_unmeasurable_without_result_dates(self):
        result = scorers.temporal_grounding(
            {}, _row_output(scorers.SURFACE_URLS_ONLY,
                            [{"rank": 1, "url": "https://a.example/x"}],
                            used_searches=1),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

    def test_temporal_grounding_still_works_on_the_anthropic_native_arm(self):
        result = scorers.temporal_grounding(
            {}, _row_output(scorers.SURFACE_NO_SNIPPET, [
                {"rank": 1, "url": "https://a.example/x",
                 "published_date": "2026-07-30"}], used_searches=1),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertEqual(result["score"], 1.0)

    def test_snippet_scorers_return_none_rather_than_a_false_zero(self):
        output = _row_output(scorers.SURFACE_NO_SNIPPET, [
            {"rank": 1, "url": "https://a.example/x", "title": "A",
             "snippet": "", "published_date": "2026-07-30"}], used_searches=1)
        for scorer in (scorers.snippet_sufficiency, scorers.evidence_precision,
                       scorers.token_discounted_gain,
                       scorers.compression_redundancy):
            with self.subTest(scorer=scorer.__name__):
                self.assertIsNone(scorer({}, output, "Team Alpha")["score"])

    def test_domain_entropy_survives_a_urls_only_surface(self):
        result = scorers.domain_entropy({}, _row_output(
            scorers.SURFACE_URLS_ONLY,
            [{"rank": 1, "url": "https://a.example/x"},
             {"rank": 2, "url": "https://b.example/y"}], used_searches=1), "x")
        self.assertEqual(result["score"], 1.0)

    def test_dealbreaker_gate_does_not_pass_when_no_rule_was_evaluated(self):
        # The compounding bug: leakage_guard and budget_economy both returning
        # 1.0 on an unobservable arm made the gate report a clean pass with zero
        # rules checked, which is what a gated headline number multiplies by.
        result = scorers.dealbreaker_gate(
            {}, _row_output(scorers.SURFACE_NONE), "Team Alpha",
            metadata=LNB_METADATA)
        self.assertIsNone(result["score"])
        self.assertEqual(result["metadata"]["gates_checked"], [])

    def test_gated_answer_match_zeroes_a_leaking_row(self):
        leaked = _row_output(
            scorers.SURFACE_FULL,
            [{"rank": 1, "url": "https://source.example/article", "title": "A",
              "snippet": "Team Alpha won", "published_date": "2026-07-30"}],
            used_searches=1)
        result = scorers.gated_answer_match({}, leaked, "Team Alpha",
                                            metadata=LNB_METADATA)
        # The answer is right and the row is still zero — that is the point of
        # gating, and it is what keeps a leak from being averaged away.
        self.assertEqual(result["metadata"]["answer_score"], 1.0)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("leakage_guard", result["metadata"]["violated_rules"])

    def test_gated_answer_match_keeps_the_clean_row(self):
        clean = _row_output(
            scorers.SURFACE_FULL,
            [{"rank": 1, "url": "https://other.example/a", "title": "A",
              "snippet": "Team Alpha won", "published_date": "2026-07-30"}],
            used_searches=1)
        result = scorers.gated_answer_match({}, clean, "Team Alpha",
                                            metadata=LNB_METADATA)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metadata"]["gate_applied"])

    def test_gated_answer_match_keeps_the_ungatable_control_arm(self):
        # The no-tool arm cannot violate a retrieval rule. Dropping it to None
        # would delete the parametric baseline that every search arm is measured
        # against, so the answer score passes through, marked ungated.
        control = _row_output(scorers.SURFACE_NONE)
        result = scorers.gated_answer_match({}, control, "Team Alpha",
                                            metadata=LNB_METADATA)
        self.assertEqual(result["score"], 1.0)
        self.assertFalse(result["metadata"]["gate_applied"])

    def test_gated_answer_match_is_none_without_a_gold_answer(self):
        result = scorers.gated_answer_match(
            {}, _row_output(scorers.SURFACE_FULL, [], used_searches=1), None,
            metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

    def test_legacy_rows_without_a_declared_surface_still_score(self):
        # Rows written before the tier existed always carried full harness
        # results, so they must not drop out of every trajectory metric.
        legacy = {"final_answer": "Team Alpha", "used_searches": 1,
                  "used_clicks": 0,
                  "trajectory": [{"type": "search", "query": "q", "tokens": 10,
                                  "results": [{"rank": 1, "title": "A",
                                               "url": "https://a.example/x",
                                               "snippet": "Team Alpha won",
                                               "published_date": "2026-07-30"}]}]}
        self.assertEqual(scorers._surface(legacy), scorers.SURFACE_FULL)
        self.assertEqual(
            scorers.snippet_sufficiency({}, legacy, "Team Alpha")["score"], 1.0)


# --- CLI wiring -------------------------------------------------------------

class ModelCostTest(unittest.TestCase):
    """Search fees are the small half of the bill, so a search-fee-only
    comparison ranks arms on the wrong quantity."""

    def test_priced_models_produce_a_token_cost(self):
        usd, confirmed = agents.model_cost_usd("claude-fable-5", 1_000_000, 0)
        self.assertTrue(confirmed)
        self.assertAlmostEqual(usd, 10.00)
        usd, _ = agents.model_cost_usd("gpt-5.6-sol", 0, 1_000_000)
        self.assertAlmostEqual(usd, 30.00)

    def test_unpriced_model_returns_none_not_zero(self):
        # 0.0 would place an unpriced arm at the origin of a cost frontier,
        # reading as free. None keeps it out of the frontier entirely.
        usd, confirmed = agents.model_cost_usd("some/unlisted-model", 1000, 1000)
        self.assertIsNone(usd)
        self.assertFalse(confirmed)

    def test_the_oss_arm_is_priced_so_substitution_is_a_cost_claim(self):
        # "OSS + retrieval vs frontier without it" is a cost-ratio claim, and a
        # cost-ratio claim needs both sides priced.
        usd, confirmed = agents.model_cost_usd("openai/gpt-oss-120b", 1000, 1000)
        self.assertTrue(confirmed)
        self.assertGreater(usd, 0)

    def test_oss_input_tokens_are_two_orders_cheaper_than_the_frontier_default(self):
        oss_in = agents.MODEL_USD_PER_MTOK["openai/gpt-oss-120b"][0]
        frontier_in = agents.MODEL_USD_PER_MTOK["claude-fable-5"][0]
        self.assertAlmostEqual(frontier_in / oss_in, 100.0)

    def test_inference_dominates_a_realistic_native_row(self):
        # Sanity-check the finding this instrumentation exists to expose: five
        # native searches at $10/1k is $0.05, which a single search-heavy turn's
        # tokens should dwarf on a frontier model.
        search = 5 * agents.NATIVE_SEARCH_USD_PER_CALL["anthropic"]
        inference, _ = agents.model_cost_usd("claude-fable-5", 60_000, 1_500)
        self.assertGreater(inference, search)


class FakeHooks:
    def __init__(self, metadata, expected):
        self.metadata = dict(metadata)
        self.expected = expected
        self.trial_index = 0


class TaskWiringTest(unittest.TestCase):
    """End-to-end: the parts above are individually correct, but the bug lived
    in the assembly — whether `decision_surface` actually reaches the output
    payload the scorers read, and whether cost/axes land in metadata."""

    def setUp(self):
        agents.reset_clients()
        self.addCleanup(agents.reset_clients)

    def _run_task(self, vendor, response, search_mode=agents.SEARCH_MODE_NATIVE):
        client = (FakeAnthropicClient(response) if vendor == "anthropic"
                  else FakeOpenAIClient(response))
        original = run_eval.get_agent_client
        run_eval.get_agent_client = lambda v: client
        self.addCleanup(setattr, run_eval, "get_agent_client", original)
        task = run_eval.make_task(
            provider="exa", arm="normalized",
            agent_model=agents.VENDORS[vendor].default_model,
            search_mode=search_mode, model_vendor=vendor)
        hooks = FakeHooks(LNB_METADATA, "Team Alpha")
        return task({"question": "who won?"}, hooks), hooks

    def test_native_output_carries_the_surface_the_scorers_gate_on(self):
        output, _ = self._run_task("anthropic", ANTHROPIC_RESPONSE)
        self.assertEqual(output["decision_surface"], scorers.SURFACE_NO_SNIPPET)
        # And the scorers actually honor it end to end.
        self.assertIsNone(
            scorers.snippet_sufficiency({}, output, "Team Alpha")["score"])
        self.assertEqual(
            scorers.temporal_grounding({}, output, "Team Alpha",
                                       metadata=LNB_METADATA)["score"], 1.0)

    def test_native_arm_is_not_free(self):
        # search_cost_usd used to be $0.00 on every native row, making the arm
        # look free next to the harness arms.
        output, hooks = self._run_task("anthropic", ANTHROPIC_RESPONSE)
        self.assertEqual(output["used_searches"], 1)
        self.assertEqual(hooks.metadata["search_mode"], "native")
        # 1 search x $10/1k.
        self.assertAlmostEqual(
            agents.native_search_rate_usd("anthropic", "claude-opus-5")[0], 0.010)

    def test_axes_land_in_row_metadata(self):
        _, hooks = self._run_task("openai", OPENAI_RESPONSE)
        self.assertEqual(hooks.metadata["model_class"], "frontier")
        self.assertEqual(hooks.metadata["model_vendor"], "openai")
        self.assertEqual(hooks.metadata["search_mode"], "native")
        self.assertEqual(hooks.metadata["search_provider"], "openai_native")
        # Freshness is a harness-only treatment; leaving `normalized` here would
        # imply the native arm received a treatment it has no knob for.
        self.assertIsNone(hooks.metadata["freshness_treatment"])
        self.assertEqual(hooks.metadata["decision_surface"],
                         scorers.SURFACE_URLS_ONLY)

    def test_zero_search_row_flags_a_search_arm_that_never_searched(self):
        # The confound that voids a search arm: tool available, never used. This
        # is how weak OSS tool-calling silently turns the OSS+harness cell into
        # the OSS+none cell.
        empty = {"usage": {"input_tokens": 5, "output_tokens": 5},
                 "output": [{"type": "message", "content": [
                     {"type": "output_text", "text": "Team Alpha",
                      "annotations": []}]}]}
        _, hooks = self._run_task("openai", empty)
        self.assertTrue(hooks.metadata["zero_search_row"])

        _, searched = self._run_task("openai", OPENAI_RESPONSE)
        self.assertFalse(searched.metadata["zero_search_row"])

    def test_exclusion_is_enforced_on_the_native_arms(self):
        _, hooks = self._run_task("anthropic", ANTHROPIC_RESPONSE)
        self.assertTrue(hooks.metadata["exclusion_enforced"])
        self.assertIn("source.example",
                      [d for d in hooks.metadata["excluded_source_domains"]])


CORVUS_METADATA = {
    "dataset": "Corvus-QA",
    "attribute": "ceo_of",
    "entity_type": "company",
    "answer_class": "person",
    "recency_rung": "within_7d",
    "coverage_tier": "answerable",
    "event_date": "2026-07-28",
    "answer_aliases": ["C. CEO"],
    "articles": [{"url": "https://authority.example/event"}],
    "search_budget": 5,
}


class SecondDomainTest(unittest.TestCase):
    """Corvus-QA is the second domain. One domain cannot establish that a
    retrieval effect generalizes — it can be positive in one and absent in
    another, so a single-domain result is a single-domain result."""

    def test_leakage_rule_applies_to_corvus_rows(self):
        # Previously gated on `livenewsbench_release`, so Corvus rows returned
        # None -- while run_eval was already excluding their source domains at
        # search time. The rule was enforced and never verified.
        leaked = _row_output(
            scorers.SURFACE_FULL,
            [{"rank": 1, "url": "https://authority.example/event",
              "snippet": "x", "published_date": "2026-07-29"}],
            used_searches=1)
        result = scorers.leakage_guard({}, leaked, "Current CEO",
                                       metadata=CORVUS_METADATA)
        self.assertEqual(result["score"], 0.0)

    def test_rows_without_source_domains_stay_unmeasurable(self):
        # RetrievalQA carries no source URLs. With nothing to check, 1.0 would
        # assert a compliance nothing established.
        result = scorers.leakage_guard(
            {}, _row_output(scorers.SURFACE_FULL,
                            [{"rank": 1, "url": "https://a.example/x"}],
                            used_searches=1),
            ["Team Alpha"], metadata={"source_revision": "abc"})
        self.assertIsNone(result["score"])

    def test_curated_aliases_are_used_when_the_dataset_supplies_them(self):
        # Derived string variants cannot know "C. CEO" names the same person as
        # "Current CEO". Ignoring curated aliases understates every
        # surface/evidence metric, and unevenly across datasets -- which would
        # read as a domain effect.
        output = _row_output(
            scorers.SURFACE_FULL,
            [{"rank": 1, "url": "https://other.example/a", "title": "t",
              "snippet": "the board named C. CEO to the role",
              "published_date": "2026-07-29"}],
            used_searches=1)
        with_aliases = scorers.snippet_sufficiency(
            {}, output, "Current CEO", metadata=CORVUS_METADATA)
        without = scorers.snippet_sufficiency({}, output, "Current CEO")
        self.assertEqual(with_aliases["score"], 1.0)
        self.assertEqual(without["score"], 0.0)

    def test_corvus_subgroup_variables_reach_row_metadata(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        original = run_eval.get_agent_client
        run_eval.get_agent_client = lambda v: client
        self.addCleanup(setattr, run_eval, "get_agent_client", original)
        task = run_eval.make_task(
            provider="exa", arm="normalized", agent_model="claude-fable-5",
            search_mode=agents.SEARCH_MODE_NATIVE, model_vendor="anthropic")
        hooks = FakeHooks(CORVUS_METADATA, "Current CEO")
        task({"question": "who is CEO?"}, hooks)
        self.assertEqual(hooks.metadata["dataset_family"], "Corvus-QA")
        self.assertEqual(hooks.metadata["recency_rung"], "within_7d")
        self.assertEqual(hooks.metadata["coverage_tier"], "answerable")
        # Would otherwise have been "uncategorized", making the domain unsliceable.
        self.assertEqual(hooks.metadata["benchmark_category"], "ceo_of")


class ClientConstructionTest(unittest.TestCase):
    def setUp(self):
        agents.reset_clients()
        self.addCleanup(agents.reset_clients)

    def test_missing_key_raises_a_named_error_not_an_sdk_traceback(self):
        os.environ.pop("BASETEN_API_KEY", None)
        with self.assertRaises(SystemExit) as caught:
            agents.get_client("baseten", lambda c: c)
        self.assertIn("BASETEN_API_KEY", str(caught.exception))

    def test_unknown_vendor_is_rejected(self):
        with self.assertRaises(ValueError):
            agents.vendor_of("together")


class PreflightTest(unittest.TestCase):
    def _set_env(self, name, value):
        """Restore the ambient value so these cases cannot order-depend."""
        previous = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        self.addCleanup(
            lambda: os.environ.__setitem__(name, previous) if previous is not None
            else os.environ.pop(name, None))

    def test_native_search_is_refused_for_the_oss_vendor(self):
        self._set_env("BASETEN_API_KEY", "test-baseten-key")
        with self.assertRaises(SystemExit) as caught:
            run_eval._preflight("exa", "normalized", agents.SEARCH_MODE_NATIVE,
                                "baseten", "openai/gpt-oss-120b")
        self.assertIn("native", str(caught.exception))

    def test_missing_vendor_key_fails_before_spending_money(self):
        self._set_env("ANTHROPIC_API_KEY", None)
        with self.assertRaises(SystemExit) as caught:
            run_eval._preflight("exa", "normalized", agents.SEARCH_MODE_NONE,
                                "anthropic", "claude-opus-5")
        self.assertIn("ANTHROPIC_API_KEY", str(caught.exception))

    def test_no_search_arm_does_not_require_a_search_provider_key(self):
        self._set_env("ANTHROPIC_API_KEY", "test-anthropic-key")
        self._set_env("EXA_API_KEY", None)
        run_eval._preflight("exa", "no_search", agents.SEARCH_MODE_NONE,
                            "anthropic", "claude-opus-5")


if __name__ == "__main__":
    unittest.main()
