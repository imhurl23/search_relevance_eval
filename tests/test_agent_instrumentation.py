"""Pin the matrix instrumentation: axes, adapters, model rows, and scorer gating.

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

import dataclasses
import os
import unittest
from collections import Counter
from unittest import mock

import httpx

os.environ.setdefault("YDC_API_KEY", "test-ydc-key")

import agents
import import_retrievalqa
import run_eval
import run_matrix
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


class FakeAnthropicMessagesSequence:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("messages.create made more requests than scripted")
        return self.responses.pop(0)


class FakeSequencedAnthropicClient:
    def __init__(self, responses):
        self.messages = FakeAnthropicMessagesSequence(responses)


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
    def test_finalized_matrix_has_fourteen_conditions(self):
        self.assertEqual(len(run_matrix.MATRIX), 14)
        labels = [condition.label for condition in run_matrix.MATRIX]
        self.assertEqual(len(labels), len(set(labels)))

    def test_finalized_matrix_uses_only_selected_search_arms(self):
        harness_arms = {
            condition.arm for condition in run_matrix.MATRIX
            if condition.search_mode == "harness"
        }
        self.assertEqual(harness_arms, {"normalized", "wide"})

    def test_finalized_matrix_condition_counts_by_model(self):
        counts = Counter(condition.model for condition in run_matrix.MATRIX)
        self.assertEqual(counts["deepseek-ai/DeepSeek-V4-Flash-0731"], 3)
        self.assertEqual(counts["zai-org/GLM-5.2"], 3)
        self.assertEqual(counts["gpt-5.6-terra"], 4)
        self.assertEqual(counts["claude-sonnet-5"], 4)

    def test_matrix_order_is_reproducible_and_seeded(self):
        first = run_matrix.ordered_matrix("study-a")
        self.assertEqual(first, run_matrix.ordered_matrix("study-a"))
        self.assertNotEqual(first, run_matrix.ordered_matrix("study-b"))
        self.assertCountEqual(first, run_matrix.MATRIX)

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
        # Both frontier vendors reject them: Sonnet 5 400s on
        # temperature/top_p/top_k, and gpt-5-family models 400 on `temperature`
        # ("only the default (1) is supported") and support no `seed`. Sending
        # temperature=0 to either would fail every row of those arms. No vendor
        # here supports `seed` either, so "temp 0, seed 42" is unavailable.
        for vendor in ("openai", "anthropic"):
            with self.subTest(vendor=vendor):
                self.assertEqual(agents.VENDORS[vendor].sampling, {})

    def test_only_the_oss_arm_can_pin_sampling(self):
        self.assertEqual(agents.VENDORS["baseten"].sampling, {"temperature": 0})

    def test_frontier_models_are_pinned_snapshots_not_moving_aliases(self):
        # `gpt-5.6` is a moving alias; an eval condition has to name the snapshot.
        # The frontier models are not asserted to be capability-equivalent; no
        # primary contrast depends on that.
        self.assertEqual(agents.VENDORS["openai"].default_model, "gpt-5.6-terra")
        self.assertNotEqual(agents.VENDORS["openai"].default_model, "gpt-5.6")

    def test_anthropic_default_is_sonnet(self):
        self.assertEqual(agents.VENDORS["anthropic"].default_model,
                         "claude-sonnet-5")

    def test_default_judge_is_luna(self):
        self.assertEqual(scorers.DEFAULT_JUDGE_MODEL, "gpt-5.6-luna")

    def test_every_frontier_vendor_pins_effort_on_all_of_its_arms(self):
        # This assertion is inverted from its previous form, which required
        # OpenAI's effort to be None. That was a workaround for chat completions
        # rejecting reasoning_effort alongside function tools — and it did not
        # even work, because the model's default effort is not 'none' and the
        # endpoint rejected the harness arm anyway. Both vendors now run their
        # harness arm on an endpoint that accepts effort with tools, so effort is
        # declared on every arm. See HarnessProtocolRegistryTest.
        for vendor in ("openai", "anthropic"):
            with self.subTest(vendor=vendor):
                self.assertEqual(agents.VENDORS[vendor].reasoning_effort, "high")

    def test_the_oss_arm_leaves_effort_unpinned(self):
        # Open models expose no comparable knob, so there is nothing to declare.
        self.assertIsNone(agents.VENDORS["baseten"].reasoning_effort)

    def test_both_native_search_budgets_are_api_enforced(self):
        self.assertTrue(agents.NATIVE_BUDGET_ENFORCED["anthropic"])
        self.assertTrue(agents.NATIVE_BUDGET_ENFORCED["openai"])

    def test_native_result_count_target_is_ten(self):
        # Neither native API exposes an exact result-count knob. Ten remains the
        # registered target matching You.com's five web plus five news maxima.
        self.assertEqual(run_eval.NATIVE_RESULT_COUNT_TARGET, 10)

    def test_operator_policy_is_shared_by_harness_and_native_prompts(self):
        policy = "Do not use search operators"
        phrase = "Write the query as a plain natural-language phrase."
        for prompt in (run_eval.FROZEN_SYSTEM_PROMPT,
                       run_eval.NATIVE_SEARCH_SYSTEM_PROMPT):
            self.assertIn(policy, prompt)
            self.assertIn(phrase, prompt)

    def test_native_arms_run_at_their_retrieval_ceiling(self):
        # The harness arm passes every highlight through untruncated. Anything
        # below "high" here compares You.com-at-maximum against OpenAI-at-mid
        # and reads a configuration choice as a retrieval difference.
        self.assertEqual(agents.OPENAI_SEARCH_CONTEXT_SIZE, "high")

    def test_dynamic_filtering_is_pinned_off(self):
        # ["direct"] is already the default for web_search_20250305 but flips to
        # code execution on _20260209+, where the subset of results the model
        # actually consumed stops being observable. Pinned so a version bump
        # cannot change the decision surface silently.
        self.assertEqual(agents.ANTHROPIC_WEB_SEARCH_ALLOWED_CALLERS, ["direct"])
        self.assertEqual(agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
                         "web_search_20250305")

    def test_provider_date_semantics_are_declared(self):
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["youdotcom"],
                         "sectioned_provider_page_age")
        self.assertEqual(agents.DATE_FIELD_SEMANTICS["anthropic_native"],
                         "last_updated")
        self.assertIsNone(agents.DATE_FIELD_SEMANTICS["openai_native"])

    def test_removed_providers_are_gone_from_every_registry(self):
        # A leftover entry would let a run record a provider it can no longer call.
        for gone in ("exa", "parallel"):
            with self.subTest(provider=gone):
                self.assertNotIn(gone, agents.DATE_FIELD_SEMANTICS)


class ConditionLabelTest(unittest.TestCase):
    def test_native_search_and_native_fresh_get_distinct_labels(self):
        # The one naming collision that would corrupt every slice: `native_fresh`
        # is a SEARCH API's freshness parameter; `native` is the MODEL vendor's
        # own search. They must never produce the same condition.
        harness = run_eval.condition_label(
            agents.SEARCH_MODE_HARNESS, "native_fresh", "openai")
        native = run_eval.condition_label(
            agents.SEARCH_MODE_NATIVE, run_eval.DEFAULT_ARM, "openai")
        self.assertEqual(harness, "harness-youdotcom-native_fresh")
        self.assertEqual(native, "native-openai")
        self.assertNotEqual(harness, native)

    def test_native_label_names_the_vendor(self):
        # "native" alone is not a condition — it is vendor-specific, and the two
        # native arms are not interchangeable evidence.
        self.assertNotEqual(
            run_eval.condition_label(agents.SEARCH_MODE_NATIVE,
                                     run_eval.DEFAULT_ARM, "openai"),
            run_eval.condition_label(agents.SEARCH_MODE_NATIVE,
                                     run_eval.DEFAULT_ARM, "anthropic"))


class RetrievalQATemporalQualificationTest(unittest.TestCase):
    """Frozen dynamic labels must be searched in their historical frame."""

    def test_freshqa_snapshot_has_an_explicit_reference_date(self):
        date_value, basis = import_retrievalqa.retrievalqa_answer_as_of({
            "data_source": "freshqa", "question_id": "freshqa_378"})
        self.assertEqual(date_value, "2024-02-01")
        self.assertEqual(basis, "retrievalqa_freshqa_snapshot")

    def test_realtimeqa_date_comes_from_the_question_id(self):
        date_value, basis = import_retrievalqa.retrievalqa_answer_as_of({
            "data_source": "realtimeqa",
            "question_id": "realtimeqa_20231013_1",
        })
        self.assertEqual(date_value, "2023-10-13")
        self.assertEqual(basis, "realtimeqa_question_id")

    def test_static_sources_are_not_given_a_fabricated_date(self):
        self.assertEqual(
            import_retrievalqa.retrievalqa_answer_as_of({
                "data_source": "triviaqa", "question_id": "triviaqa_1"}),
            (None, None),
        )

    def test_question_tells_the_agent_to_search_in_the_historical_frame(self):
        qualified, answer_as_of, _ = run_eval.qualify_retrievalqa_question(
            "Who is the richest man on earth?",
            {"data_source": "freshqa", "question_id": "freshqa_379"},
            "RetrievalQA",
        )
        self.assertEqual(answer_as_of, "2024-02-01")
        self.assertIn("Reference date: 2024-02-01", qualified)
        self.assertIn("not as of today", qualified)
        self.assertIn("include the reference date", qualified)

    def test_other_datasets_keep_the_original_question(self):
        question = "Who won?"
        self.assertEqual(
            run_eval.qualify_retrievalqa_question(
                question, {"data_source": "freshqa"}, "LiveNewsBench"),
            (question, None, None),
        )

    def test_day_filter_resolves_to_the_reference_date(self):
        setup = run_eval.ydc_setup("native_fresh", "2024-02-01")
        self.assertEqual(setup["freshness"], "2024-02-01to2024-02-01")
        self.assertEqual(setup["freshness_reference"], "answer_as_of")

    def test_week_filter_ends_on_the_reference_date(self):
        setup = run_eval.ydc_setup("fresh_week", "2024-02-01")
        self.assertEqual(setup["freshness"], "2024-01-26to2024-02-01")

    def test_unqualified_rows_keep_the_declared_relative_filters(self):
        self.assertEqual(run_eval.ydc_setup("native_fresh")["freshness"], "day")
        self.assertEqual(run_eval.ydc_setup("fresh_week")["freshness"], "week")

    def test_experiment_metadata_declares_row_relative_resolution(self):
        setup = run_eval.experiment_ydc_setup("fresh_week", "RetrievalQA")
        self.assertEqual(setup["freshness"], "answer_as_of_week")
        self.assertEqual(setup["freshness_reference"], "row_answer_as_of")

# --- native adapters --------------------------------------------------------

class AnthropicNativeSearchTest(unittest.TestCase):
    def test_normalizes_results_onto_the_harness_trajectory_schema(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(len(run.trajectory), 1)
        results = run.trajectory[0]["results"]
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertEqual(results[0]["url"], "https://a.example/x")
        self.assertEqual(results[0]["published_date"], "2026-07-30T00:00:00Z")
        self.assertEqual(run.final_answer, "Team Alpha")

    def test_declares_a_no_snippet_surface_and_leaves_snippets_empty(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(run.surface, agents.SURFACE_NO_SNIPPET)
        for result in run.trajectory[0]["results"]:
            self.assertEqual(result["snippet"], "")

    def test_cited_text_is_not_promoted_into_a_snippet(self):
        # cited_text exists and contains the gold answer, but it is selected
        # BECAUSE it supports the answer. Mapping it to `snippet` would make
        # snippet_sufficiency score ~1.0 on this arm by construction.
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(run.citations[0]["cited_text"], "Team Alpha won")
        blob = " ".join(r["snippet"] for r in run.trajectory[0]["results"])
        self.assertNotIn("Team Alpha", blob)

    def test_sends_max_uses_and_blocked_domains(self):
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?",
            ["source.example", "web.archive.org"], 5)
        tool = client.messages.calls[0]["tools"][0]
        self.assertEqual(tool["type"], agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE)
        self.assertEqual(tool["max_uses"], 5)
        self.assertEqual(tool["allowed_callers"], ["direct"])
        self.assertEqual(tool["blocked_domains"],
                         ["source.example", "web.archive.org"])
        self.assertEqual(tool["user_location"], agents.SEARCH_USER_LOCATION)

    def test_pause_turn_receives_only_the_remaining_search_budget(self):
        first = dict(
            ANTHROPIC_RESPONSE,
            stop_reason="pause_turn",
            usage={"input_tokens": 100, "output_tokens": 20,
                   "server_tool_use": {"web_search_requests": 2}},
        )
        second = dict(
            ANTHROPIC_RESPONSE,
            usage={"input_tokens": 100, "output_tokens": 20,
                   "server_tool_use": {"web_search_requests": 3}},
        )
        client = FakeSequencedAnthropicClient([first, second])
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(client.messages.calls[0]["tools"][0]["max_uses"], 5)
        self.assertEqual(client.messages.calls[1]["tools"][0]["max_uses"], 3)
        self.assertEqual(run.n_searches, 5)

    def test_uses_the_basic_tool_not_dynamic_filtering(self):
        # Dynamic filtering (_20260209+) drops results before they reach the
        # context window, so the model would answer from a surface we cannot
        # observe and that does not match the harness arms.
        self.assertEqual(agents.ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
                         "web_search_20250305")

    def test_thinking_stays_enabled(self):
        # Keep the Anthropic arms on the same explicit adaptive-thinking mode;
        # inheriting different defaults would add another moving treatment.
        client = FakeAnthropicClient(ANTHROPIC_RESPONSE)
        agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(client.messages.calls[0]["thinking"]["type"], "adaptive")

    def test_prefers_the_vendor_reported_search_count(self):
        # A failed search still spends budget, but Anthropic documents it as
        # unbilled. Keep attempts and billable searches separate.
        client = FakeAnthropicClient(ANTHROPIC_ERROR_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(run.trajectory, [])
        self.assertEqual(run.n_searches, 1)
        self.assertEqual(run.billable_searches, 0)

    def test_records_search_errors_instead_of_reading_them_as_empty_results(self):
        client = FakeAnthropicClient(ANTHROPIC_ERROR_RESPONSE)
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
        self.assertEqual(run.search_errors,
                         [{"query": "who won", "error_code": "max_uses_exceeded"}])

    def test_refusal_is_flagged_rather_than_scored_as_a_wrong_answer(self):
        client = FakeAnthropicClient(dict(ANTHROPIC_RESPONSE,
                                          stop_reason="refusal"))
        run = agents.anthropic_native_search(
            client, "claude-sonnet-5", "sys", "who won?", [], 5)
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

    def test_enforces_the_five_search_budget(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", [], max_searches=5)
        self.assertEqual(client.responses.calls[0]["max_tool_calls"], 5)

    def test_sends_blocked_domains_through_the_filters_object(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", ["source.example"])
        tool = client.responses.calls[0]["tools"][0]
        self.assertEqual(tool["type"], "web_search")
        self.assertEqual(tool["filters"]["blocked_domains"], ["source.example"])
        self.assertEqual(tool["search_context_size"],
                         agents.OPENAI_SEARCH_CONTEXT_SIZE)
        self.assertEqual(tool["user_location"], agents.SEARCH_USER_LOCATION)

    def test_separates_plural_queries_page_opens_and_find_actions(self):
        search = dict(
            OPENAI_RESPONSE["output"][0],
            action={"type": "search", "queries": ["first query", "second query"],
                    "sources": [{"type": "url", "url": "https://a.example/x"}]},
        )
        response = dict(
            OPENAI_RESPONSE,
            output=[
                search,
                {"type": "web_search_call", "status": "completed",
                 "action": {"type": "open_page", "url": "https://a.example/x"}},
                {"type": "web_search_call", "status": "completed",
                 "action": {"type": "find_in_page", "url": "https://a.example/x",
                            "pattern": "winner"}},
                *OPENAI_RESPONSE["output"][1:],
            ],
        )
        run = agents.openai_native_search(
            FakeOpenAIClient(response), "gpt-5.6", "sys", "who won?", [])
        self.assertEqual(run.n_searches, 1)
        self.assertEqual(run.vendor_search_count, 3)
        self.assertEqual(run.emitted_queries, ["first query", "second query"])
        self.assertEqual([a["type"] for a in run.native_actions],
                         ["search", "open_page", "find_in_page"])
        self.assertEqual(run.trajectory[0]["queries"],
                         ["first query", "second query"])
        self.assertIsNone(run.trajectory[0]["results"][0]["rank"])
        self.assertEqual(run.trajectory[0]["results"][0]["provider_order"], 1)

    def test_backfills_titles_from_citations_only(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        run = agents.openai_native_search(
            client, "gpt-5.6", "sys", "who won?", [])
        results = run.trajectory[0]["results"]
        self.assertEqual(results[0]["title"], "A")
        # Uncited results stay untitled — the reason this arm is urls_only.
        self.assertEqual(results[1]["title"], "")

    def test_native_queries_receive_the_same_operator_observability(self):
        search_call = dict(
            OPENAI_RESPONSE["output"][0],
            action=dict(OPENAI_RESPONSE["output"][0]["action"],
                        query="site:reuters.com who won"),
        )
        response = dict(
            OPENAI_RESPONSE,
            output=[search_call, *OPENAI_RESPONSE["output"][1:]],
        )
        out = run_eval._run_native(
            FakeOpenAIClient(response), agents.VENDORS["openai"], "gpt-5.6",
            run_eval.NATIVE_SEARCH_SYSTEM_PROMPT, "who won?", [])
        self.assertEqual(out["operator_violations"], [{
            "query": "site:reuters.com who won",
            "operators": ["site:reuters.com"],
        }])


class NativeSearchPricingTest(unittest.TestCase):
    def test_both_native_arms_price_at_the_published_ten_dollars_per_thousand(self):
        anthropic_rate, anthropic_ok = agents.native_search_rate_usd(
            "anthropic", "claude-sonnet-5")
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

    def test_temporal_grounding_excludes_anthropic_last_updated_dates(self):
        result = scorers.temporal_grounding(
            {}, _row_output(scorers.SURFACE_NO_SNIPPET, [
                {"rank": 1, "url": "https://a.example/x",
                 "published_date": "2026-07-30",
                 "date_semantics": "last_updated"}], used_searches=1),
            "Team Alpha", metadata=LNB_METADATA)
        self.assertIsNone(result["score"])

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

    def test_domain_entropy_clamps_floating_point_upper_boundary(self):
        results = [
            {"rank": rank, "url": f"https://host-{rank}.example/story"}
            for rank in range(1, 6)
        ]
        result = scorers.domain_entropy(
            {}, _row_output(
                scorers.SURFACE_FULL, results, used_searches=1
            ), "x"
        )
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
    def test_sonnet_uses_current_standard_price(self):
        input_usd, confirmed = agents.model_cost_usd(
            "claude-sonnet-5", 1_000_000, 0
        )
        output_usd, _ = agents.model_cost_usd(
            "claude-sonnet-5", 0, 1_000_000
        )
        self.assertTrue(confirmed)
        self.assertEqual((input_usd, output_usd), (2.0, 10.0))

    """Search fees are the small half of the bill, so a search-fee-only
    comparison ranks arms on the wrong quantity."""

    def test_priced_models_produce_a_token_cost(self):
        usd, confirmed = agents.model_cost_usd("claude-fable-5", 1_000_000, 0)
        self.assertTrue(confirmed)
        self.assertAlmostEqual(usd, 10.00)
        usd, _ = agents.model_cost_usd("gpt-5.6-sol", 0, 1_000_000)
        self.assertAlmostEqual(usd, 30.00)

    def test_cached_input_bills_at_a_tenth_on_the_frontier_vendors(self):
        # Both vendors publish 0.1x: OpenAI $0.50 cached on $5.00 base,
        # Anthropic $1 cache-hit on $10 base.
        full, _ = agents.model_cost_usd("gpt-5.6-sol", 1_000_000, 0)
        cached, _ = agents.model_cost_usd("gpt-5.6-sol", 1_000_000, 0,
                                          1_000_000)
        self.assertAlmostEqual(full, 5.00)
        self.assertAlmostEqual(cached, 0.50)

    def test_a_partly_cached_prompt_splits_across_both_rates(self):
        # The realistic shape: OpenAI caches automatically, so a harness turn is
        # mostly a cache hit with a small uncached tail.
        usd, _ = agents.model_cost_usd("gpt-5.6-sol", 1_000_000, 0, 900_000)
        self.assertAlmostEqual(usd, 0.1 * 5.00 + 0.9 * 0.50)

    def test_cached_tokens_are_a_subset_never_an_addition(self):
        """The whole convention rests on cached ⊆ prompt_tokens. A caller that
        passes a cached count exceeding the total must not produce a negative
        uncached charge."""
        usd, _ = agents.model_cost_usd("gpt-5.6-sol", 1000, 0, 999_999)
        floor, _ = agents.model_cost_usd("gpt-5.6-sol", 1000, 0, 1000)
        self.assertAlmostEqual(usd, floor)
        self.assertGreater(usd, 0)

    def test_the_oss_rows_get_no_cache_discount(self):
        # Baseten publishes no cached-input rate, so cached tokens bill at full
        # price. Overstating the OSS arm is the safe direction; understating it
        # would flatter the cost-substitution claim this study is testing.
        full, _ = agents.model_cost_usd("zai-org/GLM-5.2", 1_000_000, 0)
        cached, _ = agents.model_cost_usd("zai-org/GLM-5.2", 1_000_000, 0,
                                          1_000_000)
        self.assertAlmostEqual(full, cached)

    def test_price_table_name_shapes(self):
        """cached_input_multiplier splits on the org-prefix in the model name,
        so that shape has to hold for every row in the table."""
        for model in agents.MODEL_USD_PER_MTOK:
            is_oss = "/" in model
            self.assertEqual(
                agents.cached_input_multiplier(model),
                agents.OSS_CACHED_INPUT_MULTIPLIER if is_oss
                else agents.FRONTIER_CACHED_INPUT_MULTIPLIER,
                f"{model} landed on the wrong side of the cache-rate split")

    def test_anthropic_cache_reads_are_added_into_the_input_total(self):
        """Anthropic reports cache reads OUTSIDE input_tokens. Treating its
        payload like OpenAI's would drop those tokens off the bill entirely."""
        usage = {"input_tokens": 100, "cache_read_input_tokens": 900,
                 "cache_creation_input_tokens": 0, "output_tokens": 10}
        billable, cached = agents._anthropic_token_split(usage)
        self.assertEqual(billable, 1000)
        self.assertEqual(cached, 900)

    def test_openai_cached_tokens_are_read_as_a_subset(self):
        """Verified against a live response: input_tokens=2814 carried
        cached_tokens=2811, so the cached count is already inside the total."""
        self.assertEqual(
            agents._openai_cached_tokens(
                {"input_tokens_details": {"cached_tokens": 2811}}), 2811)
        # Chat Completions names the same block prompt_tokens_details.
        self.assertEqual(
            agents._openai_cached_tokens(
                {"prompt_tokens_details": {"cached_tokens": 42}}), 42)
        self.assertEqual(agents._openai_cached_tokens({}), 0)

    def test_unpriced_model_returns_none_not_zero(self):
        # 0.0 would place an unpriced arm at the origin of a cost frontier,
        # reading as free. None keeps it out of the frontier entirely.
        usd, confirmed = agents.model_cost_usd("some/unlisted-model", 1000, 1000)
        self.assertIsNone(usd)
        self.assertFalse(confirmed)

    def test_the_oss_arm_is_priced_so_substitution_is_a_cost_claim(self):
        # "OSS + retrieval vs frontier without it" is a cost-ratio claim, and a
        # cost-ratio claim needs both sides priced.
        usd, confirmed = agents.model_cost_usd("zai-org/GLM-5.2", 1000, 1000)
        self.assertTrue(confirmed)
        self.assertGreater(usd, 0)

    def test_the_cheap_oss_row_is_an_order_of_magnitude_below_frontier(self):
        # Claim B's extreme point. Asserted against the CHEAPEST frontier row, so
        # the claim cannot be inflated by comparing to the most expensive one.
        cheapest_oss = min(agents.MODEL_USD_PER_MTOK[r.model][0]
                           for r in agents.oss_models())
        cheapest_frontier = min(agents.MODEL_USD_PER_MTOK[r.model][0]
                                for r in agents.frontier_models())
        self.assertGreaterEqual(cheapest_frontier / cheapest_oss, 10.0)

    def test_the_two_oss_rows_are_distinct_points_on_the_cost_frontier(self):
        # Two OSS rows only buy something if they differ materially in price. The
        # mid row is NOT an order of magnitude below frontier -- it is the
        # "strong but cheaper" point, and conflating the two rows would let a
        # result from one be reported as if it came from the other.
        prices = sorted(agents.MODEL_USD_PER_MTOK[r.model][0]
                        for r in agents.oss_models())
        self.assertEqual(len(prices), 2)
        self.assertGreaterEqual(prices[1] / prices[0], 3.0)

    def test_model_rows_are_all_priced(self):
        # An unpriced row silently drops out of every cost comparison.
        for row in agents.MATRIX_MODELS:
            with self.subTest(model=row.model):
                usd, confirmed = agents.model_cost_usd(row.model, 1000, 1000)
                self.assertTrue(confirmed)
                self.assertGreater(usd, 0)


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
            arm=run_eval.DEFAULT_ARM,
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
        self.assertIsNone(
            scorers.temporal_grounding({}, output, "Team Alpha",
                                       metadata=LNB_METADATA)["score"])

    def test_native_arm_is_not_free(self):
        # search_cost_usd used to be $0.00 on every native row, making the arm
        # look free next to the harness arms.
        output, hooks = self._run_task("anthropic", ANTHROPIC_RESPONSE)
        self.assertEqual(output["used_searches"], 1)
        self.assertEqual(hooks.metadata["search_mode"], "native")
        # 1 search x $10/1k.
        self.assertAlmostEqual(
            agents.native_search_rate_usd("anthropic", "claude-sonnet-5")[0], 0.010)

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

    def test_exclusion_request_is_recorded_on_the_native_arms(self):
        _, hooks = self._run_task("anthropic", ANTHROPIC_RESPONSE)
        self.assertTrue(hooks.metadata["exclusion_requested"])
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
            arm=run_eval.DEFAULT_ARM, agent_model="claude-fable-5",
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
        # A configured gateway intentionally replaces local vendor credentials,
        # so isolate the direct-vendor path even when .env enables the gateway.
        with mock.patch.dict(
                os.environ,
                {"BASETEN_API_KEY": "", agents.GATEWAY_URL_ENV: ""}):
            with self.assertRaises(SystemExit) as caught:
                agents.get_client("baseten", lambda c: c)
        self.assertIn("BASETEN_API_KEY", str(caught.exception))

    def test_unknown_vendor_is_rejected(self):
        with self.assertRaises(ValueError):
            agents.vendor_of("together")


class GatewayRoutingTest(unittest.TestCase):
    """Serving path is an experimental variable here, not a deployment detail.

    Two arms served over different stacks are not a one-variable contrast, so
    these cases pin the two properties that keep that from happening silently:
    the switch moves every vendor at once, and the run records where it went.
    """

    GATEWAY_ENV = (agents.GATEWAY_URL_ENV, agents.GATEWAY_KEY_ENV,
                   agents.GATEWAY_PROJECT_ENV, agents.GATEWAY_ORG_ENV,
                   "BRAINTRUST_API_KEY")

    def setUp(self):
        agents.reset_clients()
        self.addCleanup(agents.reset_clients)
        for name in self.GATEWAY_ENV:
            self._set_env(name, None)

    def _set_env(self, name, value):
        previous = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        self.addCleanup(
            lambda: os.environ.__setitem__(name, previous) if previous is not None
            else os.environ.pop(name, None))

    def _enable(self, url="https://gateway.braintrust.dev"):
        self._set_env(agents.GATEWAY_URL_ENV, url)
        self._set_env("BRAINTRUST_API_KEY", "bt-test-key")

    def test_unset_url_means_direct_to_vendor(self):
        self.assertIsNone(agents.gateway_config())
        self.assertEqual(agents.serving_path(), agents.SERVING_PATH_DIRECT)
        # The OSS vendor's own base URL must survive; only the gateway replaces it.
        self.assertEqual(agents.effective_base_url(agents.VENDORS["baseten"]),
                         "https://inference.baseten.co/v1")

    def test_every_vendor_moves_together(self):
        """No per-vendor opt-out: a mixed matrix is the failure being prevented."""
        self._enable()
        self.assertEqual(agents.serving_path(), agents.SERVING_PATH_GATEWAY)
        for vendor in agents.VENDORS:
            self.assertTrue(
                agents.effective_base_url(agents.VENDORS[vendor])
                .startswith("https://gateway.braintrust.dev"),
                f"{vendor} was left on a direct route")

    def test_the_two_sdks_get_the_v1_suffix_they_each_expect(self):
        """openai-python appends /chat/completions; anthropic-python appends
        /v1/messages. One shared base URL would 404 on one of them."""
        self._enable()
        gw = agents.gateway_config()
        self.assertEqual(gw.openai_base_url, "https://gateway.braintrust.dev/v1")
        self.assertEqual(gw.anthropic_base_url, "https://gateway.braintrust.dev")

    def test_a_url_written_with_v1_is_accepted(self):
        """The docs write the base both ways; neither should silently double it."""
        self._enable("https://gateway.braintrust.dev/v1/")
        gw = agents.gateway_config()
        self.assertEqual(gw.openai_base_url, "https://gateway.braintrust.dev/v1")
        self.assertEqual(gw.anthropic_base_url, "https://gateway.braintrust.dev")

    def test_routing_headers_are_omitted_when_unconfigured(self):
        """An empty x-bt-project-name is not the same as no header."""
        self._enable()
        self.assertEqual(agents.gateway_config().headers(), {})
        self._set_env(agents.GATEWAY_PROJECT_ENV, "automations-spend-control")
        self._set_env(agents.GATEWAY_ORG_ENV, "BT Staging")
        self.assertEqual(
            agents.gateway_config().headers(),
            {"x-bt-project-name": "automations-spend-control",
             "x-bt-org-name": "BT Staging"})

    def test_gateway_key_overrides_the_logging_key(self):
        self._enable()
        self._set_env(agents.GATEWAY_KEY_ENV, "bt-service-token")
        self.assertEqual(agents.gateway_config().api_key, "bt-service-token")

    def test_a_url_with_no_key_fails_loudly_rather_than_calling_unauthenticated(self):
        self._set_env(agents.GATEWAY_URL_ENV, "https://gateway.braintrust.dev")
        with self.assertRaises(SystemExit) as caught:
            agents.gateway_config()
        self.assertIn(agents.GATEWAY_KEY_ENV, str(caught.exception))

    def test_the_vendor_key_is_not_required_under_gateway_routing(self):
        """The vendor credential lives in the Braintrust org, so demanding a
        local one would reject a correctly configured run."""
        self._enable()
        self._set_env("BASETEN_API_KEY", None)
        client = agents.get_client("baseten", lambda c: c)
        self.assertEqual(str(client.base_url).rstrip("/"),
                         "https://gateway.braintrust.dev/v1")
        self.assertEqual(client.api_key, "bt-test-key")


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
            run_eval._preflight(run_eval.DEFAULT_ARM,
                                agents.SEARCH_MODE_NATIVE, "baseten",
                                "zai-org/GLM-5.2")
        self.assertIn("native", str(caught.exception))

    def test_missing_vendor_key_fails_before_spending_money(self):
        # Under gateway routing the credential lives in Braintrust, so this
        # missing-local-key assertion must explicitly exercise the direct path.
        with mock.patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "", agents.GATEWAY_URL_ENV: ""}):
            with self.assertRaises(SystemExit) as caught:
                run_eval._preflight(
                    run_eval.DEFAULT_ARM, agents.SEARCH_MODE_NONE,
                    "anthropic", "claude-sonnet-5")
        self.assertIn("ANTHROPIC_API_KEY", str(caught.exception))

    def test_no_search_arm_does_not_require_a_search_provider_key(self):
        self._set_env("ANTHROPIC_API_KEY", "test-anthropic-key")
        self._set_env("EXA_API_KEY", None)
        run_eval._preflight("no_search", agents.SEARCH_MODE_NONE,
                            "anthropic", "claude-sonnet-5")


class SubsetSelectionTest(unittest.TestCase):
    """Every contrast is paired by task_key, so two arms drawn from different
    subsets lose their pairing silently — the drop shows up as missing data
    rather than as an error. Determinism here is what makes that impossible."""

    ROWS = [{"id": f"row-{i:02d}",
             "metadata": {"livenewsbench_split": "test" if i % 2 else "val"}}
            for i in range(10)]

    def test_no_flags_returns_the_dataset_untouched(self):
        data, meta = run_eval.select_rows(self.ROWS, None, None)
        self.assertIs(data, self.ROWS)
        self.assertFalse(meta["subset_applied"])

    def test_full_count_guard_counts_and_preserves_dataset_object(self):
        data, meta = run_eval.select_rows(
            self.ROWS, None, None, count_full=True
        )
        self.assertIs(data, self.ROWS)
        self.assertFalse(meta["subset_applied"])
        self.assertEqual(meta["n_rows"], len(self.ROWS))
        self.assertEqual(meta["n_available"], len(self.ROWS))

    def test_limit_is_deterministic_across_input_orderings(self):
        # Iteration order of a dataset is not contractually stable, so the slice
        # must be taken after sorting by id -- otherwise two arms can request
        # "the first 5 rows" and get different rows.
        a, ma = run_eval.select_rows(list(self.ROWS), None, 5)
        b, mb = run_eval.select_rows(list(reversed(self.ROWS)), None, 5)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(ma["subset_id"], mb["subset_id"])

    def test_subset_id_changes_when_the_selection_changes(self):
        _, five = run_eval.select_rows(list(self.ROWS), None, 5)
        _, six = run_eval.select_rows(list(self.ROWS), None, 6)
        self.assertNotEqual(five["subset_id"], six["subset_id"])

    def test_split_filters_and_is_recorded(self):
        rows, meta = run_eval.select_rows(list(self.ROWS), "test", None)
        self.assertEqual(len(rows), 5)
        self.assertEqual(meta["split"], "test")
        self.assertEqual(meta["n_available"], 10)

    def test_unmatched_split_fails_loudly_rather_than_running_zero_rows(self):
        with self.assertRaises(SystemExit):
            run_eval.select_rows(list(self.ROWS), "human_verified_test", None)


# ---------------------------------------------------------------------------
# Harness protocol per vendor.
#
# These are regression tests for a failure that reached production: the OpenAI
# harness arm ran on /v1/chat/completions, which rejects function tools alongside
# reasoning effort on gpt-5-family models. Every row of a 40-row pilot errored
# with a 400. The prior code chose the session class with
# `if spec.name == "anthropic"`, so "OpenAI and Baseten are the same protocol"
# was an assumption nothing asserted.
# ---------------------------------------------------------------------------


class RESPONSES:
    """Fixture responses for the Responses-API harness session."""

    # A reasoning item is included because the session must echo it back
    # unmodified; a fixture without one would pass even if the code dropped it.
    TOOL_CALL = {
        "status": "completed",
        "usage": {"input_tokens": 120, "output_tokens": 40},
        "output": [
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "need to search"}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_abc",
             "name": "search_web", "arguments": '{"query": "who won"}'},
        ],
    }
    FINAL = {
        "status": "completed",
        "usage": {"input_tokens": 300, "output_tokens": 25},
        "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "Team Alpha", "annotations": []}]},
        ],
    }
    MALFORMED_CALL = {
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "output": [
            {"type": "function_call", "id": "fc_2", "call_id": "call_bad",
             "name": "search_web", "arguments": "not json"},
        ],
    }
    REFUSAL = {
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "output": [
            {"type": "message", "content": [
                {"type": "refusal", "refusal": "I can't help with that."}]},
        ],
    }
    TRUNCATED = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 10, "output_tokens": 16384},
        "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "Team Al", "annotations": []}]},
        ],
    }


class FakeResponsesSequence:
    """Returns a scripted response per call, so a multi-turn loop can be driven."""

    def __init__(self, responses):
        self.queue = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.queue:
            raise AssertionError("session made more requests than scripted")
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


class FakeSequencedOpenAIClient:
    def __init__(self, responses):
        self.responses = FakeResponsesSequence(responses)


class _ScriptedSession:
    """Replays a fixed list of Turns through the harness session interface."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.tool_results = []

    def add_user(self, question):
        del question

    def step(self, tools_enabled=True):
        del tools_enabled
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.tool_results.extend(results)


def _search_call(query, call_id="c1"):
    return {"id": call_id, "name": agents.SEARCH_TOOL_NAME,
            "arguments": {"query": query}, "malformed": False}


class HarnessSearchErrorTest(unittest.TestCase):
    """A failing search API must degrade the row, not delete it.

    Before this, a You.com error propagated out of the task and killed the whole
    row. That is the worst available option: rows do not fail at random — a
    provider fails hardest on the queries it handles worst — so the surviving
    rows are a favorable subset of the ones actually asked, and the run reports a
    score over a dataset it silently narrowed.
    """

    def _run(self, search_side_effect, turns):
        session = _ScriptedSession(turns)
        original_session = run_eval.make_harness_session
        original_search = run_eval.run_search
        run_eval.make_harness_session = lambda *a, **k: session
        run_eval.run_search = search_side_effect
        self.addCleanup(setattr, run_eval, "make_harness_session",
                        original_session)
        self.addCleanup(setattr, run_eval, "run_search", original_search)
        return run_eval._run_harness(
            object(), agents.VENDORS["openai"], "gpt-5.6-sol", "sys",
            "who won?", [], "normalized", agents.SEARCH_MODE_HARNESS)

    @staticmethod
    def _http_error(status):
        request = httpx.Request("GET", "https://ydc-index.io/v1/search")
        def raise_it(*args, **kwargs):
            del args, kwargs
            raise httpx.HTTPStatusError(
                "boom", request=request,
                response=httpx.Response(status, request=request))
        return raise_it

    def test_a_failed_search_is_recorded_and_the_row_survives(self):
        out = self._run(
            self._http_error(503),
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["final_answer"], "Team A won.")
        self.assertEqual(out["search_errors"],
                         [{"query": "who won", "error_code": "503"}])

    def test_a_failed_search_is_not_billed(self):
        # The request errored out, so it was not a served search — the same rule
        # the model vendors apply to their own server-side search.
        out = self._run(
            self._http_error(503),
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["search_cost"], 0.0)

    def test_a_failed_search_surfaces_no_trajectory_entry(self):
        # An error round is not an empty SERP. Appending a zero-result entry
        # would let leakage_guard and the snippet scorers read it as a clean,
        # compliant search instead of an unobservable one.
        out = self._run(
            self._http_error(503),
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["trajectory"], [])

    def test_a_failed_search_still_spends_budget(self):
        # Otherwise a flaky provider gets unlimited free retries inside the turn
        # cap, and the arm quietly exceeds the budget every other arm is held to.
        out = self._run(
            self._http_error(503),
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["used_searches"], 1)

    def test_the_model_is_told_the_search_failed(self):
        # A silent empty result would read as "nothing exists on the web for
        # this query", which is a different claim than "the lookup broke".
        session_turns = [agents.Turn(tool_calls=[_search_call("who won")]),
                         agents.Turn(text="Team A won.")]
        session = _ScriptedSession(session_turns)
        original_session = run_eval.make_harness_session
        original_search = run_eval.run_search
        run_eval.make_harness_session = lambda *a, **k: session
        run_eval.run_search = self._http_error(503)
        self.addCleanup(setattr, run_eval, "make_harness_session",
                        original_session)
        self.addCleanup(setattr, run_eval, "run_search", original_search)
        run_eval._run_harness(
            object(), agents.VENDORS["openai"], "gpt-5.6-sol", "sys",
            "who won?", [], "normalized", agents.SEARCH_MODE_HARNESS)
        self.assertEqual(len(session.tool_results), 1)
        self.assertIn("Search failed", session.tool_results[0][1])

    def test_a_transport_failure_with_no_status_records_its_class(self):
        def raise_timeout(*args, **kwargs):
            del args, kwargs
            raise httpx.ConnectTimeout("timed out")
        out = self._run(
            raise_timeout,
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["search_errors"],
                         [{"query": "who won", "error_code": "ConnectTimeout"}])

    def test_integrity_guards_are_not_swallowed(self):
        """_provider_json raises ValueError for a refused redirect or an
        unapproved host. Those mean the environment is wrong, not that one
        search failed, so they must stop the run rather than be logged as a
        routine error and averaged into a score."""
        def raise_guard(*args, **kwargs):
            del args, kwargs
            raise ValueError("provider API redirected to an unapproved host")
        with self.assertRaises(ValueError):
            self._run(
                raise_guard,
                [agents.Turn(tool_calls=[_search_call("who won")]),
                 agents.Turn(text="Team A won.")])

    def test_a_successful_search_is_still_billed_and_recorded(self):
        # The regression guard for the try/except: the success path must not
        # have moved.
        def succeed(*args, **kwargs):
            del args, kwargs
            return ([{"rank": 1, "url": "https://e.com", "title": "t",
                      "snippet": "s", "published_date": None}], "rendered", 7)
        out = self._run(
            succeed,
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A won.")])
        self.assertEqual(out["search_errors"], [])
        self.assertEqual(len(out["trajectory"]), 1)
        self.assertAlmostEqual(out["search_cost"], run_eval.YDC_USD_PER_CALL)


class SearchOperatorDetectionTest(unittest.TestCase):
    """The harness detects search operators so the per-model violation rate is
    auditable. The query is never altered — the raw query goes to You.com
    unchanged. Without detection, one model emitting `site:reuters.com ...`
    and another emitting a plain phrase is a query-construction difference
    reading as a retrieval difference, and nothing surfaces it."""

    def test_plain_query_detects_nothing(self):
        detected = run_eval._detect_search_operators("who won the 2026 final")
        self.assertEqual(detected, [])

    def test_site_operator_is_detected(self):
        detected = run_eval._detect_search_operators(
            "site:reuters.com Apple CEO")
        self.assertEqual(detected, ["site:reuters.com"])

    def test_intitle_operator_with_quotes_is_detected(self):
        detected = run_eval._detect_search_operators(
            'intitle:"Apple CEO" Tim Cook')
        self.assertEqual(detected, ['intitle:"Apple CEO"'])

    def test_multiple_operators_are_all_detected(self):
        detected = run_eval._detect_search_operators(
            "site:reuters.com intitle:Apple filetype:pdf CEO")
        self.assertEqual(len(detected), 3)

    def test_operator_is_case_insensitive(self):
        detected = run_eval._detect_search_operators("SITE:reuters.com Apple")
        self.assertEqual(detected, ["SITE:reuters.com"])

    def test_natural_language_colon_is_not_detected(self):
        # "note: this is important" is not an operator — "note" is not in the
        # recognized set, so it must not trigger a false positive.
        detected = run_eval._detect_search_operators(
            "note: this is a question about Apple")
        self.assertEqual(detected, [])

    def test_hyphenated_word_is_not_detected(self):
        # A hyphen inside a word is not a leading exclusion operator.
        detected = run_eval._detect_search_operators("state-of-the-art AI")
        self.assertEqual(detected, [])

    def test_prompt_forbidden_syntax_is_detected(self):
        detected = run_eval._detect_search_operators(
            'Apple OR Microsoft -rumor "chief executive"')
        self.assertEqual(detected, ["OR", "-rumor", '"chief executive"'])

    def test_non_string_input_is_safe(self):
        self.assertEqual(run_eval._detect_search_operators(None), [])


class HarnessOperatorViolationTest(unittest.TestCase):
    """When the agent emits operators, the harness records the violation and
    passes the raw query through to You.com unchanged."""

    def _run(self, search_side_effect, turns):
        session = _ScriptedSession(turns)
        self.session = session
        original_session = run_eval.make_harness_session
        original_search = run_eval.run_search
        run_eval.make_harness_session = lambda *a, **k: session
        run_eval.run_search = search_side_effect
        self.addCleanup(setattr, run_eval, "make_harness_session",
                        original_session)
        self.addCleanup(setattr, run_eval, "run_search", original_search)
        return run_eval._run_harness(
            object(), agents.VENDORS["openai"], "gpt-5.6-sol", "sys",
            "who won?", [], "normalized", agents.SEARCH_MODE_HARNESS)

    def test_raw_query_reaches_search_unchanged(self):
        captured = {}

        def capture(arm, query, excludes, setup):
            captured["query"] = query
            return ([], "No results.", 2)

        self._run(capture, [agents.Turn(tool_calls=[
            _search_call("site:reuters.com who won")]),
            agents.Turn(text="Team A.")])
        # The search layer received the raw query, not a sanitized one.
        self.assertEqual(captured["query"], "site:reuters.com who won")

    def test_violation_is_recorded_with_query_and_operators(self):
        out = self._run(
            lambda *a: ([], "No results.", 2),
            [agents.Turn(tool_calls=[
                _search_call("site:reuters.com who won")]),
             agents.Turn(text="Team A.")])
        self.assertEqual(len(out["operator_violations"]), 1)
        v = out["operator_violations"][0]
        self.assertEqual(v["query"], "site:reuters.com who won")
        self.assertEqual(v["operators"], ["site:reuters.com"])

    def test_plain_query_produces_no_violation(self):
        out = self._run(
            lambda *a: ([], "No results.", 2),
            [agents.Turn(tool_calls=[_search_call("who won")]),
             agents.Turn(text="Team A.")])
        self.assertEqual(out["operator_violations"], [])

    def test_trajectory_records_raw_query(self):
        out = self._run(
            lambda *a: ([{"rank": 1, "url": "https://e.com", "title": "t",
                          "snippet": "s", "published_date": None}], "r", 5),
            [agents.Turn(tool_calls=[
                _search_call("intitle:winner who won")]),
             agents.Turn(text="Team A.")])
        self.assertEqual(out["trajectory"][0]["query"], "intitle:winner who won")

    def test_non_string_query_is_returned_as_a_bad_tool_call(self):
        def search_must_not_run(*args):
            raise AssertionError(f"search unexpectedly called with {args!r}")

        out = self._run(
            search_must_not_run,
            [agents.Turn(tool_calls=[_search_call(None)]),
             agents.Turn(text="I could not find this.")])
        self.assertEqual(out["bad_tool_calls"], 1)
        self.assertEqual(out["used_searches"], 0)
        self.assertIn("query must be a string", self.session.tool_results[0][1])


class HarnessProtocolRegistryTest(unittest.TestCase):
    def test_openai_harness_does_not_run_on_chat_completions(self):
        # The exact production failure: chat completions returns 400 for function
        # tools + reasoning effort on gpt-5-family, and the model's DEFAULT effort
        # is not 'none', so omitting the parameter does not help.
        self.assertEqual(agents.VENDORS["openai"].harness_protocol,
                         agents.PROTOCOL_RESPONSES)

    def test_baseten_stays_on_chat_completions(self):
        # Baseten Model APIs expose no Responses endpoint, so both session types
        # must remain supported — this is not a migration.
        self.assertEqual(agents.VENDORS["baseten"].harness_protocol,
                         agents.PROTOCOL_CHAT_COMPLETIONS)

    def test_every_vendor_protocol_has_a_session_class(self):
        for name, spec in agents.VENDORS.items():
            with self.subTest(vendor=name):
                self.assertIn(spec.harness_protocol, agents.HARNESS_SESSIONS)

    def test_each_protocol_selects_its_own_session_class(self):
        expected = {
            "openai": agents.OpenAIResponsesHarnessSession,
            "baseten": agents.OpenAIHarnessSession,
            "anthropic": agents.AnthropicHarnessSession,
        }
        for vendor, session_class in expected.items():
            with self.subTest(vendor=vendor):
                session = agents.make_harness_session(
                    object(), agents.VENDORS[vendor], "m", "sys")
                self.assertIsInstance(session, session_class)

    def test_an_unknown_protocol_fails_loudly_rather_than_defaulting(self):
        spec = dataclasses.replace(agents.VENDORS["openai"],
                                   harness_protocol="grpc")
        with self.assertRaises(SystemExit) as caught:
            agents.make_harness_session(object(), spec, "m", "sys")
        self.assertIn("grpc", str(caught.exception))

    def test_openai_effort_is_pinned_now_that_responses_permits_it(self):
        # Left unpinned previously to dodge the chat-completions rejection. On
        # Responses it can be pinned, which is what makes this vendor's
        # native-vs-harness contrast one-variable.
        self.assertEqual(agents.VENDORS["openai"].reasoning_effort,
                         agents.OPENAI_EFFORT)

    def test_both_frontier_vendors_declare_the_same_effort(self):
        # Not required for validity — every primary contrast holds the model
        # fixed — but an undeclared difference in reasoning depth would be a
        # confound in any cross-vendor reading of the results.
        self.assertEqual(agents.VENDORS["openai"].reasoning_effort,
                         agents.VENDORS["anthropic"].reasoning_effort)


class OpenAIResponsesHarnessSessionTest(unittest.TestCase):
    def _session(self, responses):
        client = FakeSequencedOpenAIClient(responses)
        session = agents.make_harness_session(
            client, agents.VENDORS["openai"], "gpt-5.6-sol", "sys prompt")
        return client, session

    def test_function_tool_is_flat_not_nested_under_function(self):
        # Responses puts name/description/parameters at the top level; chat
        # completions nests them. Sending the nested shape here is a 400.
        client, session = self._session([RESPONSES.FINAL])
        session.add_user("who won?")
        session.step(tools_enabled=True)
        tool = client.responses.calls[0]["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], agents.SEARCH_TOOL_NAME)
        self.assertNotIn("function", tool)

    def test_tool_schema_is_shared_verbatim_with_the_other_protocols(self):
        # The agent must be the same across arms. A protocol-specific schema
        # would make "the tool" a different tool on this arm.
        client, session = self._session([RESPONSES.FINAL])
        session.add_user("q")
        session.step(tools_enabled=True)
        tool = client.responses.calls[0]["tools"][0]
        self.assertIs(tool["parameters"], agents.SEARCH_TOOL_PARAMETERS)
        self.assertFalse(tool["strict"])

    def test_reasoning_effort_and_output_cap_are_sent(self):
        client, session = self._session([RESPONSES.FINAL])
        session.add_user("q")
        session.step(tools_enabled=True)
        sent = client.responses.calls[0]
        self.assertEqual(sent["reasoning"], {"effort": agents.OPENAI_EFFORT})
        self.assertEqual(sent["max_output_tokens"],
                         agents.OPENAI_MAX_OUTPUT_TOKENS)

    def test_system_prompt_travels_in_instructions_not_the_turn_list(self):
        client, session = self._session([RESPONSES.FINAL])
        session.add_user("who won?")
        session.step(tools_enabled=True)
        sent = client.responses.calls[0]
        self.assertEqual(sent["instructions"], "sys prompt")
        self.assertEqual(sent["input"], [{"role": "user", "content": "who won?"}])

    def test_tool_call_is_parsed_and_keyed_on_call_id(self):
        # call_id, not id, is what function_call_output correlates on. Echoing
        # `id` back is accepted by the API and then never matched to the call.
        _, session = self._session([RESPONSES.TOOL_CALL])
        session.add_user("who won?")
        turn = session.step(tools_enabled=True)
        self.assertEqual(len(turn.tool_calls), 1)
        call = turn.tool_calls[0]
        self.assertEqual(call["id"], "call_abc")
        self.assertEqual(call["name"], agents.SEARCH_TOOL_NAME)
        self.assertEqual(call["arguments"], {"query": "who won"})
        self.assertFalse(call["malformed"])

    def test_malformed_arguments_are_flagged_not_raised(self):
        _, session = self._session([RESPONSES.MALFORMED_CALL])
        session.add_user("q")
        turn = session.step(tools_enabled=True)
        self.assertTrue(turn.tool_calls[0]["malformed"])
        self.assertEqual(turn.tool_calls[0]["arguments"], {})

    def test_reasoning_items_are_echoed_back_unmodified(self):
        # Dropping them is legal but discards reasoning context between tool
        # calls, weakening this arm relative to the Anthropic harness arm for a
        # reason that has nothing to do with search.
        client, session = self._session([RESPONSES.TOOL_CALL, RESPONSES.FINAL])
        session.add_user("who won?")
        session.step(tools_enabled=True)
        session.add_tool_results([("call_abc", "results here")])
        session.step(tools_enabled=True)
        sent = client.responses.calls[1]["input"]
        reasoning = [i for i in sent if i.get("type") == "reasoning"]
        self.assertEqual(len(reasoning), 1)
        self.assertIs(reasoning[0], RESPONSES.TOOL_CALL["output"][0])

    def test_tool_result_goes_back_as_function_call_output(self):
        client, session = self._session([RESPONSES.TOOL_CALL, RESPONSES.FINAL])
        session.add_user("who won?")
        session.step(tools_enabled=True)
        session.add_tool_results([("call_abc", "results here")])
        session.step(tools_enabled=True)
        sent = client.responses.calls[1]["input"]
        self.assertEqual(sent[-1], {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "results here",
        })

    def test_out_of_budget_forbids_calls_without_undeclaring_the_tool(self):
        # The input list already holds function_call items; a request carrying
        # those without the tool declared is rejected. Same constraint the
        # Anthropic session documents.
        client, session = self._session([RESPONSES.TOOL_CALL, RESPONSES.FINAL])
        session.add_user("who won?")
        session.step(tools_enabled=True)
        session.add_tool_results([("call_abc", "results")])
        session.step(tools_enabled=False)
        sent = client.responses.calls[1]
        self.assertEqual(sent["tool_choice"], "none")
        self.assertIn("tools", sent)

    def test_control_arm_never_declares_the_tool_at_all(self):
        # The no-search arm must differ from the harness arm ONLY in tool
        # availability, and it has no tool_use history to keep valid.
        client, session = self._session([RESPONSES.FINAL])
        session.add_user("who won?")
        turn = session.step(tools_enabled=False)
        sent = client.responses.calls[0]
        self.assertNotIn("tools", sent)
        self.assertNotIn("tool_choice", sent)
        self.assertEqual(turn.text, "Team Alpha")

    def test_usage_is_read_from_the_responses_field_names(self):
        # input_tokens/output_tokens here, prompt_tokens/completion_tokens on
        # chat completions. Reading the wrong pair silently reports zero cost.
        _, session = self._session([RESPONSES.FINAL])
        session.add_user("q")
        turn = session.step(tools_enabled=True)
        self.assertEqual(turn.prompt_tokens, 300)
        self.assertEqual(turn.completion_tokens, 25)

    def test_refusal_is_recorded_and_yields_no_answer_text(self):
        _, session = self._session([RESPONSES.REFUSAL])
        session.add_user("q")
        turn = session.step(tools_enabled=True)
        self.assertTrue(turn.refused)
        self.assertEqual(turn.text, "")

    def test_truncation_is_read_from_incomplete_details(self):
        _, session = self._session([RESPONSES.TRUNCATED])
        session.add_user("q")
        turn = session.step(tools_enabled=True)
        self.assertTrue(turn.truncated)


class OpenAINativeEffortTest(unittest.TestCase):
    def test_native_arm_sends_the_effort_it_is_given(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6-sol", "sys", "who won?", [], "high")
        self.assertEqual(client.responses.calls[0]["reasoning"],
                         {"effort": "high"})

    def test_native_arm_omits_reasoning_when_effort_is_unpinned(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6-sol", "sys", "who won?", [], None)
        self.assertNotIn("reasoning", client.responses.calls[0])

    def test_native_arm_caps_output_like_the_harness_arm(self):
        client = FakeOpenAIClient(OPENAI_RESPONSE)
        agents.openai_native_search(
            client, "gpt-5.6-sol", "sys", "who won?", [], "high")
        self.assertEqual(client.responses.calls[0]["max_output_tokens"],
                         agents.OPENAI_MAX_OUTPUT_TOKENS)

    def test_native_truncation_is_recorded(self):
        response = dict(OPENAI_RESPONSE,
                        incomplete_details={"reason": "max_output_tokens"})
        run = agents.openai_native_search(
            FakeOpenAIClient(response), "gpt-5.6-sol", "sys", "q", [], "high")
        self.assertTrue(run.truncated)

    def test_native_refusal_is_recorded(self):
        response = dict(OPENAI_RESPONSE, output=[
            {"type": "message", "content": [
                {"type": "refusal", "refusal": "no"}]}])
        run = agents.openai_native_search(
            FakeOpenAIClient(response), "gpt-5.6-sol", "sys", "q", [], "high")
        self.assertTrue(run.refused)
        self.assertEqual(run.final_answer, "")


if __name__ == "__main__":
    unittest.main()
