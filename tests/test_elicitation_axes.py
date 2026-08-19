"""Pin the two elicitation axes and the rails on model-supplied search params.

These axes exist because `native - harness` under a terse prompt and a
one-parameter tool measures tool familiarity as much as index quality: a frontier
model was post-trained to reach for its own search, and the same model handed an
unfamiliar third-party tool more often answers from memory.

Two properties matter more than any individual assertion here:

  1. SYMMETRY. `guided` is defined for the native arms too, with identical
     behavioral instructions. An elicitation pass applied only to the third-party
     arm would turn every native-vs-harness contrast into an unfair-effort
     comparison, in the direction that flatters third-party search.
  2. THE RAILS HOLD. The parametric tool schema hands retrieval parameters to the
     model, but not the ones that would disable a guard the study depends on.
     A model cannot relax the leakage exclusion, cannot reach page fetching, and
     cannot move a latency number the study reports as an outcome.
"""

import os
import unittest

os.environ.setdefault("YDC_API_KEY", "test-ydc-key")

import agents
import run_eval


class PromptVariantTest(unittest.TestCase):
    def test_guided_is_defined_for_both_search_layers(self):
        harness = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_HARNESS, run_eval.PROMPT_GUIDED)
        native = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_NATIVE, run_eval.PROMPT_GUIDED)
        self.assertNotEqual(harness, run_eval.FROZEN_SYSTEM_PROMPT)
        self.assertNotEqual(native, run_eval.NATIVE_SEARCH_SYSTEM_PROMPT)

    def test_both_guided_prompts_carry_the_same_behavioral_core(self):
        # This is the symmetry guarantee. If the core ever diverges, the guided
        # contrast stops being a retrieval comparison and becomes a prompt
        # comparison that happens to favor whichever arm got the better prompt.
        harness = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_HARNESS, run_eval.PROMPT_GUIDED)
        native = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_NATIVE, run_eval.PROMPT_GUIDED)
        self.assertIn(run_eval._GUIDED_CORE, harness)
        self.assertIn(run_eval._GUIDED_CORE, native)

    def test_every_prompt_keeps_the_frozen_answer_contract(self):
        # The scorers depend on this exact refusal string, in every variant.
        for mode in run_eval.SEARCH_MODES:
            for variant in run_eval.PROMPT_VARIANTS:
                with self.subTest(mode=mode, variant=variant):
                    self.assertIn(
                        "I could not find this",
                        run_eval._system_prompt_for(mode, variant))

    def test_every_search_prompt_states_the_five_search_budget(self):
        for mode in (run_eval.SEARCH_MODE_HARNESS, run_eval.SEARCH_MODE_NATIVE):
            for variant in run_eval.PROMPT_VARIANTS:
                with self.subTest(mode=mode, variant=variant):
                    self.assertIn(
                        "5", run_eval._system_prompt_for(mode, variant))

    def test_control_arm_has_no_guided_variant(self):
        # There is no tool to elicit use of, so both variants resolve to the same
        # control prompt rather than inventing a third control condition.
        self.assertEqual(
            run_eval._system_prompt_for(run_eval.SEARCH_MODE_NONE,
                                        run_eval.PROMPT_TERSE),
            run_eval._system_prompt_for(run_eval.SEARCH_MODE_NONE,
                                        run_eval.PROMPT_GUIDED))

    def test_every_mode_variant_pair_has_a_recorded_version(self):
        for mode in run_eval.SEARCH_MODES:
            for variant in run_eval.PROMPT_VARIANTS:
                with self.subTest(mode=mode, variant=variant):
                    self.assertIn((mode, variant), run_eval.PROMPT_VERSIONS)

    def test_parametric_suffix_only_appears_on_the_parametric_schema(self):
        minimal = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_HARNESS, run_eval.PROMPT_TERSE,
            agents.TOOL_SCHEMA_MINIMAL)
        parametric = run_eval._system_prompt_for(
            run_eval.SEARCH_MODE_HARNESS, run_eval.PROMPT_TERSE,
            agents.TOOL_SCHEMA_PARAMETRIC)
        self.assertNotIn("extraction_mode", minimal)
        self.assertIn("extraction_mode", parametric)


class ConditionLabelTest(unittest.TestCase):
    def test_default_axes_leave_preregistered_labels_untouched(self):
        # Old experiments must still join on condition_id.
        self.assertEqual(
            run_eval.condition_label(run_eval.SEARCH_MODE_HARNESS,
                                     "normalized", "openai"),
            "harness-youdotcom-normalized")
        self.assertEqual(
            run_eval.condition_label(run_eval.SEARCH_MODE_NATIVE,
                                     "normalized", "openai"),
            "native-openai")
        self.assertEqual(
            run_eval.condition_label(run_eval.SEARCH_MODE_NONE,
                                     "normalized", "openai"),
            "no_search")

    def test_non_default_axes_are_visible_in_the_label(self):
        label = run_eval.condition_label(
            run_eval.SEARCH_MODE_HARNESS, "wide_highlights", "openai",
            run_eval.PROMPT_GUIDED, agents.TOOL_SCHEMA_PARAMETRIC)
        self.assertEqual(
            label, "harness-youdotcom-wide_highlights-guided-parametric")


class ToolSchemaTest(unittest.TestCase):
    def test_minimal_schema_exposes_query_only(self):
        _, schema = agents.search_tool_contract(agents.TOOL_SCHEMA_MINIMAL)
        self.assertEqual(set(schema["properties"]), {"query"})

    def test_parametric_schema_withholds_the_guard_parameters(self):
        _, schema = agents.search_tool_contract(agents.TOOL_SCHEMA_PARAMETRIC)
        exposed = set(schema["properties"])
        # include_domains is mutually exclusive with exclude_domains, which
        # carries the leakage guard; exclude_domains is the guard itself;
        # crawl_timeout buys quality with a reported outcome.
        for withheld in ("include_domains", "exclude_domains", "crawl_timeout"):
            with self.subTest(parameter=withheld):
                self.assertNotIn(withheld, exposed)

    def test_parametric_extraction_cannot_reach_page_fetching(self):
        _, schema = agents.search_tool_contract(agents.TOOL_SCHEMA_PARAMETRIC)
        self.assertNotIn(
            "full_page", schema["properties"]["extraction_mode"]["enum"])

    def test_query_stays_the_only_required_field(self):
        # A model must be able to ignore every parameter and still search, or the
        # arm stops degrading cleanly into its fixed-config twin.
        _, schema = agents.search_tool_contract(agents.TOOL_SCHEMA_PARAMETRIC)
        self.assertEqual(schema["required"], ["query"])

    def test_all_three_protocols_declare_the_same_contract(self):
        # One tool definition per wire protocol, so the vendors cannot drift into
        # offering different parameter surfaces.
        spec_by_protocol = {
            s.harness_protocol: s for s in agents.VENDORS.values()}
        self.assertEqual(len(spec_by_protocol), 3, "expected all three protocols")
        seen = []
        for spec in spec_by_protocol.values():
            session = agents.make_harness_session(
                None, spec, spec.default_model, "sys",
                agents.TOOL_SCHEMA_PARAMETRIC)
            seen.append(session.tool_parameters)
        for parameters in seen[1:]:
            self.assertEqual(parameters, seen[0])

    def test_unknown_schema_fails_loudly(self):
        with self.assertRaises(SystemExit):
            agents.search_tool_contract("everything")


class AgentParameterMergeTest(unittest.TestCase):
    def _merge(self, arguments, arm="normalized", answer_as_of=None, excludes=()):
        setup = run_eval.ydc_setup(arm)
        setup["_exclude_domains"] = list(excludes)
        return run_eval.merge_agent_search_params(setup, arguments, answer_as_of)

    def test_no_parameters_reproduces_the_arm_default(self):
        resolved, rejections = self._merge({"query": "q"})
        base = run_eval.ydc_setup("normalized")
        self.assertEqual(resolved["count"], base["count"])
        self.assertEqual(resolved["extraction"], base["extraction"])
        self.assertEqual(resolved["freshness"], base["freshness"])
        self.assertEqual(rejections, [])
        self.assertEqual(resolved["_agent_set"], [])

    def test_model_values_override_the_arm_default(self):
        resolved, rejections = self._merge(
            {"query": "q", "count": 25, "freshness": "week",
             "extraction_mode": "highlights"})
        self.assertEqual(resolved["count"], 25)
        self.assertEqual(resolved["freshness"], "week")
        self.assertEqual(resolved["extraction"], "highlights")
        self.assertEqual(rejections, [])
        self.assertEqual(set(resolved["_agent_set"]),
                         {"count", "freshness", "extraction_mode"})

    def test_out_of_range_count_is_clamped_and_reported(self):
        resolved, rejections = self._merge({"query": "q", "count": 5000})
        self.assertEqual(resolved["count"], agents.YDC_MAX_AGENT_COUNT)
        self.assertTrue(any("count" in r for r in rejections))

    def test_undocumented_enum_value_is_refused_not_forwarded(self):
        # Forwarding would 422 and spend one of five searches for nothing.
        resolved, rejections = self._merge(
            {"query": "q", "freshness": "hour", "country": "XX",
             "safesearch": "sometimes"})
        self.assertIsNone(resolved["freshness"])
        self.assertNotIn("country", resolved)
        self.assertNotIn("safesearch", resolved)
        self.assertEqual(len(rejections), 3)

    def test_absolute_freshness_window_stays_harness_owned(self):
        # A model-chosen absolute window would either override the row's
        # reference-date anchor or leak the label's date into retrieval.
        resolved, rejections = self._merge(
            {"query": "q", "freshness": "2024-01-01to2024-02-01"})
        self.assertIsNone(resolved["freshness"])
        self.assertTrue(rejections)

    def test_model_relative_freshness_is_anchored_to_the_reference_date(self):
        # Otherwise the model searches today's web against a historical label.
        resolved, _ = self._merge(
            {"query": "q", "freshness": "week"}, answer_as_of="2024-02-01")
        self.assertEqual(resolved["freshness"], "2024-01-26to2024-02-01")
        self.assertEqual(resolved["freshness_reference"], "answer_as_of")

    def test_boost_cannot_resurrect_an_excluded_gold_source(self):
        resolved, rejections = self._merge(
            {"query": "q", "boost_domains": ["source.example", "reuters.com"]},
            excludes=["source.example"])
        self.assertEqual(resolved["boost_domains"], ["reuters.com"])
        self.assertTrue(any("source.example" in r for r in rejections))

    def test_unknown_parameter_is_reported_not_forwarded(self):
        resolved, rejections = self._merge(
            {"query": "q", "include_domains": ["gold.example"],
             "crawl_timeout": 60})
        self.assertNotIn("include_domains", resolved)
        self.assertNotIn("crawl_timeout", resolved)
        self.assertEqual(len(rejections), 2)

    def test_merged_setup_never_reaches_page_fetching(self):
        # Asserted against the arm default rather than a literal, so the rail is
        # pinned independently of which text layer the arms currently run.
        resolved, rejections = self._merge(
            {"query": "q", "extraction_mode": "full_page"})
        self.assertEqual(resolved["extraction"],
                         run_eval.ydc_setup("normalized")["extraction"])
        self.assertTrue(rejections)


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


class ParametricHarnessLoopTest(unittest.TestCase):
    """The loop integration: per-call resolution, surface declaration, rejections."""

    def _run(self, turns, arm="normalized",
             tool_schema=agents.TOOL_SCHEMA_PARAMETRIC, excludes=()):
        session = _ScriptedSession(turns)
        seen: list[dict] = []

        def fake_search(arm_, query, exclude_domains, setup=None):
            seen.append(dict(setup or {}))
            results = [{"rank": 1, "url": "https://p.example/a", "title": "A",
                        "snippet": "text", "published_date": None}]
            return results, "rendered", 10

        original_session = run_eval.make_harness_session
        original_search = run_eval.run_search
        run_eval.make_harness_session = lambda *a, **k: session
        run_eval.run_search = fake_search
        self.addCleanup(setattr, run_eval, "make_harness_session",
                        original_session)
        self.addCleanup(setattr, run_eval, "run_search", original_search)
        out = run_eval._run_harness(
            object(), agents.VENDORS["openai"], "gpt-5.6-terra", "sys",
            "who won?", list(excludes), arm, agents.SEARCH_MODE_HARNESS,
            run_eval.ydc_setup(arm), tool_schema)
        return out, seen

    @staticmethod
    def _call(arguments, call_id="c1"):
        return {"id": call_id, "name": agents.SEARCH_TOOL_NAME,
                "arguments": arguments, "malformed": False}

    def test_model_parameters_reach_the_provider_call(self):
        _, seen = self._run([
            agents.Turn(tool_calls=[self._call(
                {"query": "who won", "count": 20, "extraction_mode": "highlights"})]),
            agents.Turn(text="Team A."),
        ])
        self.assertEqual(seen[0]["count"], 20)
        self.assertEqual(seen[0]["extraction"], "highlights")

    def test_choosing_highlights_declares_the_highlights_surface(self):
        # If the surface stayed `full`, a highlights row would be pooled into the
        # snippet arms' mean without anything indicating the text layer changed.
        out, _ = self._run([
            agents.Turn(tool_calls=[self._call(
                {"query": "q", "extraction_mode": "highlights"})]),
            agents.Turn(text="Team A."),
        ])
        self.assertEqual(out["decision_surface"], agents.SURFACE_HIGHLIGHTS)

    def test_setting_no_parameters_keeps_the_arm_surface(self):
        out, seen = self._run([
            agents.Turn(tool_calls=[self._call({"query": "q"})]),
            agents.Turn(text="Team A."),
        ])
        self.assertEqual(
            out["decision_surface"],
            run_eval.EXTRACTION_SURFACES[
                run_eval.ydc_setup("normalized")["extraction"]])
        self.assertEqual(out["agent_param_calls"], 0)
        self.assertEqual(seen[0]["count"], 8)

    def test_mixed_extraction_within_a_row_is_flagged(self):
        out, _ = self._run([
            agents.Turn(tool_calls=[self._call({"query": "a"}, "c1")]),
            agents.Turn(tool_calls=[self._call(
                {"query": "b", "extraction_mode": "snippets"}, "c2")]),
            agents.Turn(text="Team A."),
        ])
        self.assertTrue(out["extraction_mixed"])
        self.assertEqual(out["decision_surface"], agents.SURFACE_HIGHLIGHTS)

    def test_parameter_use_and_rejections_are_counted_on_the_row(self):
        out, _ = self._run([
            agents.Turn(tool_calls=[self._call(
                {"query": "q", "count": 9999, "freshness": "fortnight"})]),
            agents.Turn(text="Team A."),
        ])
        self.assertEqual(out["agent_param_calls"], 1)
        self.assertEqual(len(out["param_rejections"]), 2)

    def test_minimal_schema_ignores_parameters_the_model_smuggles_in(self):
        # A model can emit extra keys even when the schema does not declare them.
        # On the minimal arm they must not silently change the retrieval config,
        # or the fixed-config arm stops being fixed.
        _, seen = self._run(
            [agents.Turn(tool_calls=[self._call(
                {"query": "q", "count": 50, "extraction_mode": "highlights"})]),
             agents.Turn(text="Team A.")],
            tool_schema=agents.TOOL_SCHEMA_MINIMAL)
        base = run_eval.ydc_setup("normalized")
        self.assertEqual(seen[0]["count"], base["count"])
        self.assertEqual(seen[0]["extraction"], base["extraction"])

    def test_boost_cannot_resurrect_an_excluded_gold_source_in_the_loop(self):
        _, seen = self._run(
            [agents.Turn(tool_calls=[self._call(
                {"query": "q", "boost_domains": ["gold.example"]})]),
             agents.Turn(text="Team A.")],
            excludes=["gold.example"])
        self.assertNotIn("boost_domains", seen[0])


class SurfaceDeclarationTest(unittest.TestCase):
    def test_highlights_is_a_known_tier_in_both_modules(self):
        import scorers

        self.assertEqual(agents.SURFACE_HIGHLIGHTS, scorers.SURFACE_HIGHLIGHTS)
        self.assertIn(scorers.SURFACE_HIGHLIGHTS, scorers._KNOWN_SURFACES)

    def test_highlights_keeps_the_mediator_metrics_computable(self):
        # Gating these off would drop the highlights arms out of the very metrics
        # they exist to move.
        import scorers

        for tier_set in (scorers._URL_SURFACES, scorers._DATE_SURFACES,
                         scorers._SNIPPET_SURFACES):
            self.assertIn(scorers.SURFACE_HIGHLIGHTS, tier_set)

    def test_native_tiers_still_gate_snippet_metrics_off(self):
        import scorers

        self.assertNotIn(scorers.SURFACE_NO_SNIPPET, scorers._SNIPPET_SURFACES)
        self.assertNotIn(scorers.SURFACE_URLS_ONLY, scorers._SNIPPET_SURFACES)
