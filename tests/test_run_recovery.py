"""Operational safety tests for full-matrix crash and resume behavior."""

import json
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_eval
import run_matrix


def _args(**overrides):
    values = {
        "dataset_name": "LiveNewsBench",
        "dataset_version": "version-1",
        "study_id": "recovery-test",
        "trials": 1,
        "limit": 2,
        "expected_rows": None,
        "env_file": Path(".env"),
        "order_seed": "fixed-order",
        "checkpoint": None,
        "resume": False,
        "retry_running": False,
        "condition_retries": 0,
        "condition_timeout_s": 600,
        "max_concurrency": 8,
        "max_row_error_rate": 0.0,
        "max_row_executions": 100,
        "max_search_cost_usd": 100.0,
        "allow_dirty": False,
        "execute": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class MatrixCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.conditions = run_matrix.MATRIX[:2]
        self.commit = patch.object(run_matrix, "_git_commit", return_value="abc123")
        self.commit.start()
        self.addCleanup(self.commit.stop)

    def test_completed_conditions_are_skipped_on_resume(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            calls = []

            def succeed(command, **kwargs):
                calls.append(command)
                return types.SimpleNamespace(returncode=0)

            args = _args()
            self.assertEqual(
                run_matrix._run_conditions(
                    args, self.conditions, checkpoint, run_command=succeed),
                0,
            )
            self.assertEqual(len(calls), 2)

            calls.clear()
            args.resume = True
            self.assertEqual(
                run_matrix._run_conditions(
                    args, self.conditions, checkpoint, run_command=succeed),
                0,
            )
            self.assertEqual(calls, [])
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(saved["status"], "completed")

    def test_failed_condition_resumes_at_next_attempt(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            first = _args(condition_retries=0)
            self.assertEqual(
                run_matrix._run_conditions(
                    first,
                    self.conditions,
                    checkpoint,
                    run_command=lambda *a, **k: types.SimpleNamespace(returncode=7),
                ),
                1,
            )

            commands = []

            def succeed(command, **kwargs):
                commands.append(command)
                return types.SimpleNamespace(returncode=0)

            resumed = _args(resume=True, condition_retries=1)
            self.assertEqual(
                run_matrix._run_conditions(
                    resumed, self.conditions, checkpoint, run_command=succeed),
                0,
            )
            self.assertEqual(
                commands[0][commands[0].index("--condition-attempt") + 1], "2"
            )
            self.assertIn("--condition-attempt", commands[1])

    def test_timeout_is_checkpointed_as_a_failed_attempt(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"

            def timeout(command, **kwargs):
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])

            result = run_matrix._run_conditions(
                _args(), self.conditions, checkpoint, run_command=timeout
            )
            self.assertEqual(result, 1)
            saved = json.loads(checkpoint.read_text())
            state = saved["conditions"][self.conditions[0].label]
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["attempts"][0]["returncode"], 124)

    def test_resume_rejects_a_different_plan(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            run_matrix._run_conditions(
                _args(),
                self.conditions,
                checkpoint,
                run_command=lambda *a, **k: types.SimpleNamespace(returncode=0),
            )
            with self.assertRaises(SystemExit):
                run_matrix._run_conditions(
                    _args(resume=True, dataset_version="version-2"),
                    self.conditions,
                    checkpoint,
                    run_command=lambda *a, **k: types.SimpleNamespace(returncode=0),
                )

    def test_checkpoint_lock_rejects_a_concurrent_launcher(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with run_matrix._checkpoint_lock(checkpoint):
                with self.assertRaises(SystemExit):
                    with run_matrix._checkpoint_lock(checkpoint):
                        self.fail("second launcher acquired the same checkpoint")

    def test_completion_marker_closes_parent_crash_window(self):
        class ParentCrash(BaseException):
            pass

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            one_condition = self.conditions[:1]

            def child_finishes_then_parent_is_interrupted(command, **kwargs):
                marker = Path(command[command.index("--completion-marker") + 1])
                run_eval._write_completion_marker(marker, "completed-experiment")
                raise ParentCrash

            with self.assertRaises(ParentCrash):
                run_matrix._run_conditions(
                    _args(), one_condition, checkpoint,
                    run_command=child_finishes_then_parent_is_interrupted,
                )

            calls = []
            resumed = _args(resume=True)
            self.assertEqual(
                run_matrix._run_conditions(
                    resumed, one_condition, checkpoint,
                    run_command=lambda command, **kwargs: calls.append(command),
                ),
                0,
            )
            self.assertEqual(calls, [])
            saved = json.loads(checkpoint.read_text())
            state = saved["conditions"][one_condition[0].label]
            self.assertEqual(state["status"], "completed")
            self.assertTrue(state["reconciled_from_marker"])

    def test_running_attempt_requires_explicit_retry_confirmation(self):
        class HardCrash(BaseException):
            pass

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            one_condition = self.conditions[:1]
            with self.assertRaises(HardCrash):
                run_matrix._run_conditions(
                    _args(), one_condition, checkpoint,
                    run_command=lambda *a, **k: (_ for _ in ()).throw(HardCrash()),
                )

            with self.assertRaisesRegex(SystemExit, "may still be active"):
                run_matrix._run_conditions(
                    _args(resume=True), one_condition, checkpoint,
                    run_command=lambda *a, **k: types.SimpleNamespace(returncode=0),
                )

            self.assertEqual(
                run_matrix._run_conditions(
                    _args(resume=True, retry_running=True, condition_retries=1),
                    one_condition,
                    checkpoint,
                    run_command=lambda *a, **k: types.SimpleNamespace(returncode=0),
                ),
                0,
            )

    def test_search_cost_ceiling_includes_all_attempts(self):
        once = run_matrix._max_search_cost_usd(
            run_matrix.MATRIX, rows=1329, trials=1, max_attempts=1
        )
        twice = run_matrix._max_search_cost_usd(
            run_matrix.MATRIX, rows=1329, trials=1, max_attempts=2
        )
        self.assertAlmostEqual(once, 398.70)
        self.assertAlmostEqual(twice, 797.40)


class EvalErrorGateTest(unittest.TestCase):
    def test_counts_task_and_scorer_failures(self):
        results = [
            types.SimpleNamespace(error=None, metadata={}),
            types.SimpleNamespace(error=RuntimeError("task"), metadata={}),
            types.SimpleNamespace(error=None, metadata={"scorer_errors": {"x": "bad"}}),
        ]
        failed, total, rate = run_eval._eval_error_summary(
            types.SimpleNamespace(results=results)
        )
        self.assertEqual((failed, total), (2, 3))
        self.assertAlmostEqual(rate, 2 / 3)

    def test_empty_condition_fails_closed(self):
        failed, total, rate = run_eval._eval_error_summary(
            types.SimpleNamespace(results=[])
        )
        self.assertEqual((failed, total, rate), (0, 0, 1.0))

    def test_result_gate_rejects_partial_condition(self):
        results = [types.SimpleNamespace(error=None, metadata={})]
        with self.assertRaisesRegex(SystemExit, "expected 2"):
            run_eval._enforce_eval_result_gate(
                types.SimpleNamespace(results=results),
                expected_results=2,
                max_row_error_rate=0.0,
            )

    def test_completion_marker_is_written_after_success(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "condition.done"
            run_eval._write_completion_marker(marker, "experiment-name")
            self.assertIn("experiment=experiment-name", marker.read_text())


if __name__ == "__main__":
    unittest.main()
