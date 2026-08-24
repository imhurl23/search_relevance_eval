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
        "condition_concurrency": 1,
        "ydc_requests_per_second": 1.0,
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

    def test_condition_command_forwards_ydc_request_rate(self):
        command = run_matrix.condition_command(
            run_matrix.MATRIX[0],
            _args(ydc_requests_per_second=4.0),
            index=1,
        )
        self.assertEqual(
            command[command.index("--ydc-requests-per-second") + 1], "4.0"
        )


class ParallelMatrixSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.commit = patch.object(run_matrix, "_git_commit", return_value="abc123")
        self.commit.start()
        self.addCleanup(self.commit.stop)

    def test_compatibility_rejects_shared_vendor_and_two_harnesses(self):
        baseten_harness = run_matrix.MatrixCondition(
            "baseten", "open-a", "harness", "normalized"
        )
        openai_harness = run_matrix.MatrixCondition(
            "openai", "frontier-a", "harness", "wide"
        )
        baseten_none = run_matrix.MatrixCondition(
            "baseten", "open-b", "none"
        )
        anthropic_native = run_matrix.MatrixCondition(
            "anthropic", "frontier-b", "native"
        )

        self.assertFalse(run_matrix._condition_can_start(
            baseten_none, (baseten_harness,)
        ))
        self.assertFalse(run_matrix._condition_can_start(
            openai_harness, (baseten_harness,)
        ))
        self.assertTrue(run_matrix._condition_can_start(
            anthropic_native, (baseten_harness,)
        ))

    def test_parallel_scheduler_never_exceeds_safe_pair(self):
        conditions = (
            run_matrix.MatrixCondition(
                "baseten", "open-a", "harness", "normalized"
            ),
            run_matrix.MatrixCondition(
                "openai", "frontier-a", "harness", "wide"
            ),
            run_matrix.MatrixCondition(
                "anthropic", "frontier-b", "native"
            ),
            run_matrix.MatrixCondition("baseten", "open-b", "none"),
        )
        live = []
        max_live = 0

        class FakeProcess:
            next_pid = 90000

            def __init__(self, command, **kwargs):
                nonlocal max_live
                self.command = command
                self.vendor = command[command.index("--model-vendor") + 1]
                self.mode = command[command.index("--search-mode") + 1]
                self.polls = 0
                self.finished = False
                self.pid = FakeProcess.next_pid
                FakeProcess.next_pid += 1
                active = [item for item in live if not item.finished]
                self.assert_safe(active)
                live.append(self)
                max_live = max(max_live, len(active) + 1)

            def assert_safe(self, active):
                self_outer.assertFalse(any(
                    item.vendor == self.vendor for item in active
                ))
                if self.mode == "harness":
                    self_outer.assertFalse(any(
                        item.mode == "harness" for item in active
                    ))

            def poll(self):
                if self.finished:
                    return 0
                self.polls += 1
                if self.polls < 2:
                    return None
                self.finished = True
                return 0

        self_outer = self
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parallel.json"
            result = run_matrix._run_conditions_parallel(
                _args(condition_concurrency=2),
                conditions,
                checkpoint,
                popen_factory=FakeProcess,
            )
            self.assertEqual(result, 0)
            self.assertEqual(max_live, 2)
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertTrue(all(
                state["status"] == "completed"
                for state in saved["conditions"].values()
            ))

    def test_parallel_scheduler_retries_a_failed_condition(self):
        condition = run_matrix.MatrixCondition(
            "baseten", "open-a", "harness", "normalized"
        )
        launches = 0

        class FakeProcess:
            def __init__(self, command, **kwargs):
                nonlocal launches
                launches += 1
                self.pid = 91000 + launches
                self.returncode = 7 if launches == 1 else 0

            def poll(self):
                return self.returncode

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parallel-retry.json"
            result = run_matrix._run_conditions_parallel(
                _args(condition_concurrency=2, condition_retries=1),
                (condition,),
                checkpoint,
                popen_factory=FakeProcess,
            )
            self.assertEqual(result, 0)
            self.assertEqual(launches, 2)
            state = json.loads(checkpoint.read_text())["conditions"][condition.label]
            self.assertEqual(state["status"], "completed")
            self.assertEqual(
                [attempt["returncode"] for attempt in state["attempts"]],
                [7, 0],
            )

    def test_parallel_interrupt_stops_and_checkpoints_every_child(self):
        conditions = (
            run_matrix.MatrixCondition(
                "baseten", "open-a", "harness", "normalized"
            ),
            run_matrix.MatrixCondition(
                "anthropic", "frontier-b", "native"
            ),
        )
        processes = []

        class InterruptProcess:
            next_pid = 92000
            interrupt_raised = False

            def __init__(self, command, **kwargs):
                self.pid = InterruptProcess.next_pid
                InterruptProcess.next_pid += 1
                self.returncode = None
                processes.append(self)

            def poll(self):
                if not InterruptProcess.interrupt_raised:
                    InterruptProcess.interrupt_raised = True
                    raise KeyboardInterrupt
                return self.returncode

            def send_signal(self, sig):
                self.returncode = -sig

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parallel-interrupt.json"
            with self.assertRaises(KeyboardInterrupt):
                run_matrix._run_conditions_parallel(
                    _args(condition_concurrency=2),
                    conditions,
                    checkpoint,
                    popen_factory=InterruptProcess,
                )
            self.assertEqual(len(processes), 2)
            self.assertTrue(all(process.returncode is not None for process in processes))
            saved = json.loads(checkpoint.read_text())
            self.assertTrue(all(
                state["status"] == "interrupted"
                and state["attempts"][-1]["returncode"] == 130
                for state in saved["conditions"].values()
            ))


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
