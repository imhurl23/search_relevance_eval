"""Launch the finalized 14-condition retrieval matrix.

The default action prints every command. Pass ``--execute`` only after checking
the dataset version, study ID, and optional row limit.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agents import MATRIX_MODELS, native_search_rate_usd
from import_livenewsbench import DATASET_NAME
from run_eval import (
    DEFAULT_YDC_REQUESTS_PER_SECOND,
    MAX_SEARCHES,
    MAX_YDC_REQUESTS_PER_SECOND,
    YDC_USD_PER_CALL,
)


CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MAX_ROW_EXECUTIONS = 40_000
YDC_MAX_SEARCH_COST_PER_ROW = MAX_SEARCHES * YDC_USD_PER_CALL
DEFAULT_CONDITION_TIMEOUT_S = 4 * 60 * 60
DEFAULT_CONDITION_CONCURRENCY = 2
MAX_SAFE_CONDITION_CONCURRENCY = 2
CONDITION_SCHEDULE_POLICY = "one-harness-distinct-vendors-v1"


@dataclass(frozen=True)
class MatrixCondition:
    vendor: str
    model: str
    search_mode: str
    arm: str | None = None

    @property
    def label(self) -> str:
        treatment = self.arm if self.search_mode == "harness" else self.search_mode
        return f"{self.model}::{treatment}"


OPEN_MODELS = tuple(
    (row.vendor, row.model) for row in MATRIX_MODELS
    if row.spec.model_class == "oss"
)
FRONTIER_MODELS = tuple(
    (row.vendor, row.model) for row in MATRIX_MODELS
    if row.spec.model_class == "frontier"
)
ALL_MODELS = (*OPEN_MODELS, *FRONTIER_MODELS)

# Interleave vendors by treatment so live-web drift is not confounded with one
# model running all of its conditions first.
MATRIX: tuple[MatrixCondition, ...] = (
    *(MatrixCondition(vendor, model, "none") for vendor, model in ALL_MODELS),
    *(MatrixCondition(vendor, model, "harness", "normalized")
      for vendor, model in ALL_MODELS),
    *(MatrixCondition(vendor, model, "harness", "wide")
      for vendor, model in ALL_MODELS),
    *(MatrixCondition(vendor, model, "native")
      for vendor, model in FRONTIER_MODELS),
)


def ordered_matrix(seed: str) -> tuple[MatrixCondition, ...]:
    """Return a reproducible randomized order for one live-web study.

    A fixed treatment-block order would always run native search after You.com.
    Hashing the study seed makes the order reproducible without making treatment
    synonymous with elapsed time across studies.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    conditions = list(MATRIX)
    rng.shuffle(conditions)
    return tuple(conditions)


def condition_command(
    condition: MatrixCondition,
    args,
    index: int,
    attempt: int = 1,
    completion_marker: Path | None = None,
    resume_experiment_attempt: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_eval.py")),
        "run",
        "--dataset-name", args.dataset_name,
        "--dataset-version", args.dataset_version,
        "--study-id", args.study_id,
        "--trials", str(args.trials),
        "--model-vendor", condition.vendor,
        "--agent-model", condition.model,
        "--search-mode", condition.search_mode,
        "--env-file", str(args.env_file),
        "--matrix-order-seed", args.order_seed or args.study_id,
        "--matrix-order-index", str(index),
        "--condition-attempt", str(attempt),
        "--resume-experiment-attempt", str(
            resume_experiment_attempt or attempt
        ),
        "--eval-timeout-s", str(args.condition_timeout_s - 60),
        "--max-row-error-rate", str(args.max_row_error_rate),
        "--max-concurrency", str(args.max_concurrency),
        "--matrix-condition-concurrency", str(args.condition_concurrency),
        "--matrix-schedule-policy", CONDITION_SCHEDULE_POLICY,
        "--ydc-requests-per-second", str(args.ydc_requests_per_second),
    ]
    if condition.arm:
        command.extend(["--arm", condition.arm])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    guarded_rows = args.limit if args.limit is not None else args.expected_rows
    if guarded_rows is not None:
        command.extend(["--expected-rows", str(guarded_rows)])
    if completion_marker is not None:
        command.extend(["--completion-marker", str(completion_marker)])
    return command


def _row_resume_experiment_attempt(state: dict, next_attempt: int) -> int:
    """Pin all future retries to one Braintrust experiment.

    New conditions pin their first attempt. Checkpoints created before row-level
    resume pin the most recent partial attempt, preserving the work that was in
    flight when the launcher was stopped.
    """
    pinned = state.get("row_resume_experiment_attempt")
    if pinned is None:
        pinned = (
            int(state["attempts"][-1]["attempt"])
            if state.get("attempts") else next_attempt
        )
        state["row_resume_experiment_attempt"] = pinned
    return int(pinned)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_path(args) -> Path:
    if args.checkpoint:
        return args.checkpoint
    safe_study = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.study_id).strip("-")
    return Path(".eval-checkpoints") / f"{safe_study}.json"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_worktree_is_clean() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet"], check=False).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], check=False
    ).returncode
    return unstaged == 0 and staged == 0


def _plan_fingerprint(args, conditions: tuple[MatrixCondition, ...]) -> dict:
    if not args.env_file.is_file():
        raise SystemExit(f"environment file not found: {args.env_file}")
    return {
        "study_id": args.study_id,
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "trials": args.trials,
        "limit": args.limit,
        "expected_rows": args.expected_rows,
        "order_seed": args.order_seed or args.study_id,
        "condition_order": [condition.label for condition in conditions],
        "git_commit": _git_commit(),
        # Detect routing/provider changes without storing any secret value.
        "env_file": str(args.env_file.resolve()),
        "env_file_sha256": hashlib.sha256(args.env_file.read_bytes()).hexdigest(),
        "max_concurrency": args.max_concurrency,
        "condition_concurrency": args.condition_concurrency,
        "condition_schedule_policy": CONDITION_SCHEDULE_POLICY,
        "ydc_requests_per_second": args.ydc_requests_per_second,
        "max_row_error_rate": args.max_row_error_rate,
        "condition_timeout_s": args.condition_timeout_s,
    }


def _write_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = _utc_now()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _checkpoint_lock(path: Path):
    """Prevent two launchers from executing the same checkpoint concurrently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"another matrix process holds the checkpoint lock: {lock_path}"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_or_create_checkpoint(
    path: Path,
    args,
    conditions: tuple[MatrixCondition, ...],
) -> dict:
    fingerprint = _plan_fingerprint(args, conditions)
    if path.exists():
        if not args.resume:
            raise SystemExit(
                f"checkpoint already exists: {path}. Pass --resume to continue "
                "that exact plan, or use a new --study-id."
            )
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read checkpoint {path}: {exc}") from None
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise SystemExit(f"unsupported checkpoint schema in {path}")
        if checkpoint.get("fingerprint") != fingerprint:
            previous = checkpoint.get("fingerprint") or {}
            # Operational row concurrency is intentionally mutable on resume:
            # a provider can throttle an otherwise identical study. Preserve
            # every scientific-plan field while recording the new setting.
            mutable = {"max_concurrency"}
            same_plan = (
                {k: v for k, v in previous.items() if k not in mutable}
                == {k: v for k, v in fingerprint.items() if k not in mutable}
            )
            if not same_plan:
                raise SystemExit(
                    f"checkpoint plan does not match this launch: {path}. "
                    "Dataset, order, commit, and error policy must match."
                )
            changes = checkpoint.setdefault("configuration_changes", [])
            changes.append({
                "at": _utc_now(),
                "field": "max_concurrency",
                "from": previous.get("max_concurrency"),
                "to": fingerprint.get("max_concurrency"),
                "reason": "provider rate-limit backoff",
            })
            checkpoint["fingerprint"] = fingerprint
            _write_checkpoint(path, checkpoint)
        return checkpoint
    if args.resume:
        raise SystemExit(f"--resume requested but checkpoint does not exist: {path}")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "fingerprint": fingerprint,
        "conditions": {
            condition.label: {"status": "pending", "attempts": []}
            for condition in conditions
        },
    }
    _write_checkpoint(path, checkpoint)
    return checkpoint


def _max_search_cost_usd(
    conditions: tuple[MatrixCondition, ...],
    rows: int,
    trials: int,
    max_attempts: int,
) -> float:
    per_pass = 0.0
    for condition in conditions:
        if condition.search_mode == "harness":
            per_pass += rows * trials * YDC_MAX_SEARCH_COST_PER_ROW
        elif condition.search_mode == "native":
            rate, _ = native_search_rate_usd(condition.vendor, condition.model)
            per_pass += rows * trials * MAX_SEARCHES * rate
    return per_pass * max_attempts


def _run_conditions(
    args,
    conditions: tuple[MatrixCondition, ...],
    checkpoint_path: Path,
    run_command=subprocess.run,
) -> int:
    checkpoint = _load_or_create_checkpoint(checkpoint_path, args, conditions)
    max_attempts = 1 + args.condition_retries
    for index, condition in enumerate(conditions, start=1):
        state = checkpoint["conditions"][condition.label]
        # The child writes this marker only after Braintrust returns and the row
        # error gate passes. It closes the small crash window between child exit
        # and the parent's next checkpoint write.
        for prior_attempt in reversed(state["attempts"]):
            marker = prior_attempt.get("completion_marker")
            if marker and Path(marker).is_file():
                state["status"] = "completed"
                state["completed_at"] = prior_attempt.get("finished_at") or _utc_now()
                state["reconciled_from_marker"] = True
                _write_checkpoint(checkpoint_path, checkpoint)
                break
        if state["status"] == "completed":
            print(f"[{index:02d}/{len(conditions)}] SKIP completed {condition.label}")
            continue
        if state["status"] == "running" and not args.retry_running:
            raise SystemExit(
                f"{condition.label} was still marked running when the launcher "
                "stopped. Its child process may still be active. Wait and resume "
                "again so its completion marker can be reconciled, or pass "
                "--retry-running only after confirming the prior process is dead."
            )
        attempts_used = len(state["attempts"])
        if attempts_used >= max_attempts:
            print(
                f"[{index:02d}/{len(conditions)}] {condition.label} exhausted "
                f"{max_attempts} attempt(s). Increase --condition-retries to retry."
            )
            return 1

        for attempt in range(attempts_used + 1, max_attempts + 1):
            resume_experiment_attempt = _row_resume_experiment_attempt(
                state, attempt
            )
            condition_hash = hashlib.sha256(
                condition.label.encode("utf-8")
            ).hexdigest()[:12]
            completion_marker = (
                checkpoint_path.parent / ".attempt-markers" /
                f"{checkpoint_path.stem}-{condition_hash}-{attempt:02d}.done"
            )
            command = condition_command(
                condition,
                args,
                index,
                attempt,
                completion_marker,
                resume_experiment_attempt,
            )
            attempt_record = {
                "attempt": attempt,
                "started_at": _utc_now(),
                "command": command,
                "completion_marker": str(completion_marker),
            }
            state["status"] = "running"
            state["attempts"].append(attempt_record)
            _write_checkpoint(checkpoint_path, checkpoint)
            print(
                f"[{index:02d}/{len(conditions)}] {condition.label} "
                f"attempt {attempt}/{max_attempts}"
            )
            print(f"  {shlex.join(command)}")
            try:
                completed = run_command(
                    command, check=False, timeout=args.condition_timeout_s
                )
                returncode = completed.returncode
                failure = None
            except subprocess.TimeoutExpired:
                returncode = 124
                failure = f"condition exceeded {args.condition_timeout_s}s timeout"
            except KeyboardInterrupt:
                attempt_record.update({
                    "finished_at": _utc_now(),
                    "returncode": 130,
                    "failure": "interrupted",
                })
                state["status"] = "interrupted"
                _write_checkpoint(checkpoint_path, checkpoint)
                raise
            except OSError as exc:
                returncode = 126
                failure = f"{type(exc).__name__}: {exc}"

            attempt_record.update({
                "finished_at": _utc_now(),
                "returncode": returncode,
                "failure": failure,
            })
            if returncode == 0:
                state["status"] = "completed"
                state["completed_at"] = _utc_now()
                _write_checkpoint(checkpoint_path, checkpoint)
                break
            state["status"] = "failed"
            _write_checkpoint(checkpoint_path, checkpoint)
            print(f"  attempt failed with exit code {returncode}")
        if state["status"] != "completed":
            print(f"Stopped after failed condition. Resume with: --resume")
            return 1
    checkpoint["status"] = "completed"
    checkpoint["completed_at"] = _utc_now()
    _write_checkpoint(checkpoint_path, checkpoint)
    return 0


@dataclass
class _ActiveCondition:
    index: int
    condition: MatrixCondition
    process: subprocess.Popen
    attempt_record: dict
    deadline: float


def _condition_can_start(
    candidate: MatrixCondition,
    active: tuple[MatrixCondition, ...],
) -> bool:
    """Keep parallel work on independent provider and retrieval surfaces."""
    if any(candidate.vendor == running.vendor for running in active):
        return False
    if candidate.search_mode == "harness" and any(
        running.search_mode == "harness" for running in active
    ):
        return False
    return True


def _next_parallel_condition(
    conditions: tuple[MatrixCondition, ...],
    checkpoint: dict,
    active: dict[str, _ActiveCondition],
    max_attempts: int,
) -> tuple[int, MatrixCondition] | None:
    running = tuple(item.condition for item in active.values())
    eligible = []
    for index, condition in enumerate(conditions, start=1):
        state = checkpoint["conditions"][condition.label]
        if condition.label in active or state["status"] == "completed":
            continue
        if len(state["attempts"]) >= max_attempts:
            continue
        if _condition_can_start(condition, running):
            eligible.append((index, condition))
    if not eligible:
        return None

    # Keep the single You.com lane occupied. Once a harness condition is active,
    # the second slot may only run native/no-search work. Relative order within
    # each class remains the registered randomized matrix order.
    if not any(item.search_mode == "harness" for item in running):
        for item in eligible:
            if item[1].search_mode == "harness":
                return item
    return eligible[0]


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            process.send_signal(sig)
        except (AttributeError, OSError, ProcessLookupError):
            pass


def _stop_process(process: subprocess.Popen, sig: int, grace_s: float = 10.0) -> None:
    _signal_process_group(process, sig)
    deadline = time.monotonic() + grace_s
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        _signal_process_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=2)
        except (AttributeError, subprocess.TimeoutExpired):
            pass


def _reconcile_parallel_checkpoint(
    args,
    conditions: tuple[MatrixCondition, ...],
    checkpoint: dict,
    checkpoint_path: Path,
) -> None:
    changed = False
    for condition in conditions:
        state = checkpoint["conditions"][condition.label]
        if args.retry_failed and state["status"] == "failed":
            state.setdefault("prior_attempts", []).extend(state["attempts"])
            state["attempts"] = []
            state["status"] = "pending"
            changed = True
        for prior_attempt in reversed(state["attempts"]):
            marker = prior_attempt.get("completion_marker")
            if marker and Path(marker).is_file():
                state["status"] = "completed"
                state["completed_at"] = (
                    prior_attempt.get("finished_at") or _utc_now()
                )
                state["reconciled_from_marker"] = True
                changed = True
                break
        if state["status"] == "running":
            if not args.retry_running:
                raise SystemExit(
                    f"{condition.label} was still marked running when the launcher "
                    "stopped. Its child process may still be active. Wait and resume "
                    "again so its completion marker can be reconciled, or pass "
                    "--retry-running only after confirming the prior process is dead."
                )
            prior_attempt = state["attempts"][-1]
            prior_attempt.setdefault("finished_at", _utc_now())
            prior_attempt.setdefault("returncode", 130)
            prior_attempt.setdefault(
                "failure", "abandoned running attempt confirmed dead on resume"
            )
            state["status"] = "interrupted"
            changed = True
    if changed:
        _write_checkpoint(checkpoint_path, checkpoint)


def _run_conditions_parallel(
    args,
    conditions: tuple[MatrixCondition, ...],
    checkpoint_path: Path,
    popen_factory=subprocess.Popen,
) -> int:
    """Run a bounded, provider-aware schedule with one checkpoint owner."""
    checkpoint = _load_or_create_checkpoint(checkpoint_path, args, conditions)
    _reconcile_parallel_checkpoint(args, conditions, checkpoint, checkpoint_path)
    max_attempts = 1 + args.condition_retries
    active: dict[str, _ActiveCondition] = {}
    exhausted_failure = False
    gateway_stop_file = checkpoint_path.parent / "gateway-cap-stop"

    try:
        while True:
            now = time.monotonic()
            if gateway_stop_file.exists():
                print("Gateway budget stop detected; stopping active conditions.")
                for item in active.values():
                    _signal_process_group(item.process, signal.SIGINT)
                for item in active.values():
                    item.process.wait(timeout=10)
                    item.attempt_record.update({
                        "finished_at": _utc_now(),
                        "returncode": 130,
                        "failure": "gateway budget stop",
                    })
                    checkpoint["conditions"][item.condition.label]["status"] = "failed"
                checkpoint["status"] = "gateway_budget_exhausted"
                _write_checkpoint(checkpoint_path, checkpoint)
                return 2
            for label, item in list(active.items()):
                returncode = item.process.poll()
                failure = None
                if returncode is None and now >= item.deadline:
                    _stop_process(item.process, signal.SIGTERM)
                    returncode = 124
                    failure = (
                        f"condition exceeded {args.condition_timeout_s}s timeout"
                    )
                if returncode is None:
                    continue

                item.attempt_record.update({
                    "finished_at": _utc_now(),
                    "returncode": returncode,
                    "failure": failure,
                })
                state = checkpoint["conditions"][label]
                if returncode == 0:
                    state["status"] = "completed"
                    state["completed_at"] = _utc_now()
                    print(
                        f"[{item.index:02d}/{len(conditions)}] COMPLETE {label}",
                        flush=True,
                    )
                else:
                    state["status"] = "failed"
                    print(
                        f"[{item.index:02d}/{len(conditions)}] {label} "
                        f"attempt failed with exit code {returncode}",
                        flush=True,
                    )
                    if len(state["attempts"]) >= max_attempts:
                        exhausted_failure = True
                del active[label]
                _write_checkpoint(checkpoint_path, checkpoint)

            # An exhausted condition must not block independent conditions;
            # keep scheduling everything else and report a nonzero result only
            # after no runnable work remains.
            while len(active) < args.condition_concurrency:
                selected = _next_parallel_condition(
                    conditions, checkpoint, active, max_attempts
                )
                if selected is None:
                    break
                index, condition = selected
                state = checkpoint["conditions"][condition.label]
                attempt = len(state["attempts"]) + 1
                resume_experiment_attempt = _row_resume_experiment_attempt(
                    state, attempt
                )
                condition_hash = hashlib.sha256(
                    condition.label.encode("utf-8")
                ).hexdigest()[:12]
                completion_marker = (
                    checkpoint_path.parent / ".attempt-markers" /
                    f"{checkpoint_path.stem}-{condition_hash}-{attempt:02d}.done"
                )
                command = condition_command(
                    condition,
                    args,
                    index,
                    attempt,
                    completion_marker,
                    resume_experiment_attempt,
                )
                attempt_record = {
                    "attempt": attempt,
                    "started_at": _utc_now(),
                    "command": command,
                    "completion_marker": str(completion_marker),
                }
                state["status"] = "running"
                state["attempts"].append(attempt_record)
                _write_checkpoint(checkpoint_path, checkpoint)
                print(
                    f"[{index:02d}/{len(conditions)}] {condition.label} "
                    f"attempt {attempt}/{max_attempts}",
                    flush=True,
                )
                print(f"  {shlex.join(command)}", flush=True)
                try:
                    process = popen_factory(command, start_new_session=True)
                except OSError as exc:
                    attempt_record.update({
                        "finished_at": _utc_now(),
                        "returncode": 126,
                        "failure": f"{type(exc).__name__}: {exc}",
                    })
                    state["status"] = "failed"
                    _write_checkpoint(checkpoint_path, checkpoint)
                    if len(state["attempts"]) >= max_attempts:
                        exhausted_failure = True
                        break
                    continue
                active[condition.label] = _ActiveCondition(
                        index=index,
                        condition=condition,
                        process=process,
                        attempt_record=attempt_record,
                        deadline=time.monotonic() + args.condition_timeout_s,
                    )

            if all(
                checkpoint["conditions"][condition.label]["status"] == "completed"
                for condition in conditions
            ):
                checkpoint["status"] = "completed"
                checkpoint["completed_at"] = _utc_now()
                _write_checkpoint(checkpoint_path, checkpoint)
                return 0
            if not active:
                # Nothing can run only when every incomplete condition has used
                # all allowed attempts. Fail closed rather than spinning.
                return 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        for item in active.values():
            _signal_process_group(item.process, signal.SIGINT)
        deadline = time.monotonic() + 10.0
        while (any(item.process.poll() is None for item in active.values())
               and time.monotonic() < deadline):
            time.sleep(0.05)
        for label, item in active.items():
            if item.process.poll() is None:
                _stop_process(item.process, signal.SIGTERM, grace_s=2.0)
            marker = item.attempt_record.get("completion_marker")
            state = checkpoint["conditions"][label]
            if marker and Path(marker).is_file():
                state["status"] = "completed"
                state["completed_at"] = _utc_now()
                state["reconciled_from_marker"] = True
                failure = None
                returncode = item.process.poll() or 0
            else:
                state["status"] = "interrupted"
                failure = "interrupted"
                returncode = 130
            item.attempt_record.update({
                "finished_at": _utc_now(),
                "returncode": returncode,
                "failure": failure,
            })
        _write_checkpoint(checkpoint_path, checkpoint)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="Required for an unlimited execution; each condition verifies this count.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--order-seed",
        help="Reproducible condition-order seed (defaults to --study-id).",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-running", action="store_true",
        help="Retry a checkpoint attempt left in 'running' state after first confirming its child process is dead.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry failed conditions from row checkpoints, preserving prior attempts.",
    )
    parser.add_argument(
        "--condition-retries", type=int, default=1,
        help="Additional attempts for a failed or timed-out condition.",
    )
    parser.add_argument(
        "--condition-timeout-s", type=int, default=DEFAULT_CONDITION_TIMEOUT_S,
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--ydc-requests-per-second", type=float,
        default=DEFAULT_YDC_REQUESTS_PER_SECOND,
        help="Shared You.com request rate across local processes (default: 1; "
             f"hard cap: {MAX_YDC_REQUESTS_PER_SECOND:g}).",
    )
    parser.add_argument(
        "--condition-concurrency", type=int,
        default=DEFAULT_CONDITION_CONCURRENCY,
        choices=range(1, MAX_SAFE_CONDITION_CONCURRENCY + 1),
        help=(
            "Concurrent matrix conditions. Two uses the safe scheduler: at most "
            "one harness arm and one condition per vendor at a time."
        ),
    )
    parser.add_argument(
        "--max-row-error-rate", type=float, default=0.0,
        help="A condition exits nonzero when its task/scorer error rate exceeds this.",
    )
    parser.add_argument(
        "--max-row-executions", type=int, default=DEFAULT_MAX_ROW_EXECUTIONS,
    )
    parser.add_argument(
        "--max-search-cost-usd", type=float,
        help="Required with --execute. Conservative search-fee ceiling including retries; excludes model and judge inference.",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow tracked changes; the commit still becomes part of the checkpoint fingerprint.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the commands. Without this flag, print the launch plan only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.expected_rows is not None and args.expected_rows < 1:
        raise SystemExit("--expected-rows must be at least 1")
    if (args.limit is not None and args.expected_rows is not None
            and args.limit != args.expected_rows):
        raise SystemExit("--limit and --expected-rows must match when both are set")
    if args.condition_retries < 0:
        raise SystemExit("--condition-retries cannot be negative")
    if args.retry_running and not args.resume:
        raise SystemExit("--retry-running requires --resume")
    if args.retry_failed and not args.resume:
        raise SystemExit("--retry-failed requires --resume")
    if args.condition_timeout_s <= 60:
        raise SystemExit("--condition-timeout-s must be greater than 60")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")
    if not 0 < args.ydc_requests_per_second <= MAX_YDC_REQUESTS_PER_SECOND:
        raise SystemExit(
            "--ydc-requests-per-second must be greater than zero and no more "
            f"than {MAX_YDC_REQUESTS_PER_SECOND:g}"
        )
    if not 0 <= args.max_row_error_rate <= 1:
        raise SystemExit("--max-row-error-rate must be between 0 and 1")

    order_seed = args.order_seed or args.study_id
    conditions = ordered_matrix(order_seed)
    planned_rows = args.limit or args.expected_rows
    if args.execute and planned_rows is None:
        raise SystemExit(
            "--expected-rows is required for a full execution so row and cost "
            "guards cannot silently use a stale dataset size"
        )
    if args.execute and args.max_search_cost_usd is None:
        raise SystemExit(
            "--max-search-cost-usd is required with --execute; this acknowledges "
            "the conservative search-fee ceiling (model/judge inference is extra)"
        )
    if args.execute and not args.allow_dirty and not _tracked_worktree_is_clean():
        raise SystemExit(
            "tracked working-tree changes detected; commit them before a full run "
            "or pass --allow-dirty for an intentional exploratory launch"
        )
    if planned_rows is not None:
        nominal_row_executions = planned_rows * args.trials * len(conditions)
        max_row_executions = nominal_row_executions * (1 + args.condition_retries)
        if max_row_executions > args.max_row_executions:
            raise SystemExit(
                f"worst-case row executions {max_row_executions:,} exceed "
                f"--max-row-executions {args.max_row_executions:,}"
            )
        search_ceiling = _max_search_cost_usd(
            conditions, planned_rows, args.trials, 1 + args.condition_retries
        )
        if (args.max_search_cost_usd is not None
                and search_ceiling > args.max_search_cost_usd):
            raise SystemExit(
                f"conservative search-fee ceiling ${search_ceiling:.2f} exceeds "
                f"--max-search-cost-usd ${args.max_search_cost_usd:.2f}"
            )
    else:
        nominal_row_executions = None
        max_row_executions = None
        search_ceiling = None
    print(
        f"Matrix: {len(conditions)} conditions | dataset={args.dataset_name} "
        f"| trials={args.trials} | study={args.study_id} "
        f"| order_seed={order_seed} | condition_concurrency="
        f"{args.condition_concurrency} | ydc_rps={args.ydc_requests_per_second:g} "
        f"| schedule={CONDITION_SCHEDULE_POLICY}"
    )
    if nominal_row_executions is not None:
        print(
            f"Guarded plan: {nominal_row_executions:,} nominal / "
            f"{max_row_executions:,} max row executions | "
            f"max search fees including retries=${search_ceiling:.2f} | "
            "model/judge inference excluded"
        )
    for index, condition in enumerate(conditions, start=1):
        command = condition_command(condition, args, index, 1)
        print(f"[{index:02d}/{len(conditions)}] {condition.label}")
        print(f"  {shlex.join(command)}")

    if not args.execute:
        print("\nDry run only. Add --execute to launch.")
        return 0
    checkpoint_path = _checkpoint_path(args)
    print(f"Checkpoint: {checkpoint_path}")
    with _checkpoint_lock(checkpoint_path):
        if args.condition_concurrency > 1:
            return _run_conditions_parallel(args, conditions, checkpoint_path)
        return _run_conditions(args, conditions, checkpoint_path)


if __name__ == "__main__":
    raise SystemExit(main())
