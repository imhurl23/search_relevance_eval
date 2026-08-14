"""Launch the finalized 14-condition retrieval matrix.

The default action prints every command. Pass ``--execute`` only after checking
the dataset version, study ID, and optional row limit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agents import MATRIX_MODELS
from import_livenewsbench import DATASET_NAME


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


def condition_command(condition: MatrixCondition, args) -> list[str]:
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
    ]
    if condition.arm:
        command.extend(["--arm", condition.arm])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
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

    print(
        f"Matrix: {len(MATRIX)} conditions | dataset={args.dataset_name} "
        f"| trials={args.trials} | study={args.study_id}"
    )
    for index, condition in enumerate(MATRIX, start=1):
        command = condition_command(condition, args)
        print(f"[{index:02d}/{len(MATRIX)}] {condition.label}")
        print(f"  {shlex.join(command)}")
        if args.execute:
            subprocess.run(command, check=True)

    if not args.execute:
        print("\nDry run only. Add --execute to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
