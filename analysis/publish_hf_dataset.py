#!/usr/bin/env python3
"""Publish hf_dataset/ to the Hugging Face Hub as a private, gated dataset.

Order matters. The repo is created private, gating is switched on, and only then
is the data uploaded, so the rows are never reachable while the gate is off.
Flipping to public is left as a separate manual step in the Hub UI.

    python analysis/publish_hf_dataset.py --dry-run
    python analysis/publish_hf_dataset.py --confirm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "hf_dataset"
DEFAULT_REPO = "BraintrustDataDev/livenewsbench-search-arms"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--gated", default="manual", choices=["auto", "manual"],
                        help="auto approves access requests; manual requires review")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true",
                        help="required to actually create the repo and upload")
    args = parser.parse_args()

    files = sorted(p for p in FOLDER.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"nothing to upload in {FOLDER}")

    print(f"repo    {args.repo_id}  (dataset)")
    print(f"gated   {args.gated}")
    print("private true on create; make it public manually once reviewed")
    print(f"folder  {FOLDER}")
    for path in files:
        print(f"  {path.relative_to(FOLDER)}  {path.stat().st_size:,} bytes")

    api = HfApi()
    who = api.whoami()
    print(f"\nauthenticated as {who.get('name')}")
    namespace = args.repo_id.split("/")[0]
    roles = {org.get("name"): org.get("roleInOrg") for org in who.get("orgs") or []}
    if namespace == who.get("name"):
        print("namespace is the authenticated user")
    elif namespace in roles:
        print(f"org role in {namespace}: {roles[namespace]}")
        if roles[namespace] not in {"admin", "write", "contributor"}:
            print(f"WARNING: role '{roles[namespace]}' may not permit creating a dataset")
    else:
        # A fine-grained token scoped to specific repos reports no orgs at all,
        # and the org list has come back empty transiently, so this is advisory.
        print(f"NOTE: this token lists no membership in '{namespace}'. That is "
              f"expected for a fine-grained token; if the push 403s, use a token "
              f"with write access to the org.")

    if args.dry_run or not args.confirm:
        print("\ndry run; pass --confirm to create the repo and upload")
        return 0

    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    print("created (private)")
    api.update_repo_settings(args.repo_id, repo_type="dataset", gated=args.gated)
    print(f"gating set to {args.gated}")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(FOLDER),
        commit_message="Add LiveNewsBench search-arm results (18,606 paired rows)",
    )
    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}")
    print("Review the card and the access form, then switch the repo to public.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
