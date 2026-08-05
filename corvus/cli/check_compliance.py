#!/usr/bin/env python3
"""Read-only Corvus compliance preflight; never prints credentials."""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path

from corvus.compliance import (
    SOURCE_POLICY_PATH,
    load_source_policy,
    require_import_approval,
)
from import_livenewsbench import load_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--source-policy", type=Path, default=SOURCE_POLICY_PATH)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--import-approval", type=Path)
    parser.add_argument("--schema-file", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    policy = load_source_policy(args.source_policy)
    print(f"PASS source review valid through {policy['review_valid_until']}")

    if not args.env_file.is_file():
        failures.append(f"{args.env_file}: missing")
        env = {}
    else:
        permissions = stat.S_IMODE(args.env_file.stat().st_mode)
        if permissions & 0o077:
            failures.append(
                f"{args.env_file}: permissions are {permissions:o}; expected 600"
            )
        else:
            print(f"PASS {args.env_file} is not group/world-accessible")
        env = load_env(args.env_file)

    contact = env.get("CORVUS_CONTACT_EMAIL", "")
    if contact.count("@") != 1 or any(ch.isspace() for ch in contact):
        failures.append("CORVUS_CONTACT_EMAIL is missing or invalid")
    else:
        print("PASS contact email is configured (value hidden)")
    for name, label in (
        (
            "CORVUS_CONTACT_EMAIL_IS_ROLE_ACCOUNT",
            "monitored organizational role inbox",
        ),
        (
            "CORVUS_SINGLE_DEPLOYMENT_CONFIRMED",
            "single SEC collector deployment",
        ),
    ):
        if env.get(name) != "yes":
            failures.append(f"{name}=yes not attested ({label})")
        else:
            print(f"PASS {label} attested")
    for name, label in (
        ("CORVUS_WIKIPEDIA_TERMS_CONFIRMED", "Wikipedia Current Events"),
        ("CORVUS_OPENLIGADB_LICENSE_CONFIRMED", "OpenLigaDB"),
        ("CORVUS_THESPORTSDB_TERMS_CONFIRMED", "TheSportsDB"),
    ):
        if env.get(name) == "yes":
            print(f"PASS {label} terms attested")
        else:
            print(f"NOTICE {label} adapter disabled until {name}=yes")

    if args.schema_file and not args.artifact:
        failures.append("--schema-file requires --artifact and --import-approval")
    if bool(args.artifact) != bool(args.import_approval):
        failures.append("--artifact and --import-approval must be supplied together")
    elif args.artifact and args.import_approval:
        require_import_approval(
            args.import_approval,
            artifact_path=args.artifact,
            source_policy_path=args.source_policy,
            schema_path=args.schema_file,
        )
        suffix = ", source policy, and schema" if args.schema_file else " and source policy"
        print(f"PASS private-import approval matches artifact{suffix}")
    else:
        print("PASS external Corvus import remains gated (no approval supplied)")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS collection preflight complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
