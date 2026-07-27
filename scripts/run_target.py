#!/usr/bin/env python3
"""Run a whitelisted portability operation on one explicit SSH target."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.kernelblaster.portability.targets import load_targets, run_explicit_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "smoke", "export"))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    target = load_targets(args.targets).get(args.target)
    if target is None:
        parser.error(f"target is not configured: {args.target}")
    result = run_explicit_target(target, args.operation, args.arguments)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
