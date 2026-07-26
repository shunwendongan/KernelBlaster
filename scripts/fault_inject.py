#!/usr/bin/env python3
"""Emit the release fault matrix; dangerous faults require explicit opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.release import fault_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-real-faults", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = {"schema_version": "release-fault-plan/v1", "faults": fault_plan(allow_real_faults=args.allow_real_faults)}
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
