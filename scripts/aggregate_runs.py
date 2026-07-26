#!/usr/bin/env python3
"""Aggregate local and imported standalone results without cross-GPU ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.portability.report import build_aggregate_report, write_aggregate_report
from src.kernelblaster.storage import StateStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    store = StateStore(state_dir=args.state_dir)
    report = build_aggregate_report(store.repository)
    paths = write_aggregate_report(report, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
