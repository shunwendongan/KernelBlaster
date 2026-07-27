#!/usr/bin/env python3
"""Export one terminal local run as a portable, deterministic bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.portability.exporter import export_run
from src.kernelblaster.storage import StateStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--gzip", action="store_true")
    args = parser.parse_args()
    result = export_run(
        StateStore(state_dir=args.state_dir),
        args.run_id,
        args.bundle,
        allow_incomplete=args.allow_incomplete,
        gzip_compress=args.gzip,
    )
    print(json.dumps(result.__dict__, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
