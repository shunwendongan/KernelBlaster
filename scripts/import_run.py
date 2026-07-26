#!/usr/bin/env python3
"""Strictly import a portable KernelBlaster run bundle into local state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.portability.importer import import_run
from src.kernelblaster.storage import StateStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    result = import_run(StateStore(state_dir=args.state_dir), args.bundle)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
