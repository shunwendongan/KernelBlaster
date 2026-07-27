#!/usr/bin/env python3
"""Create a release rollback backup of the local SQLite control database."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.kernelblaster.release import create_state_backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    print(create_state_backup(args.state_dir, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
