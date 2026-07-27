#!/usr/bin/env python3
"""Restore a SQLite state backup only after explicit confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.kernelblaster.release import restore_state_backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    print(restore_state_backup(args.backup_dir, args.state_dir, confirm=args.yes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
