#!/usr/bin/env python3
"""Verify the SHA256 index of a release evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.release import verify_release_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, default=Path("release-evidence"), nargs="?")
    args = parser.parse_args()
    result = verify_release_evidence(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
