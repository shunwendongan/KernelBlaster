#!/usr/bin/env python3
"""Create sanitized, hash-indexed local or AutoDL release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.release import build_release_manifest, load_release_profile, write_release_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", help="for example local-rtx3080 or autodl-a100")
    parser.add_argument("--root", type=Path, default=Path("release-evidence"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    evidence = json.loads(args.evidence_json.read_text(encoding="utf-8"))
    sources: dict[str, Path] = {}
    for entry in args.include:
        name, separator, path = entry.partition("=")
        if not separator or not name:
            parser.error("--include must use NAME=PATH")
        sources[name] = Path(path)
    profile = load_release_profile(args.profile).to_dict()
    written = write_release_evidence(args.root, scope=args.scope, profile=profile, evidence=evidence, source_files=sources)
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
