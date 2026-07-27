#!/usr/bin/env python3
"""Show or explicitly rotate the stable identity of a standalone instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kernelblaster.portability import load_or_create_instance_identity, rotate_instance_identity
from src.kernelblaster.storage import StateStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("show", "rotate"))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--yes", action="store_true", help="Required for rotate.")
    args = parser.parse_args()
    store = StateStore(state_dir=args.state_dir)
    if args.command == "rotate":
        if not args.yes:
            parser.error("rotate requires --yes because copied state disks need an intentional new identity")
        identity = rotate_instance_identity(store.state_dir)
        store.repository.register_instance(identity.to_dict())
    else:
        identity = load_or_create_instance_identity(store.state_dir)
    print(json.dumps(identity.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
