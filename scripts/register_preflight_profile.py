#!/usr/bin/env python3
"""Upload the fixed private preflight driver bundle and write its external profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.gpu_jobs import build_deterministic_bundle  # noqa: E402
from src.kernelblaster.preflight.client import ControlPlaneClient  # noqa: E402


PROFILE_ID = "preflight-vector-add-v1"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def _run(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise RuntimeError("private evaluation profiles must be written outside the repository")
    if not args.token:
        raise RuntimeError("KERNELBLASTER_CONTROL_TOKEN is required")
    driver = (ROOT / "portfolio" / "trusted_gpu_smoke" / "driver.cpp").read_bytes()
    bundle = build_deterministic_bundle({"driver.cpp": driver})
    uploaded = await ControlPlaneClient(args.control_url, args.token).upload(
        bundle,
        media_type="application/x-tar",
        schema="gpu-private-evaluation-bundle/v1",
    )
    profile = {
        "id": PROFILE_ID,
        "bundle_digest": uploaded["digest"],
        "driver_path": "driver.cpp",
    }
    manifest = {
        "schema_version": "gpu-private-evaluation-profiles/v1",
        "profiles": [],
    }
    if output.exists():
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "gpu-private-evaluation-profiles/v1":
            raise RuntimeError("existing private profile manifest has an unsupported schema")
    profiles = list(manifest.get("profiles") or ())
    existing = [item for item in profiles if item.get("id") == PROFILE_ID]
    if existing and existing[0] != profile:
        raise RuntimeError("preflight profile ID already resolves to a different bundle")
    if not existing:
        profiles.append(profile)
    manifest["profiles"] = sorted(profiles, key=lambda item: str(item["id"]))
    _atomic_json(output, manifest)
    print(
        json.dumps(
            {
                "profile_id": PROFILE_ID,
                "bundle_digest": uploaded["digest"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("KERNELBLASTER_CONTROL_TOKEN"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        parser.exit(2, f"Profile registration failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
