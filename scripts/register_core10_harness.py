#!/usr/bin/env python3
"""Validate and upload Core 10 TaskSpecs, disclosed cases and trusted baselines."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.gpu_jobs import build_deterministic_bundle  # noqa: E402
from src.kernelblaster.harness import (  # noqa: E402
    CaseBundle,
    build_development_case_bundle,
    core10_task_specs,
)
from src.kernelblaster.preflight.client import ControlPlaneClient  # noqa: E402


def _source_for(task_id: str) -> tuple[str, bytes, bytes | None]:
    number = task_id.split(".")[2]
    direction = task_id.rsplit(".", 1)[-1]
    if direction == "forward":
        matches = sorted((ROOT / "data" / "kernelbench-cuda" / "level1").glob(f"{number}_*"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one upstream task directory for {number}")
        return "init.cu", (matches[0] / "init.cu").read_bytes(), (matches[0] / "driver.cpp").read_bytes()
    matches = sorted((ROOT / "portfolio" / "harness" / "core10" / "backward").glob(f"{number}_*.cu"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one backward baseline for {number}")
    return "baseline.cu", matches[0].read_bytes(), None


def _external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(ROOT):
        raise RuntimeError("private case catalogs must remain outside the repository")
    return resolved


async def _run(args: argparse.Namespace) -> int:
    if not args.token:
        raise RuntimeError("KERNELBLASTER_CONTROL_TOKEN is required")
    case_root = _external(args.case_root)
    output = _external(args.output)
    case_root.mkdir(parents=True, exist_ok=True)
    client = ControlPlaneClient(args.control_url, args.token)
    entries: list[dict[str, object]] = []
    for task in core10_task_specs():
        case_path = case_root / f"{task.id}.json"
        if case_path.exists():
            case_bundle = CaseBundle.model_validate_json(case_path.read_bytes())
        elif args.public_fixtures:
            case_bundle = build_development_case_bundle(task)
            case_path.write_bytes(case_bundle.canonical_bytes() + b"\n")
        else:
            raise FileNotFoundError(f"missing external case bundle: {case_path}")
        case_bundle.validate_for(task)
        task_upload = await client.upload(
            task.canonical_bytes(), media_type="application/json", schema="harness-task/v1"
        )
        case_upload = await client.upload(
            case_bundle.canonical_bytes(),
            media_type="application/json",
            schema="harness-case-bundle/v1",
        )
        source_name, source, driver = _source_for(task.id)
        files = {
            "task-spec.json": task.canonical_bytes(),
            "case-bundle.json": case_bundle.canonical_bytes(),
            source_name: source,
        }
        if driver is not None:
            files["driver.cpp"] = driver
        bundle_upload = await client.upload(
            build_deterministic_bundle(files),
            media_type="application/x-tar",
            schema="harness-private-evaluation-bundle/v1",
        )
        entries.append(
            {
                "task_id": task.id,
                "direction": task.direction.value,
                "adapter_id": task.adapter_id,
                "adapter_version": task.adapter_version,
                "task_spec_digest": task_upload["digest"],
                "case_bundle_digest": case_upload["digest"],
                "evaluation_bundle_digest": bundle_upload["digest"],
                "baseline_source_sha256": hashlib.sha256(source).hexdigest(),
            }
        )
    catalog = {
        "schema_version": "harness-catalog/v1",
        "disclosure": "adaptive_disclosed",
        "tasks": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "tasks": len(entries)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("KERNELBLASTER_CONTROL_TOKEN"))
    parser.add_argument(
        "--case-root", type=Path, default=Path.home() / "secrets" / "kernelblaster" / "cases"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "secrets" / "kernelblaster" / "core10-harness-catalog.json",
    )
    parser.add_argument(
        "--public-fixtures",
        action="store_true",
        help="Seed external files with deterministic disclosed fixtures when absent",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        parser.exit(2, f"Harness registration failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
