#!/usr/bin/env python3
"""Write or execute the bounded local/AutoDL release E2E command plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from src.kernelblaster.release import build_e2e_plan, load_release_profile
from src.kernelblaster.release.e2e import CAPABILITY_DIGEST_PLACEHOLDER


def _default_output() -> Path:
    return Path("release-evidence") / "plans" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_results(output: Path, results: list[dict[str, object]]) -> None:
    (output / "e2e-result.json").write_text(
        json.dumps({"stages": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_e2e_plan(plan: list[dict[str, object]], output: Path) -> int:
    """Execute public stages and bind the Agent to the preflight report digest."""
    capability_digest: str | None = None
    results: list[dict[str, object]] = []
    for stage in plan:
        command = list(stage["command"])  # type: ignore[arg-type]
        if command[0] == "profile-driven":
            results.append({"name": stage["name"], "status": "owned_by_agent_funnel"})
            continue
        if CAPABILITY_DIGEST_PLACEHOLDER in command:
            if not capability_digest:
                results.append({"name": stage["name"], "status": "blocked_missing_capability_digest"})
                _write_results(output, results)
                return 2
            command[command.index(CAPABILITY_DIGEST_PLACEHOLDER)] = capability_digest
        environment = os.environ.copy()
        environment.update(stage.get("environment", {}))  # type: ignore[arg-type]
        try:
            result = subprocess.run(
                command,
                check=False,
                timeout=int(stage["timeout_seconds"]),  # type: ignore[arg-type]
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            results.append({"name": stage["name"], "status": "timed_out"})
            _write_results(output, results)
            return 124
        if result.stdout:
            print(result.stdout, end="")
        results.append({"name": stage["name"], "status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode})
        if stage["name"] == "capability_preflight" and result.returncode == 0:
            marker = "KERNELBLASTER_PREFLIGHT_JSON "
            for line in result.stdout.splitlines():
                if line.startswith(marker):
                    capability_digest = str(json.loads(line[len(marker):])["report_digest"])
                    break
            if not capability_digest:
                results[-1]["status"] = "failed_missing_capability_digest"
                _write_results(output, results)
                return 2
        if result.returncode:
            _write_results(output, results)
            return result.returncode
    _write_results(output, results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path, default=_default_output())
    parser.add_argument("--execute", action="store_true", help="Run executable stages after writing the plan.")
    parser.add_argument("--mode", choices=("local", "autodl"), default=None)
    args = parser.parse_args()
    profile = load_release_profile(args.profile, overrides={"mode": args.mode})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = build_e2e_plan(profile, output)
    (output / "e2e-plan.json").write_text(json.dumps({"profile": profile.to_dict(), "stages": plan}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.execute:
        print(output / "e2e-plan.json")
        return 0
    return execute_e2e_plan(plan, output)


if __name__ == "__main__":
    raise SystemExit(main())
