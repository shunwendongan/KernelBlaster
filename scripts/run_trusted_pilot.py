#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout: float,
):
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=str(error.stdout or ""),
            stderr=f"TimeoutExpired: exceeded {timeout} seconds",
        )
    except OSError as error:
        completed = subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated RMSNorm Agent Pilot in the required startup order."
    )
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-5.6-terra"))
    parser.add_argument("--gpu", default="rtx3080")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "out" / "trusted-pilot" / _timestamp(),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    stages: list[dict[str, Any]] = []

    # 1. Environment and dependency check.
    runtime = _run(
        [sys.executable, "scripts/check_runtime_versions.py", "--require-gpu"],
        log_path=output_dir / "01-runtime.log",
        timeout=120,
    )
    stages.append({"stage": "runtime", "returncode": runtime.returncode})
    if runtime.returncode:
        _atomic_json(output_dir / "preflight.json", {"schema_version": "2.0", "stages": stages})
        return 2

    # 2-4. CUDA compilation, official/edge correctness and three-session Events stability.
    events_dir = output_dir / "events-discovery"
    events = _run(
        [
            sys.executable,
            "scripts/benchmark_candidates.py",
            "--task-id",
            "036",
            "--phase",
            "discovery",
            "--cooldown-seconds",
            "0",
            "--output-dir",
            str(events_dir),
        ],
        log_path=output_dir / "02-04-events.log",
        timeout=3600,
    )
    stages.append({"stage": "compile_correctness_events", "returncode": events.returncode})
    if events.returncode:
        _atomic_json(output_dir / "preflight.json", {"schema_version": "2.0", "stages": stages})
        return 2

    # 5. Optional NCU permission probe; ERR_NVGPUCTRPERM explicitly selects events_only.
    executable = (
        events_dir
        / "036"
        / "build"
        / "rmsnorm_v3c"
        / "benchmark"
        / "build"
        / "main"
    )
    ncu = _run(
        ["ncu", "--section", "SpeedOfLight", str(executable)],
        log_path=output_dir / "05-ncu-probe.log",
        timeout=600,
    )
    ncu_text = ncu.stdout + ncu.stderr
    profiling_mode = "ncu" if ncu.returncode == 0 else "events_only"
    ncu_status = (
        "blocked_permission"
        if "ERR_NVGPUCTRPERM" in ncu_text
        else ("available" if ncu.returncode == 0 else "unavailable")
    )
    stages.append(
        {
            "stage": "ncu_permission_probe",
            "returncode": ncu.returncode,
            "status": ncu_status,
            "profiling_mode": profiling_mode,
        }
    )

    # 6. Exactly one bounded authentication smoke request.
    smoke = _run(
        [
            sys.executable,
            "scripts/smoke_llm.py",
            "--model",
            args.model,
            "--max-completion-tokens",
            "64",
            "--max-total-tokens",
            "10000",
            "--output-dir",
            str(output_dir / "api-smoke"),
        ],
        log_path=output_dir / "06-api-smoke.log",
        timeout=300,
    )
    stages.append({"stage": "api_smoke", "returncode": smoke.returncode})
    if smoke.returncode:
        _atomic_json(output_dir / "preflight.json", {"schema_version": "2.0", "stages": stages})
        return 2

    # 7. RMSNorm Pilot only: 2 rollouts x 2 steps, bounded to 32 requests/250k tokens.
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_MAX_REQUESTS": "32",
            "LLM_MAX_TOTAL_TOKENS": "250000",
            "LLM_MAX_CONCURRENCY": "2",
        }
    )
    pilot = _run(
        [
            sys.executable,
            "scripts/run_RL.py",
            "--experiment-name",
            "trusted-rmsnorm-pilot",
            "--dataset",
            "kernelbench-cuda",
            "--precision",
            "fp16",
            "--cuda",
            "--cuda-perf",
            "--use-rl",
            "--rl-iterations",
            "2",
            "--rl-rollout-steps",
            "2",
            "--rl-buffer-size",
            "16",
            "--rl-update-frequency",
            "2",
            "--concurrency",
            "1",
            "--problem-numbers",
            "36",
            "--subset",
            "level1",
            "--gpu",
            args.gpu,
            "--model",
            args.model,
            "--run-record-dir",
            str(output_dir / "pilot-record"),
        ],
        log_path=output_dir / "07-pilot.log",
        env=environment,
        timeout=7200,
    )
    stages.append({"stage": "agent_pilot", "returncode": pilot.returncode})
    _atomic_json(
        output_dir / "preflight.json",
        {
            "schema_version": "2.0",
            "profiling_mode": profiling_mode,
            "stages": stages,
        },
    )
    return pilot.returncode


if __name__ == "__main__":
    raise SystemExit(main())
