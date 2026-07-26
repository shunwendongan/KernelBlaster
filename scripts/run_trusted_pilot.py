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

OPENAI_BASE_URL = "https://api.openai.com/v1"
PILOT_MODEL = "gpt-5.6-sol"
LLM_SECRET_ENV_VARS = ("KERNELBLASTER_LLM_API_KEY", "OPENAI_API_KEY")
PREFLIGHT_MARKER = "KERNELBLASTER_PREFLIGHT_JSON "
USAGE_FIELDS = (
    "requests_started",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


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


def _environment_without_llm_secrets() -> dict[str, str]:
    environment = os.environ.copy()
    for name in LLM_SECRET_ENV_VARS:
        environment.pop(name, None)
    return environment


def _preflight_result(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(PREFLIGHT_MARKER):
            payload = json.loads(line[len(PREFLIGHT_MARKER) :])
            if isinstance(payload, dict):
                return payload
    raise ValueError("preflight output did not contain its result marker")


def _llm_usage(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        llm = payload.get("llm") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        llm = {}
    return {field: int(llm.get(field, 0) or 0) for field in USAGE_FIELDS}


def _combined_usage(output_dir: Path) -> dict[str, dict[str, int]]:
    preflight = _llm_usage(output_dir / "runtime-preflight" / "summary.json")
    agent = _llm_usage(output_dir / "pilot-record" / "summary.json")
    return {
        "preflight": preflight,
        "agent": agent,
        "total": {field: preflight[field] + agent[field] for field in USAGE_FIELDS},
    }


def _pilot_summary(
    output_dir: Path,
    *,
    report_digest: Any,
    agent_mode: Any,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "trusted-pilot-preflight/v1",
        "report_digest": report_digest,
        "agent_mode": agent_mode,
        "stages": stages,
        "llm_usage": _combined_usage(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated RMSNorm Agent Pilot in the required startup order."
    )
    parser.add_argument(
        "--model",
        default=PILOT_MODEL,
        choices=(PILOT_MODEL,),
        help="Fixed trusted-pilot model; alternate model IDs are rejected.",
    )
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

    # One runtime contract owns Provider auth, storage, sandbox, Events, NSYS, and NCU.
    preflight = _run(
        [
            sys.executable,
            "scripts/run_preflight.py",
            "--model",
            args.model,
            "--gpu",
            args.gpu,
            "--output-dir",
            str(output_dir / "runtime-preflight"),
        ],
        log_path=output_dir / "01-runtime-preflight.log",
        timeout=300,
    )
    stages.append({"stage": "runtime_preflight", "returncode": preflight.returncode})
    try:
        preflight_result = _preflight_result(preflight.stdout)
    except (json.JSONDecodeError, ValueError):
        preflight_result = {"report_digest": None, "agent_mode": "unavailable"}
    _atomic_json(
        output_dir / "preflight.json",
        _pilot_summary(
            output_dir,
            report_digest=preflight_result.get("report_digest"),
            agent_mode=preflight_result.get("agent_mode"),
            stages=stages,
        ),
    )
    if preflight.returncode or not preflight_result.get("report_digest"):
        _atomic_json(
            output_dir / "preflight.json",
            _pilot_summary(
                output_dir,
                report_digest=preflight_result.get("report_digest"),
                agent_mode=preflight_result.get("agent_mode", "unavailable"),
                stages=stages,
            ),
        )
        return 2

    # RMSNorm Pilot only: 2 rollouts x 2 steps, bounded to 32 requests/250k tokens.
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_MAX_REQUESTS": "32",
            "LLM_MAX_TOTAL_TOKENS": "250000",
            "LLM_MAX_CONCURRENCY": "2",
            "LLM_MAX_RETRIES": "2",
            "LLM_REASONING_EFFORT": "low",
            "KERNELBLASTER_LLM_PROVIDER": "openai_compatible",
            "KERNELBLASTER_LLM_BASE_URL": OPENAI_BASE_URL,
            "MODEL": args.model,
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
            "--portfolio-suite",
            "portfolio/suites/rmsnorm.json",
            "--subset",
            "level1",
            "--gpu",
            args.gpu,
            "--model",
            args.model,
            "--run-record-dir",
            str(output_dir / "pilot-record"),
            "--execution-backend",
            "sandbox",
            "--capability-report-digest",
            str(preflight_result["report_digest"]),
        ],
        log_path=output_dir / "02-pilot.log",
        env=environment,
        timeout=7200,
    )
    stages.append({"stage": "agent_pilot", "returncode": pilot.returncode})
    _atomic_json(
        output_dir / "preflight.json",
        _pilot_summary(
            output_dir,
            report_digest=preflight_result["report_digest"],
            agent_mode=preflight_result["agent_mode"],
            stages=stages,
        ),
    )
    return pilot.returncode


if __name__ == "__main__":
    raise SystemExit(main())
