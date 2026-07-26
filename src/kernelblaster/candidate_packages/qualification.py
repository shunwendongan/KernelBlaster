"""Fixed compute-sanitizer publication gate, independent of ranking state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
from typing import Literal


class SanitizerPlan(str, Enum):
    MEMCHECK = "memcheck"
    INITCHECK = "initcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"


@dataclass(frozen=True)
class SanitizerResult:
    plan: SanitizerPlan
    status: Literal["succeeded", "failed", "unavailable", "timed_out"]
    reason_code: str
    log: bytes = b""


def sanitizer_command(
    plan: SanitizerPlan,
    *,
    replay_executable: Path,
    capsule_path: Path,
) -> list[str]:
    return [
        "compute-sanitizer",
        "--tool",
        plan.value,
        "--error-exitcode",
        "86",
        "--destroy-on-device-error",
        "kernel",
        str(replay_executable),
        "--capsule",
        str(capsule_path),
        "--mode",
        "correctness",
    ]


def run_sanitizer(
    plan: SanitizerPlan,
    *,
    replay_executable: Path,
    capsule_path: Path,
    root: Path,
    timeout_seconds: int = 300,
) -> SanitizerResult:
    try:
        result = subprocess.run(
            sanitizer_command(
                plan, replay_executable=replay_executable, capsule_path=capsule_path
            ),
            cwd=root,
            env={
                "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(root),
                "TMPDIR": str(root),
                "CUDA_VISIBLE_DEVICES": "0",
            },
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            start_new_session=True,
        )
    except FileNotFoundError:
        return SanitizerResult(plan, "unavailable", "tool_missing")
    except subprocess.TimeoutExpired:
        return SanitizerResult(plan, "timed_out", "sanitizer_timeout")
    log = (result.stdout + b"\n" + result.stderr)[:8 * 1024 * 1024]
    return SanitizerResult(
        plan,
        "succeeded" if result.returncode == 0 else "failed",
        "none" if result.returncode == 0 else "sanitizer_failed",
        log,
    )


@dataclass(frozen=True)
class CudaWinnerQualification:
    backend: Literal["cuda", "triton"]
    correctness_passed: bool
    events_gate_passed: bool
    sanitizer_results: tuple[SanitizerResult, ...]
    nsys_status: str = "not_run"
    ncu_status: str = "not_run"

    @property
    def qualified(self) -> bool:
        plans = {result.plan for result in self.sanitizer_results if result.status == "succeeded"}
        return bool(
            self.backend == "cuda"
            and self.correctness_passed
            and self.events_gate_passed
            and plans == set(SanitizerPlan)
        )


__all__ = [
    "CudaWinnerQualification",
    "SanitizerPlan",
    "SanitizerResult",
    "run_sanitizer",
    "sanitizer_command",
]
