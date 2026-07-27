"""Configuration-driven E2E release orchestration over existing scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any

from ..portability.profile import load_profile


ROOT = Path(__file__).resolve().parents[3]
CAPABILITY_DIGEST_PLACEHOLDER = "${CAPABILITY_REPORT_DIGEST}"


@dataclass(frozen=True)
class ReleaseProfile:
    mode: str = "local"
    model: str | None = None
    gpu_label: str = "auto"
    target_arch: str | None = None
    task_id: str = "036"
    suite: str = "portfolio/suites/rmsnorm.json"
    rollouts: int = 2
    steps: int = 2
    max_requests: int = 32
    max_total_tokens: int = 250_000
    max_completion_tokens: int = 64
    provider_timeout_seconds: int = 180
    agent_timeout_seconds: int = 7_200
    top_k: int = 3
    state_dir: str | None = None
    control_url: str = "http://127.0.0.1:8000"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_release_profile(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None) -> ReleaseProfile:
    values = load_profile(path, overrides=overrides)
    def integer(name: str, fallback: int) -> int:
        return int(values.get(name, fallback))
    profile = ReleaseProfile(
        mode=str(values.get("mode", "local")),
        model=(str(values["model"]) if values.get("model") else os.getenv("MODEL") or None),
        gpu_label=str(values.get("gpu", values.get("gpu_label", "auto"))),
        target_arch=(str(values["target_arch"]) if values.get("target_arch") else None),
        task_id=str(values.get("task_id", "036")),
        suite=str(values.get("suite", "portfolio/suites/rmsnorm.json")),
        rollouts=integer("rollouts", 2),
        steps=integer("steps", 2),
        max_requests=integer("max_requests", 32),
        max_total_tokens=integer("max_total_tokens", 250_000),
        max_completion_tokens=integer("max_completion_tokens", 64),
        provider_timeout_seconds=integer("provider_timeout_seconds", 180),
        agent_timeout_seconds=integer("agent_timeout_seconds", 7_200),
        top_k=integer("top_k", 3),
        state_dir=(str(values["state_dir"]) if values.get("state_dir") else None),
        control_url=str(values.get("control_url", "http://127.0.0.1:8000")),
    )
    if profile.mode not in {"local", "autodl"}:
        raise ValueError("release mode must be local or autodl")
    for name in (
        "rollouts",
        "steps",
        "max_requests",
        "max_total_tokens",
        "max_completion_tokens",
        "provider_timeout_seconds",
        "agent_timeout_seconds",
        "top_k",
    ):
        if getattr(profile, name) < 1:
            raise ValueError(f"release profile {name} must be positive")
    if profile.max_total_tokens < profile.max_completion_tokens:
        raise ValueError("max_total_tokens must be at least max_completion_tokens")
    return profile


def build_e2e_plan(profile: ReleaseProfile, output_dir: str | Path) -> list[dict[str, Any]]:
    """Return an auditable ordered command plan without starting infrastructure."""
    output = Path(output_dir)
    common_state = ["--state-dir", profile.state_dir] if profile.state_dir else []
    agent_environment = {
        "LLM_MAX_REQUESTS": str(profile.max_requests),
        "LLM_MAX_TOTAL_TOKENS": str(profile.max_total_tokens),
        "LLM_MAX_COMPLETION_TOKENS": str(profile.max_completion_tokens),
        "LLM_REQUEST_TIMEOUT_SECONDS": str(profile.provider_timeout_seconds),
    }
    if profile.target_arch:
        agent_environment["KERNELBLASTER_TARGET_ARCH"] = profile.target_arch
    preflight = [
        sys.executable,
        "scripts/run_preflight.py",
        "--gpu", profile.gpu_label,
        "--control-url", profile.control_url,
        "--provider-timeout-seconds", str(profile.provider_timeout_seconds),
        "--output-dir", str(output / "01-preflight"),
    ]
    if profile.model:
        preflight.extend(["--model", profile.model])
    smoke = [
        sys.executable,
        "scripts/smoke_llm.py",
        "--max-completion-tokens", str(profile.max_completion_tokens),
        "--max-total-tokens", str(profile.max_total_tokens),
        "--timeout-seconds", str(profile.provider_timeout_seconds),
        "--output-dir", str(output / "02-provider-smoke"),
    ]
    agent = [
        sys.executable,
        "scripts/run_RL.py",
        "--experiment-name", "release-rmsnorm",
        "--dataset", "kernelbench-cuda",
        "--precision", "fp16",
        "--cuda", "--cuda-perf", "--use-rl",
        "--rl-iterations", str(profile.rollouts),
        "--rl-rollout-steps", str(profile.steps),
        "--timeout", str(math.ceil(profile.agent_timeout_seconds / 60)),
        "--problem-numbers", profile.task_id,
        "--portfolio-suite", profile.suite,
        "--gpu", profile.gpu_label,
        "--execution-backend", "sandbox",
        "--capability-report-digest", CAPABILITY_DIGEST_PLACEHOLDER,
        "--run-record-dir", str(output / "03-agent"),
        *common_state,
    ]
    if profile.model:
        agent.extend(["--model", profile.model])
    return [
        {"name": "capability_preflight", "command": preflight, "timeout_seconds": profile.provider_timeout_seconds + 600},
        {"name": "provider_smoke", "command": smoke, "timeout_seconds": profile.provider_timeout_seconds + 60},
        {
            "name": "rmsnorm_agent",
            "command": agent,
            "environment": agent_environment,
            "timeout_seconds": profile.agent_timeout_seconds,
        },
        {
            "name": "confirmation_and_nsys",
            "command": ["profile-driven", "top-k", str(profile.top_k), "events", "nsys"],
            "timeout_seconds": profile.agent_timeout_seconds,
            "notes": "The existing Agent funnel owns candidate selection; NCU remains optional and Events-only is valid.",
        },
    ]
