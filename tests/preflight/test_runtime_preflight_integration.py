from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.kernelblaster.gpu_jobs import build_deterministic_bundle
from src.kernelblaster.preflight.client import ControlPlaneClient
from src.kernelblaster.preflight.contracts import AgentCapabilityMode, PreflightCheckName
from src.kernelblaster.preflight.runner import PreflightRunner


pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.getenv("KERNELBLASTER_RUN_PREFLIGHT_INTEGRATION") != "1",
    reason="set KERNELBLASTER_RUN_PREFLIGHT_INTEGRATION=1 for live Control/GPU smoke",
)
@pytest.mark.asyncio
async def test_real_control_sandbox_and_profiler_preflight():
    token = os.environ["KERNELBLASTER_CONTROL_TOKEN"]
    control_url = os.getenv("KERNELBLASTER_CONTROL_URL", "http://127.0.0.1:8000")

    async def bounded_test_provider():
        # This integration target exercises Control/GPU/Profiler. Provider HTTP
        # authentication remains covered separately and is never bypassed by
        # the production CLI.
        return {
            "provider": "integration_fixture",
            "response_model": "none",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "attempts": 1,
        }

    source = (
        ROOT / "portfolio" / "trusted_gpu_smoke" / "vector_add.cu"
    ).read_bytes()
    result = await PreflightRunner(
        ControlPlaneClient(control_url, token),
        bounded_test_provider,
    ).run(source_bundle=build_deterministic_bundle({"vector_add.cu": source}))

    assert result.report_digest == result.report.sha256()
    assert result.report.agent_mode in {
        AgentCapabilityMode.FULL_DIAGNOSTICS,
        AgentCapabilityMode.EVENTS_ONLY,
    }
    assert result.report.checks[PreflightCheckName.SANDBOX_EXECUTOR].status.value == (
        "available"
    )
    assert result.report.checks[PreflightCheckName.CUDA_EVENTS].observed["measurement"][
        "source"
    ] == "cuda_events"
