from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.kernelblaster.evaluation import SandboxCandidateEvaluator
from src.kernelblaster.preflight.client import ControlPlaneClient
from src.kernelblaster.preflight.runner import capability_hardware_fingerprint


pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.getenv("KERNELBLASTER_RUN_FUNNEL_INTEGRATION") != "1",
    reason="set KERNELBLASTER_RUN_FUNNEL_INTEGRATION=1 for the live Agent funnel",
)
@pytest.mark.asyncio
async def test_real_sandbox_events_confirmation_nsys_and_ncu_funnel(tmp_path):
    control = ControlPlaneClient(
        os.getenv("KERNELBLASTER_CONTROL_URL", "http://127.0.0.1:8000"),
        os.environ["KERNELBLASTER_CONTROL_TOKEN"],
    )
    capabilities = await control.gpu_capabilities()
    report = SimpleNamespace(
        target_arch=capabilities["device"]["target_arch"],
        hardware_fingerprint=capability_hardware_fingerprint(capabilities),
    )
    source = (ROOT / "portfolio" / "trusted_gpu_smoke" / "vector_add.cu").read_text(
        encoding="utf-8"
    )
    baseline_path = tmp_path / "baseline.cu"
    candidate_path = tmp_path / "candidate.cu"
    baseline_path.write_text(source, encoding="utf-8")
    candidate_path.write_text(source + "\n// funnel candidate variant\n", encoding="utf-8")

    evaluator = SandboxCandidateEvaluator(
        control,
        report,
        run_id=f"funnel-integration-{uuid.uuid4().hex[:12]}",
        private_evaluation_profile_id="preflight-vector-add-v1",
        benchmark_protocol_id="trusted-smoke-v1",
    )
    baseline = await evaluator.evaluate(baseline_path)
    candidate = await evaluator.evaluate(candidate_path)
    result = await evaluator.finalize(baseline, [candidate])

    assert baseline.rankable and candidate.rankable
    assert len(baseline.discovery_samples_us) == 3
    assert len(candidate.discovery_samples_us) == 3
    assert candidate.compilation_metrics.kernel_names == ("vector_add_kernel",)
    assert result.confirmation_calls == 10
    assert result.nsys_calls == 1
    assert result.ncu_calls == 1
    assert result.winner is candidate
    if os.getenv("KERNELBLASTER_EXPECT_NCU") == "1":
        assert candidate.diagnostics["ncu_triage_v1"]["status"] == "succeeded"
