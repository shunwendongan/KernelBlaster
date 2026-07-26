from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from src.kernelblaster.evaluation import (
    CandidateEvaluation,
    CandidateStageStatus,
    CompilationMetrics,
    FunnelPolicy,
    SandboxCandidateEvaluator,
    TrustedLocalCandidateEvaluator,
    candidate_kernel_names,
    parse_ptxas_metrics,
    safe_kernel_base_name,
)
from src.kernelblaster.measurements import (
    Measurement,
    MeasurementSource,
    MeasurementUnit,
)
from src.kernelblaster.outcomes import DiagnosticStatus
from src.kernelblaster.profiling import PerformanceGateResult, ProfilingMode, ProfilingResult


def _measurement(value: float, samples: tuple[float, ...]) -> Measurement:
    return Measurement(
        value=value,
        unit=MeasurementUnit.MICROSECONDS,
        source=MeasurementSource.CUDA_EVENTS,
        samples=samples,
        protocol_id="events-v1",
        hardware_fingerprint="gpu-a",
    )


def _candidate(
    digest: str,
    value: float,
    *,
    samples: tuple[float, ...] | None = None,
) -> CandidateEvaluation:
    samples = samples or (value, value, value)
    return CandidateEvaluation(
        candidate_id=digest,
        source_digest=digest,
        executable_digest=(digest[::-1]),
        execution_status=CandidateStageStatus.SUCCEEDED,
        correctness_status=CandidateStageStatus.SUCCEEDED,
        events_status=CandidateStageStatus.SUCCEEDED,
        measurement=_measurement(value, samples),
        discovery_samples_us=samples,
        compilation_metrics=CompilationMetrics(kernel_names=("candidate_kernel",)),
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0), (1, 1), (10, 1), (11, 2), (20, 2), (21, 3), (100, 3)],
)
def test_confirmation_count_boundaries(count, expected):
    assert FunnelPolicy.confirmation_count(count) == expected


def test_funnel_deduplicates_and_excludes_incomparable_measurements():
    fastest = _candidate("a" * 64, 8.0)
    duplicate = _candidate("a" * 64, 1.0)
    slower = _candidate("b" * 64, 9.0)
    incomparable = _candidate("c" * 64, 7.0)
    incomparable.measurement = Measurement(
        value=7,
        unit=MeasurementUnit.CYCLES,
        source=MeasurementSource.NCU,
        protocol_id="ncu-v1",
        hardware_fingerprint="gpu-a",
    )

    decision = FunnelPolicy().decide([fastest, duplicate, slower, incomparable])
    assert decision.eligible_source_digests == ("a" * 64, "b" * 64)
    assert decision.confirmation_source_digests == ("a" * 64,)


def test_funnel_selects_at_most_one_highest_variance_anomaly():
    stable = _candidate("a" * 64, 8.0, samples=(8.0, 8.1, 7.9))
    anomaly = _candidate("b" * 64, 9.0, samples=(8.0, 10.0, 9.0))
    larger = _candidate("c" * 64, 10.0, samples=(8.0, 12.0, 10.0))
    decision = FunnelPolicy().decide([stable, anomaly, larger])
    assert decision.anomaly_source_digest == "c" * 64


def test_ncu_top_two_threshold_is_inclusive_and_capped():
    first = _candidate("a" * 64, 8.0)
    second = _candidate("b" * 64, 8.1)
    third = _candidate("c" * 64, 8.2)
    first.confirmation_samples_us = (10.0,) * 5
    second.confirmation_samples_us = (10.1,) * 5
    third.confirmation_samples_us = (10.1001,) * 5
    targets = FunnelPolicy().ncu_targets([third, second, first])
    assert [item.source_digest for item in targets] == ["a" * 64, "b" * 64]


def test_ptxas_parser_keeps_only_candidate_metrics_and_no_private_content():
    source = """
    extern "C" __global__ void candidate_kernel(float *out) { out[0] = 1; }
    """
    assert candidate_kernel_names(source) == ("candidate_kernel",)
    raw = """
    /input/private/driver.cpp: private-seed-marker
    ptxas info    : Compiling entry function '_Z16candidate_kernelPf' for 'sm_86'
    ptxas info    : Function properties for _Z16candidate_kernelPf
        16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
    ptxas info    : Used 32 registers, 256 bytes smem, 64 bytes cmem[0], 16 bytes cmem[2]
    ptxas info    : Compiling entry function '_Z19private_seed_kernelPi' for 'sm_86'
    ptxas info    : Function properties for _Z19private_seed_kernelPi
        1024 bytes stack frame, 512 bytes spill stores, 256 bytes spill loads
    ptxas info    : Used 128 registers, 4096 bytes smem, 2048 bytes cmem[0]
    """
    metrics = parse_ptxas_metrics(raw, candidate_kernels=("candidate_kernel",))
    assert metrics.to_dict() == {
        "registers": 32,
        "spill_load_bytes": 4,
        "spill_store_bytes": 8,
        "stack_frame_bytes": 16,
        "shared_memory_bytes": 256,
        "constant_memory_bytes": 80,
        "kernel_names": ["candidate_kernel"],
    }
    evaluation = _candidate("a" * 64, 8.0)
    evaluation.compilation_metrics = metrics
    feedback = str(evaluation.public_feedback())
    assert "private" not in feedback
    assert "seed" not in feedback
    assert "/input" not in feedback


def test_safe_kernel_base_name_rejects_commands_and_extracts_signature():
    assert safe_kernel_base_name("candidate_kernel(float *, int)") == "candidate_kernel"
    assert safe_kernel_base_name("void ns::candidate_kernel<float>(float *)") == (
        "candidate_kernel"
    )
    assert safe_kernel_base_name("--import /etc/passwd") is None


@pytest.mark.asyncio
async def test_diagnostic_failure_does_not_change_candidate_events_or_correctness():
    baseline = _candidate("0" * 64, 10.0)
    candidate = _candidate("1" * 64, 9.0, samples=(8.9, 9.0, 9.1))

    class Evaluator(SandboxCandidateEvaluator):
        async def _event_value(self, evaluation, *, phase, index):
            return 10.0 if evaluation.source_digest == baseline.source_digest else 9.0

        async def _profile(self, evaluation, *, plan_id, kernel_filter):
            assert kernel_filter == "candidate_kernel"
            if plan_id == "nsys_timeline_v1":
                return {
                    "status": "succeeded",
                    "reason_code": "none",
                    "summary": {
                        "kernel_name": "candidate_kernel(float *)",
                        "metrics": [{"name": "gpu_time", "value": 9, "unit": "us"}],
                    },
                }
            return {
                "status": "blocked",
                "reason_code": "permission_denied",
                "summary": None,
            }

    evaluator = Evaluator(
        None,
        SimpleNamespace(target_arch="sm_86", hardware_fingerprint="gpu-a"),
        run_id="run-1",
        private_evaluation_profile_id="private-v1",
        benchmark_protocol_id="events-v1",
    )
    result = await evaluator.finalize(baseline, [candidate])

    assert result.winner is candidate
    assert result.performance_gate is not None and result.performance_gate.passed
    assert result.nsys_calls == 1
    assert result.ncu_calls == 1
    assert candidate.correctness_status is CandidateStageStatus.SUCCEEDED
    assert candidate.events_status is CandidateStageStatus.SUCCEEDED
    assert candidate.diagnostic_status is DiagnosticStatus.PARTIAL
    assert candidate.diagnostics["ncu_triage_v1"]["reason_code"] == "permission_denied"


@pytest.mark.asyncio
async def test_profiler_transport_crash_is_diagnostic_only():
    baseline = _candidate("0" * 64, 10.0)
    candidate = _candidate("1" * 64, 9.0)

    class Control:
        async def profile(self, request):
            raise ConnectionError("profiler unavailable")

    class Evaluator(SandboxCandidateEvaluator):
        async def _event_value(self, evaluation, *, phase, index):
            return 10.0 if evaluation is baseline else 9.0

    evaluator = Evaluator(
        Control(),
        SimpleNamespace(target_arch="sm_86", hardware_fingerprint="gpu-a"),
        run_id="run-1",
        private_evaluation_profile_id="private-v1",
        benchmark_protocol_id="events-v1",
    )
    result = await evaluator.finalize(baseline, [candidate])
    assert result.performance_gate is not None and result.performance_gate.passed
    assert candidate.correctness_status is CandidateStageStatus.SUCCEEDED
    assert candidate.events_status is CandidateStageStatus.SUCCEEDED
    assert candidate.diagnostic_status is DiagnosticStatus.UNAVAILABLE
    assert candidate.diagnostics["nsys_timeline_v1"]["reason_code"] == "internal_error"


@pytest.mark.asyncio
async def test_sandbox_evaluator_deduplicates_and_keeps_raw_compile_log_out_of_feedback(
    tmp_path,
):
    compile_log = b"""
    /input/private/driver.cpp private-seed-marker
    ptxas info : Compiling entry function '_Z16candidate_kernelPf' for 'sm_86'
    ptxas info : Used 24 registers, 32 bytes smem, 8 bytes cmem[0]
    """
    compile_digest = hashlib.sha256(compile_log).hexdigest()
    executable_digest = "e" * 64

    class Control:
        def __init__(self):
            self.jobs = {}
            self.submissions = []

        async def create_run(self, run_id, metadata):
            return {"id": run_id, "metadata": metadata}

        async def get_run(self, run_id):
            return {"id": run_id}

        async def upload(self, payload, *, media_type, schema=None):
            return {"digest": hashlib.sha256(payload).hexdigest()}

        async def download(self, digest):
            assert digest == compile_digest
            return compile_log

        async def submit_gpu_job(self, manifest):
            self.jobs[manifest["job_id"]] = manifest
            self.submissions.append(manifest)
            return {"status": "queued"}

        async def wait_gpu_job(self, job_id, *, timeout_seconds):
            manifest = self.jobs[job_id]
            stage = manifest["stage"]
            roles = {}
            measurement = None
            if stage == "compile":
                roles = {
                    executable_digest: "executable",
                    compile_digest: "compile_log",
                }
            elif stage == "events":
                measurement = {
                    "value": 9.0,
                    "unit": "us",
                    "source": "cuda_events",
                    "protocol_id": "events-v1",
                }
            return {
                "status": "succeeded",
                "result": {
                    "reason_code": "none",
                    "artifact_roles": roles,
                    "measurement": measurement,
                },
            }

    source_path = tmp_path / "candidate.cu"
    source_path.write_text(
        'extern "C" __global__ void candidate_kernel(float *out) { out[0] = 1; }',
        encoding="utf-8",
    )
    control = Control()
    evaluator = SandboxCandidateEvaluator(
        control,
        SimpleNamespace(target_arch="sm_86", hardware_fingerprint="gpu-a"),
        run_id="run-1",
        private_evaluation_profile_id="private-v1",
        benchmark_protocol_id="events-v1",
    )
    first = await evaluator.evaluate(source_path)
    second = await evaluator.evaluate(source_path)

    assert first is second and first.rankable
    assert [job["stage"] for job in control.submissions] == [
        "compile",
        "correctness",
        "events",
        "events",
        "events",
    ]
    assert first.compilation_metrics.registers == 24
    feedback = json.dumps(first.public_feedback())
    assert "private" not in feedback
    assert "seed" not in feedback


@pytest.mark.asyncio
async def test_trusted_local_is_explicit_and_still_runs_paired_confirmation(tmp_path):
    baseline_path = tmp_path / "baseline.cu"
    candidate_path = tmp_path / "candidate.cu"
    baseline_path.write_text("baseline", encoding="utf-8")
    candidate_path.write_text("candidate", encoding="utf-8")

    class Backend:
        confirmation_sessions = 5

        async def profile(self, filepath):
            value = 10.0 if filepath == baseline_path else 9.0
            return ProfilingResult(
                mode=ProfilingMode.EVENTS_ONLY,
                measurement=_measurement(value, (value, value, value)),
            )

        async def confirm_pair(self, baseline, candidate):
            assert baseline == baseline_path and candidate == candidate_path
            return PerformanceGateResult(
                passed=True,
                median_speedup=10 / 9,
                bootstrap_95_lower=10 / 9,
                bootstrap_95_upper=10 / 9,
            )

    evaluator = TrustedLocalCandidateEvaluator(Backend())
    baseline = await evaluator.evaluate(baseline_path)
    candidate = await evaluator.evaluate(candidate_path)
    result = await evaluator.finalize(baseline, [candidate])
    assert result.winner is candidate
    assert result.performance_gate is not None and result.performance_gate.passed
    assert result.confirmation_calls == 10
