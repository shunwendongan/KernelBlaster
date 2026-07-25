from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.kernelblaster.preflight.backends import build_backend_bundle
from src.kernelblaster.preflight.contracts import (
    AgentCapabilityMode,
    CapabilityCheck,
    CapabilityReasonCode,
    CapabilityReport,
    CapabilityStatus,
    ExecutionBackend,
    PreflightCheckName,
    unavailable_check,
)
from src.kernelblaster.preflight.runner import (
    PreflightConfiguration,
    PreflightRunner,
    capability_hardware_fingerprint,
)
from src.kernelblaster.preflight.provider import build_provider_auth_probe


SOURCE = b"preflight-source-bundle"
SOURCE_DIGEST = hashlib.sha256(SOURCE).hexdigest()
EXECUTABLE_DIGEST = "e" * 64


def _capabilities(*, generated: bool = True, free_bytes: int = 8 * 1024**3):
    return {
        "schema_version": "gpu-capabilities/v1",
        "supervisor_id": "gpu-0",
        "device": {
            "logical_id": "0",
            "name": "Test GPU",
            "compute_capability": "8.6",
            "target_arch": "sm_86",
            "total_memory_bytes": 10 * 1024**3,
            "free_memory_bytes": free_bytes,
        },
        "runtime": {
            "cuda_version": "12.8",
            "driver_version": "test-driver",
            "image_digest": "sha256:test",
        },
        "generated_jobs_enabled": generated,
    }


class FakeControl:
    def __init__(
        self,
        *,
        generated: bool = True,
        free_bytes: int = 8 * 1024**3,
        ncu_result: str = "succeeded",
    ) -> None:
        self.capabilities = _capabilities(generated=generated, free_bytes=free_bytes)
        self.ncu_result = ncu_result
        self.uploads: dict[str, bytes] = {}
        self.gpu_submissions: list[dict] = []
        self.current_stage: dict[str, str] = {}

    async def create_run(self, run_id, metadata):
        return {"id": run_id, "metadata": metadata}

    async def get_run(self, run_id):
        return {"id": run_id}

    async def upload(self, payload, *, media_type, schema=None):
        digest = hashlib.sha256(payload).hexdigest()
        self.uploads[digest] = payload
        return {"digest": digest, "media_type": media_type, "schema": schema}

    async def download(self, digest):
        return self.uploads[digest]

    async def gpu_capabilities(self):
        return self.capabilities

    async def submit_gpu_job(self, manifest):
        self.gpu_submissions.append(manifest)
        self.current_stage[manifest["job_id"]] = manifest["stage"]
        return {"status": "queued"}

    async def wait_gpu_job(self, job_id, *, timeout_seconds):
        stage = self.current_stage[job_id]
        roles = {EXECUTABLE_DIGEST: "executable"} if stage == "compile" else {}
        measurement = (
            {
                "value": 12.5,
                "unit": "us",
                "source": "cuda_events",
                "protocol_id": "trusted-smoke-v1",
            }
            if stage == "events"
            else None
        )
        return {
            "status": "succeeded",
            "result": {
                "reason_code": "none",
                "artifact_roles": roles,
                "measurement": measurement,
            },
        }

    async def profiler_capabilities(self):
        return {
            "supported_plans": [
                "nsys_timeline_v1",
                "ncu_triage_v1",
                "ncu_memory_v1",
                "ncu_scheduler_v1",
            ],
            "nsys_status": "available",
            "ncu_status": "available",
        }

    async def profile(self, request):
        if request["plan_id"] == "ncu_triage_v1" and self.ncu_result != "succeeded":
            return {
                "status": "blocked",
                "reason_code": self.ncu_result,
                "artifact_roles": {},
            }
        workaround = (
            ["nsys_timestamp_retry"]
            if request["plan_id"] == "nsys_timeline_v1"
            else []
        )
        digest = hashlib.sha256(request["plan_id"].encode()).hexdigest()
        return {
            "status": "succeeded",
            "reason_code": "none",
            "summary": {"metrics": [{"name": "gpu_time", "value": 1, "unit": "ns"}]},
            "provenance": {"workarounds": workaround},
            "artifact_roles": {digest: "structured_summary"},
        }


async def _provider():
    return {
        "provider": "openai_compatible",
        "response_model": "unit-model",
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "attempts": 1,
    }


@pytest.mark.asyncio
async def test_provider_probe_has_one_request_and_64_total_tokens(monkeypatch):
    captured = []

    class Provider:
        def __init__(self, settings):
            captured.append(settings)

        async def generate(self, messages, *, model, n):
            return SimpleNamespace(
                response="KERNELBLASTER_OK",
                response_models=[model],
                model=model,
                provider="openai_compatible",
                usage={"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
                attempts=1,
            )

    monkeypatch.setattr("src.kernelblaster.llm.OpenAICompatibleProvider", Provider)
    result = await build_provider_auth_probe(
        base_url="https://provider.invalid/v1",
        api_key="unit-test-key",
        model="unit-model",
    )()

    assert result["attempts"] == 1
    assert len(captured) == 1
    settings = captured[0]
    assert settings.max_requests == 1
    assert settings.max_retries == 0
    assert settings.max_total_tokens == 64
    assert settings.max_completion_tokens == 64


def _checks() -> dict[PreflightCheckName, CapabilityCheck]:
    return {
        name: CapabilityCheck(status=CapabilityStatus.AVAILABLE)
        for name in PreflightCheckName
    }


def _report(**overrides) -> CapabilityReport:
    now = datetime.now(timezone.utc)
    values = {
        "run_id": "run-1",
        "generated_at": now,
        "expires_at": now + timedelta(minutes=15),
        "execution_backend": ExecutionBackend.SANDBOX,
        "agent_mode": AgentCapabilityMode.FULL_DIAGNOSTICS,
        "ranking_backend": "cuda_events",
        "hardware_fingerprint": "sha256:test",
        "supervisor_id": "gpu-0",
        "target_arch": "sm_86",
        "available_diagnostic_plans": ("nsys_timeline_v1", "ncu_triage_v1"),
        "checks": _checks(),
    }
    values.update(overrides)
    return CapabilityReport(**values)


def test_report_requires_all_checks_separates_status_and_reason_and_rejects_secrets():
    report = _report()
    assert set(report.checks) == set(PreflightCheckName)
    with pytest.raises(ValidationError, match="require a reason_code"):
        CapabilityCheck(status=CapabilityStatus.UNAVAILABLE)
    with pytest.raises(ValidationError, match="sensitive field"):
        _report(
            checks={
                **_checks(),
                PreflightCheckName.PROVIDER_AUTH: CapabilityCheck(
                    status=CapabilityStatus.AVAILABLE,
                    observed={"api_key": "must-not-appear"},
                ),
            }
        )


def test_report_digest_expiry_and_hardware_binding_are_fail_closed():
    report = _report()
    report.validate_for_run(digest=report.sha256(), hardware_fingerprint="sha256:test")
    with pytest.raises(ValueError, match="digest_mismatch"):
        report.validate_for_run(digest="0" * 64)
    with pytest.raises(ValueError, match="expired"):
        report.validate_for_run(
            digest=report.sha256(),
            now=report.expires_at + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="hardware_mismatch"):
        report.validate_for_run(
            digest=report.sha256(), hardware_fingerprint="sha256:other"
        )


@pytest.mark.asyncio
async def test_successful_preflight_uses_generated_sandbox_and_real_profile_results():
    control = FakeControl()
    result = await PreflightRunner(control, _provider).run(source_bundle=SOURCE)

    assert result.report_digest == result.report.sha256()
    assert result.report.agent_mode is AgentCapabilityMode.FULL_DIAGNOSTICS
    assert [job["stage"] for job in control.gpu_submissions] == [
        "compile",
        "correctness",
        "events",
    ]
    assert all(job["trusted_bundle_kind"] == "generated_v1" for job in control.gpu_submissions)
    assert all("driver_digest" not in job for job in control.gpu_submissions)
    assert (
        result.report.checks[PreflightCheckName.NSYS_CUDA_TRACE].status
        is CapabilityStatus.AVAILABLE_WITH_WORKAROUND
    )
    assert result.report.ranking_backend == "cuda_events"
    assert set(result.report.available_diagnostic_plans) == {
        "nsys_timeline_v1",
        "ncu_triage_v1",
        "ncu_memory_v1",
        "ncu_scheduler_v1",
    }


@pytest.mark.asyncio
async def test_provider_401_stops_before_every_gpu_job_and_still_returns_complete_report():
    class Unauthorized(RuntimeError):
        status_code = 401

    async def denied():
        raise Unauthorized("invalid key")

    control = FakeControl()
    result = await PreflightRunner(control, denied).run(source_bundle=SOURCE)

    assert control.gpu_submissions == []
    assert result.report.agent_mode is AgentCapabilityMode.UNAVAILABLE
    assert set(result.report.checks) == set(PreflightCheckName)
    assert (
        result.report.checks[PreflightCheckName.PROVIDER_AUTH].reason_code
        is CapabilityReasonCode.AUTH_INVALID
    )


@pytest.mark.asyncio
async def test_permission_denied_ncu_is_events_only_not_candidate_failure():
    control = FakeControl(ncu_result="permission_denied")
    result = await PreflightRunner(control, _provider).run(source_bundle=SOURCE)

    assert result.report.agent_mode is AgentCapabilityMode.EVENTS_ONLY
    assert result.report.ranking_backend == "cuda_events"
    assert (
        result.report.checks[PreflightCheckName.NCU_METRICS].reason_code
        is CapabilityReasonCode.PERMISSION_DENIED
    )
    assert "ncu_triage_v1" not in result.report.available_diagnostic_plans


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control", "failed_check", "reason"),
    [
        (
            FakeControl(generated=False),
            PreflightCheckName.SANDBOX_EXECUTOR,
            CapabilityReasonCode.SANDBOX_UNAVAILABLE,
        ),
        (
            FakeControl(free_bytes=1024),
            PreflightCheckName.FREE_VRAM,
            CapabilityReasonCode.INSUFFICIENT_FREE_VRAM,
        ),
    ],
)
async def test_missing_sandbox_or_vram_fails_before_candidate_execution(
    control, failed_check, reason
):
    result = await PreflightRunner(
        control,
        _provider,
        configuration=PreflightConfiguration(minimum_free_vram_bytes=2 * 1024**3),
    ).run(source_bundle=SOURCE)
    assert result.report.agent_mode is AgentCapabilityMode.UNAVAILABLE
    assert result.report.checks[failed_check].reason_code is reason
    assert control.gpu_submissions == []


@pytest.mark.asyncio
async def test_sqlite_and_artifact_failures_are_distinct_and_submit_no_gpu_jobs():
    class SqliteFailure(FakeControl):
        async def create_run(self, run_id, metadata):
            raise PermissionError("read only")

    sqlite = SqliteFailure()
    sqlite_result = await PreflightRunner(sqlite, _provider).run(source_bundle=SOURCE)
    assert (
        sqlite_result.report.checks[PreflightCheckName.SQLITE_WRITABLE].reason_code
        is CapabilityReasonCode.STORAGE_UNWRITABLE
    )
    assert sqlite.gpu_submissions == []

    class ArtifactMismatch(FakeControl):
        async def upload(self, payload, *, media_type, schema=None):
            uploaded = await super().upload(payload, media_type=media_type, schema=schema)
            if media_type == "application/x-tar":
                uploaded["digest"] = "0" * 64
            return uploaded

    artifact = ArtifactMismatch()
    artifact_result = await PreflightRunner(artifact, _provider).run(source_bundle=SOURCE)
    assert (
        artifact_result.report.checks[PreflightCheckName.ARTIFACT_STORE].reason_code
        is CapabilityReasonCode.ARTIFACT_HASH_MISMATCH
    )
    assert artifact.gpu_submissions == []


@pytest.mark.asyncio
async def test_invalid_architecture_and_empty_nsys_report_are_not_available():
    class InvalidArchitecture(FakeControl):
        async def gpu_capabilities(self):
            payload = _capabilities()
            payload["device"]["target_arch"] = "compute_86"
            return payload

    invalid = InvalidArchitecture()
    invalid_result = await PreflightRunner(invalid, _provider).run(source_bundle=SOURCE)
    assert (
        invalid_result.report.checks[
            PreflightCheckName.COMPILE_TARGET_ARCH
        ].reason_code
        is CapabilityReasonCode.ARCHITECTURE_UNSUPPORTED
    )
    assert (
        invalid_result.report.checks[PreflightCheckName.GPU_VISIBLE].status
        is CapabilityStatus.AVAILABLE
    )
    assert invalid.gpu_submissions == []

    class EmptyNsys(FakeControl):
        async def profile(self, request):
            if request["plan_id"] == "nsys_timeline_v1":
                return {
                    "status": "failed",
                    "reason_code": "metrics_empty",
                    "artifact_roles": {},
                }
            return await super().profile(request)

    empty = EmptyNsys()
    empty_result = await PreflightRunner(empty, _provider).run(source_bundle=SOURCE)
    assert empty_result.report.agent_mode is AgentCapabilityMode.FULL_DIAGNOSTICS
    assert (
        empty_result.report.checks[PreflightCheckName.NSYS_CUDA_TRACE].reason_code
        is CapabilityReasonCode.METRICS_EMPTY
    )
    assert "nsys_timeline_v1" not in empty_result.report.available_diagnostic_plans


def test_backend_factory_never_falls_back_from_sandbox_to_trusted_local():
    report = _report()
    control = object()
    bundle = build_backend_bundle(
        requested=ExecutionBackend.SANDBOX,
        report=report,
        control=control,
    )
    assert bundle.execution_backend is ExecutionBackend.SANDBOX
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        bundle.create_events_backend(
            driver_path=None,
            gpu=None,
            logger=None,
            work_dir=None,
        )

    unavailable = _report(
        agent_mode=AgentCapabilityMode.UNAVAILABLE,
        ranking_backend="none",
        checks={
            **_checks(),
            PreflightCheckName.SANDBOX_EXECUTOR: unavailable_check(
                CapabilityReasonCode.SANDBOX_UNAVAILABLE
            ),
        },
    )
    with pytest.raises(ValueError, match="Agent unavailable"):
        build_backend_bundle(
            requested=ExecutionBackend.SANDBOX,
            report=unavailable,
            control=control,
        )

    local = build_backend_bundle(requested=ExecutionBackend.TRUSTED_LOCAL)
    assert local.execution_backend is ExecutionBackend.TRUSTED_LOCAL


def test_hardware_fingerprint_ignores_volatile_free_memory():
    first = _capabilities(free_bytes=8 * 1024**3)
    second = _capabilities(free_bytes=4 * 1024**3)
    assert capability_hardware_fingerprint(first) == capability_hardware_fingerprint(second)
