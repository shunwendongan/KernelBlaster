# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered runtime preflight using existing Control service APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
import time
from typing import Any, Awaitable, Callable
import uuid

from .client import ControlApiError, ControlPlaneClient
from .contracts import (
    AgentCapabilityMode,
    CapabilityCheck,
    CapabilityReasonCode,
    CapabilityReport,
    CapabilityStatus,
    ExecutionBackend,
    PreflightCheckName,
    unavailable_check,
)
from ..portability.contracts import HardwareIdentity


ProviderAuthProbe = Callable[[], Awaitable[dict[str, Any]]]
CheckObserver = Callable[[PreflightCheckName, CapabilityCheck], None]

HARD_CHECKS = {
    PreflightCheckName.PROVIDER_AUTH,
    PreflightCheckName.GPU_VISIBLE,
    PreflightCheckName.COMPILE_TARGET_ARCH,
    PreflightCheckName.KERNEL_LAUNCH,
    PreflightCheckName.CUDA_EVENTS,
    PreflightCheckName.FREE_VRAM,
    PreflightCheckName.ARTIFACT_STORE,
    PreflightCheckName.SQLITE_WRITABLE,
    PreflightCheckName.SANDBOX_EXECUTOR,
}


@dataclass(frozen=True)
class PreflightConfiguration:
    private_evaluation_profile_id: str = "preflight-vector-add-v1"
    benchmark_protocol_id: str = "trusted-smoke-v1"
    kernel_filter: str = "vector_add_kernel"
    minimum_free_vram_bytes: int = 2 * 1024**3
    report_ttl_seconds: int = 15 * 60
    job_timeout_seconds: int = 10 * 60


@dataclass(frozen=True)
class PreflightResult:
    report: CapabilityReport
    report_digest: str | None


def capability_hardware_fingerprint(capabilities: dict[str, Any]) -> str:
    """Return the portable comparison group, not a supervisor-local identity."""
    device = dict(capabilities.get("device") or {})
    runtime = dict(capabilities.get("runtime") or {})
    identity = HardwareIdentity(
        logical_id=str(device.get("logical_id") or "unknown"),
        name=str(device.get("name") or "unknown"),
        compute_capability=str(device.get("compute_capability") or "unknown"),
        target_arch=str(device.get("target_arch") or "unknown"),
        total_memory_bytes=device.get("total_memory_bytes"),
        driver_version=runtime.get("driver_version"),
        cuda_version=runtime.get("cuda_version"),
        image_digest=runtime.get("image_digest"),
        gpu_uuid=device.get("uuid"),
        runtime=runtime.get("runtime_id"),
    )
    return identity.comparison_group


def _artifact_roles(payload: dict[str, Any]) -> dict[str, str]:
    result = payload.get("result") or {}
    roles = result.get("artifact_roles") or {}
    return {str(digest): str(role) for digest, role in roles.items()}


def _artifact_for_role(payload: dict[str, Any], role: str) -> str:
    for digest, actual_role in _artifact_roles(payload).items():
        if actual_role == role:
            return digest
    raise ValueError(f"GPU Job did not produce artifact role {role}")


def _reason_from_job(payload: dict[str, Any]) -> CapabilityReasonCode:
    reason = str((payload.get("result") or {}).get("reason_code") or "")
    return {
        "stage_timeout": CapabilityReasonCode.PROBE_TIMEOUT,
        "deadline_exceeded": CapabilityReasonCode.PROBE_TIMEOUT,
        "sandbox_unavailable": CapabilityReasonCode.SANDBOX_UNAVAILABLE,
        "sandbox_violation": CapabilityReasonCode.PROBE_CRASH,
        "gpu_recovery_failed": CapabilityReasonCode.DRIVER_INCOMPATIBLE,
        "gpu_oom": CapabilityReasonCode.EXECUTION_FAILED,
        "compile_failed": CapabilityReasonCode.EXECUTION_FAILED,
        "correctness_failed": CapabilityReasonCode.EXECUTION_FAILED,
        "events_failed": CapabilityReasonCode.METRICS_EMPTY,
    }.get(reason, CapabilityReasonCode.EXECUTION_FAILED)


def _reason_from_profile(payload: dict[str, Any]) -> CapabilityReasonCode:
    return {
        "permission_denied": CapabilityReasonCode.PERMISSION_DENIED,
        "tool_missing": CapabilityReasonCode.TOOL_MISSING,
        "unsupported": CapabilityReasonCode.DRIVER_INCOMPATIBLE,
        "metrics_empty": CapabilityReasonCode.METRICS_EMPTY,
        "kernel_not_found": CapabilityReasonCode.METRICS_EMPTY,
        "timeout": CapabilityReasonCode.PROBE_TIMEOUT,
        "execution_failed": CapabilityReasonCode.PROBE_CRASH,
        "internal_error": CapabilityReasonCode.INTERNAL_ERROR,
    }.get(str(payload.get("reason_code")), CapabilityReasonCode.PROBE_CRASH)


def _safe_provider_observation(payload: dict[str, Any]) -> dict[str, Any]:
    usage = dict(payload.get("usage") or {})
    return {
        "provider": str(payload.get("provider") or "openai_compatible"),
        "response_model": str(payload.get("response_model") or ""),
        "usage": {
            key: int(usage.get(key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "attempts": int(payload.get("attempts", 1) or 1),
    }


class PreflightRunner:
    def __init__(
        self,
        control: ControlPlaneClient,
        provider_auth: ProviderAuthProbe,
        *,
        configuration: PreflightConfiguration | None = None,
        observer: CheckObserver | None = None,
    ) -> None:
        self.control = control
        self.provider_auth = provider_auth
        self.configuration = configuration or PreflightConfiguration()
        self.observer = observer

    def _record(
        self,
        checks: dict[PreflightCheckName, CapabilityCheck],
        name: PreflightCheckName,
        check: CapabilityCheck,
    ) -> None:
        checks[name] = check
        if self.observer is not None:
            self.observer(name, check)

    async def _finish(
        self,
        *,
        run_id: str,
        checks: dict[PreflightCheckName, CapabilityCheck],
        capabilities: dict[str, Any] | None,
        diagnostic_plans: tuple[str, ...],
        artifact_roles: dict[str, str],
    ) -> PreflightResult:
        for name in PreflightCheckName:
            checks.setdefault(name, unavailable_check())
        hard_available = all(
            checks[name].status is not CapabilityStatus.UNAVAILABLE
            for name in HARD_CHECKS
        )
        if not hard_available:
            agent_mode = AgentCapabilityMode.UNAVAILABLE
            ranking_backend = "none"
        elif checks[PreflightCheckName.NCU_METRICS].status is CapabilityStatus.UNAVAILABLE:
            agent_mode = AgentCapabilityMode.EVENTS_ONLY
            ranking_backend = "cuda_events"
        else:
            agent_mode = AgentCapabilityMode.FULL_DIAGNOSTICS
            ranking_backend = "cuda_events"
        generated_at = datetime.now(timezone.utc)
        capabilities = capabilities or {}
        device = dict(capabilities.get("device") or {})
        reported_arch = str(device.get("target_arch") or "")
        target_arch = reported_arch if re.fullmatch(r"sm_[0-9]{2,3}", reported_arch) else None
        report = CapabilityReport(
            run_id=run_id,
            generated_at=generated_at,
            expires_at=generated_at
            + timedelta(seconds=self.configuration.report_ttl_seconds),
            execution_backend=ExecutionBackend.SANDBOX,
            agent_mode=agent_mode,
            ranking_backend=ranking_backend,
            hardware_fingerprint=(
                capability_hardware_fingerprint(capabilities)
                if capabilities
                else "unavailable"
            ),
            supervisor_id=(str(capabilities.get("supervisor_id")) if capabilities else None),
            target_arch=target_arch,
            available_diagnostic_plans=diagnostic_plans,
            checks=checks,
            artifact_roles=artifact_roles,
            budget={
                "provider_auth_requests": 1,
                "provider_auth_max_total_tokens": 64,
                "max_completion_tokens": 64,
            },
        )
        payload = report.canonical_bytes()
        try:
            uploaded = await self.control.upload(
                payload,
                media_type="application/json",
                schema="capability-report/v1",
            )
            digest = str(uploaded.get("digest") or "")
            if digest != report.sha256():
                return PreflightResult(report=report, report_digest=None)
            return PreflightResult(report=report, report_digest=digest)
        except Exception:
            return PreflightResult(report=report, report_digest=None)

    async def run(self, *, source_bundle: bytes) -> PreflightResult:
        run_id = "preflight-" + uuid.uuid4().hex
        checks: dict[PreflightCheckName, CapabilityCheck] = {}
        artifact_roles: dict[str, str] = {}
        capabilities: dict[str, Any] | None = None
        diagnostic_plans: tuple[str, ...] = ()

        started = time.monotonic()
        try:
            provider = await self.provider_auth()
            self._record(
                checks,
                PreflightCheckName.PROVIDER_AUTH,
                CapabilityCheck(
                    status=CapabilityStatus.AVAILABLE,
                    duration_ms=(time.monotonic() - started) * 1000,
                    observed=_safe_provider_observation(provider),
                ),
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            reason = (
                CapabilityReasonCode.AUTH_INVALID
                if status_code in {401, 403}
                else CapabilityReasonCode.PROVIDER_UNAVAILABLE
            )
            self._record(
                checks,
                PreflightCheckName.PROVIDER_AUTH,
                CapabilityCheck(
                    status=CapabilityStatus.UNAVAILABLE,
                    reason_code=reason,
                    duration_ms=(time.monotonic() - started) * 1000,
                    observed={"error_type": type(error).__name__, "status_code": status_code},
                ),
            )
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=None,
                diagnostic_plans=(),
                artifact_roles={},
            )

        started = time.monotonic()
        try:
            created = await self.control.create_run(
                run_id,
                {"kind": "capability-preflight", "schema_version": "capability-report/v1"},
            )
            loaded = await self.control.get_run(run_id)
            if created.get("id") != run_id or loaded.get("id") != run_id:
                raise ValueError("preflight run did not round-trip")
            sqlite_check = CapabilityCheck(
                status=CapabilityStatus.AVAILABLE,
                duration_ms=(time.monotonic() - started) * 1000,
                observed={"run_id_sha256": hashlib.sha256(run_id.encode()).hexdigest()},
            )
        except Exception as error:
            sqlite_check = CapabilityCheck(
                status=CapabilityStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.STORAGE_UNWRITABLE,
                duration_ms=(time.monotonic() - started) * 1000,
                observed={"error_type": type(error).__name__},
            )
        self._record(checks, PreflightCheckName.SQLITE_WRITABLE, sqlite_check)
        if sqlite_check.status is CapabilityStatus.UNAVAILABLE:
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=None,
                diagnostic_plans=(),
                artifact_roles={},
            )

        started = time.monotonic()
        try:
            source = await self.control.upload(
                source_bundle,
                media_type="application/x-tar",
                schema="gpu-source-bundle/v1",
            )
            source_digest = str(source.get("digest") or "")
            expected = hashlib.sha256(source_bundle).hexdigest()
            if source_digest != expected:
                raise ValueError("artifact hash mismatch")
            downloaded = await self.control.download(source_digest)
            if hashlib.sha256(downloaded).hexdigest() != expected:
                raise ValueError("artifact hash mismatch")
            artifact_roles[source_digest] = "preflight_source_bundle"
            artifact_check = CapabilityCheck(
                status=CapabilityStatus.AVAILABLE,
                duration_ms=(time.monotonic() - started) * 1000,
                observed={"round_trip_bytes": len(downloaded)},
                artifact_roles={source_digest: "preflight_source_bundle"},
            )
        except Exception as error:
            artifact_check = CapabilityCheck(
                status=CapabilityStatus.UNAVAILABLE,
                reason_code=(
                    CapabilityReasonCode.ARTIFACT_HASH_MISMATCH
                    if isinstance(error, ValueError)
                    else CapabilityReasonCode.STORAGE_UNWRITABLE
                ),
                duration_ms=(time.monotonic() - started) * 1000,
                observed={"error_type": type(error).__name__},
            )
            source_digest = ""
        self._record(checks, PreflightCheckName.ARTIFACT_STORE, artifact_check)
        if artifact_check.status is CapabilityStatus.UNAVAILABLE:
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=None,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        started = time.monotonic()
        try:
            capabilities = await self.control.gpu_capabilities()
            device = dict(capabilities.get("device") or {})
            if not device or not str(device.get("name") or "").strip():
                raise RuntimeError("GPU device is unavailable")
            target_arch = str(device["target_arch"])
            gpu_check = CapabilityCheck(
                status=CapabilityStatus.AVAILABLE,
                duration_ms=(time.monotonic() - started) * 1000,
                observed={
                    "device_name": str(device.get("name") or "unknown"),
                    "compute_capability": str(device.get("compute_capability") or "unknown"),
                    "target_arch": target_arch,
                },
            )
        except Exception as error:
            target_arch = ""
            gpu_check = CapabilityCheck(
                status=CapabilityStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.DRIVER_INCOMPATIBLE,
                duration_ms=(time.monotonic() - started) * 1000,
                observed={"error_type": type(error).__name__},
            )
        self._record(checks, PreflightCheckName.GPU_VISIBLE, gpu_check)
        if gpu_check.status is CapabilityStatus.UNAVAILABLE:
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=capabilities,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        if not re.fullmatch(r"sm_[0-9]{2,3}", target_arch):
            self._record(
                checks,
                PreflightCheckName.COMPILE_TARGET_ARCH,
                CapabilityCheck(
                    status=CapabilityStatus.UNAVAILABLE,
                    reason_code=CapabilityReasonCode.ARCHITECTURE_UNSUPPORTED,
                    observed={"target_arch": target_arch},
                ),
            )
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=capabilities,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        device = dict((capabilities or {}).get("device") or {})
        free_bytes = device.get("free_memory_bytes")
        if isinstance(free_bytes, int) and free_bytes >= self.configuration.minimum_free_vram_bytes:
            free_check = CapabilityCheck(
                status=CapabilityStatus.AVAILABLE,
                observed={
                    "free_memory_bytes": free_bytes,
                    "minimum_free_memory_bytes": self.configuration.minimum_free_vram_bytes,
                },
            )
        else:
            free_check = CapabilityCheck(
                status=CapabilityStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.INSUFFICIENT_FREE_VRAM,
                observed={
                    "free_memory_bytes": free_bytes,
                    "minimum_free_memory_bytes": self.configuration.minimum_free_vram_bytes,
                },
            )
        self._record(checks, PreflightCheckName.FREE_VRAM, free_check)
        if free_check.status is CapabilityStatus.UNAVAILABLE:
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=capabilities,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        generated_enabled = bool((capabilities or {}).get("generated_jobs_enabled"))
        if not generated_enabled:
            self._record(
                checks,
                PreflightCheckName.SANDBOX_EXECUTOR,
                unavailable_check(CapabilityReasonCode.SANDBOX_UNAVAILABLE),
            )
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=capabilities,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        common = {
            "schema_version": "gpu-job/v1",
            "run_id": run_id,
            "source_bundle_digest": source_digest,
            "private_evaluation_profile_id": self.configuration.private_evaluation_profile_id,
            "target_arch": target_arch,
            "benchmark_protocol_id": self.configuration.benchmark_protocol_id,
            "deadline": (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.configuration.job_timeout_seconds)
            ).isoformat(),
            "trusted_bundle_kind": "generated_v1",
        }
        stage_specs = (
            ("compile", PreflightCheckName.COMPILE_TARGET_ARCH, 180),
            ("correctness", PreflightCheckName.KERNEL_LAUNCH, 60),
            ("events", PreflightCheckName.CUDA_EVENTS, 90),
        )
        executable_digest: str | None = None
        sandbox_ok = True
        for stage, check_name, wall_seconds in stage_specs:
            started = time.monotonic()
            completed: dict[str, Any] | None = None
            job_id = uuid.uuid4().hex
            manifest = {
                **common,
                "job_id": job_id,
                "idempotency_key": f"preflight:{stage}",
                "stage": stage,
                "resource_limits": {"wall_seconds": wall_seconds},
            }
            if executable_digest is not None:
                manifest["executable_digest"] = executable_digest
            try:
                await self.control.submit_gpu_job(manifest)
                completed = await self.control.wait_gpu_job(
                    job_id,
                    timeout_seconds=self.configuration.job_timeout_seconds,
                )
                roles = _artifact_roles(completed)
                artifact_roles.update(roles)
                if str(completed.get("status")) != "succeeded":
                    raise RuntimeError(json.dumps(completed, sort_keys=True))
                observed: dict[str, Any] = {"stage": stage, "status": "succeeded"}
                if stage == "compile":
                    executable_digest = _artifact_for_role(completed, "executable")
                    observed["executable_digest"] = executable_digest
                elif stage == "events":
                    measurement = dict((completed.get("result") or {}).get("measurement") or {})
                    value = measurement.get("value")
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError("Events measurement is empty")
                    if (
                        measurement.get("source") != "cuda_events"
                        or measurement.get("unit") != "us"
                        or measurement.get("protocol_id")
                        != self.configuration.benchmark_protocol_id
                    ):
                        raise ValueError("Events measurement contract mismatch")
                    observed["measurement"] = measurement
                self._record(
                    checks,
                    check_name,
                    CapabilityCheck(
                        status=CapabilityStatus.AVAILABLE,
                        duration_ms=(time.monotonic() - started) * 1000,
                        observed=observed,
                        artifact_roles=roles,
                    ),
                )
            except Exception as error:
                sandbox_ok = False
                reason = (
                    _reason_from_job(completed)
                    if completed is not None
                    else (
                        CapabilityReasonCode.PROBE_TIMEOUT
                        if isinstance(error, TimeoutError)
                        else CapabilityReasonCode.SANDBOX_UNAVAILABLE
                    )
                )
                self._record(
                    checks,
                    check_name,
                    CapabilityCheck(
                        status=CapabilityStatus.UNAVAILABLE,
                        reason_code=reason,
                        duration_ms=(time.monotonic() - started) * 1000,
                        observed={"stage": stage, "error_type": type(error).__name__},
                    ),
                )
                break

        self._record(
            checks,
            PreflightCheckName.SANDBOX_EXECUTOR,
            CapabilityCheck(
                status=(
                    CapabilityStatus.AVAILABLE if sandbox_ok else CapabilityStatus.UNAVAILABLE
                ),
                reason_code=(
                    CapabilityReasonCode.NONE
                    if sandbox_ok
                    else CapabilityReasonCode.SANDBOX_UNAVAILABLE
                ),
                observed={"ephemeral_stage_count": 3 if sandbox_ok else 0},
            ),
        )
        if not sandbox_ok or executable_digest is None:
            return await self._finish(
                run_id=run_id,
                checks=checks,
                capabilities=capabilities,
                diagnostic_plans=(),
                artifact_roles=artifact_roles,
            )

        profiler_capabilities: dict[str, Any] = {}
        try:
            profiler_capabilities = await self.control.profiler_capabilities()
        except Exception:
            pass
        available_plans: list[str] = []
        for plan_id, check_name in (
            ("nsys_timeline_v1", PreflightCheckName.NSYS_CUDA_TRACE),
            ("ncu_triage_v1", PreflightCheckName.NCU_METRICS),
        ):
            supported = set(profiler_capabilities.get("supported_plans") or ())
            if plan_id not in supported:
                status_key = "nsys_status" if plan_id.startswith("nsys") else "ncu_status"
                status = str(profiler_capabilities.get(status_key) or "unsupported")
                reason = {
                    "permission_denied": CapabilityReasonCode.PERMISSION_DENIED,
                    "tool_missing": CapabilityReasonCode.TOOL_MISSING,
                }.get(status, CapabilityReasonCode.DRIVER_INCOMPATIBLE)
                self._record(checks, check_name, unavailable_check(reason))
                continue
            started = time.monotonic()
            profiled: dict[str, Any] | None = None
            try:
                profiled = await self.control.profile(
                    {
                        "artifact_digest": executable_digest,
                        "plan_id": plan_id,
                        "kernel_filter": self.configuration.kernel_filter,
                        "deadline": (
                            datetime.now(timezone.utc)
                            + timedelta(seconds=self.configuration.job_timeout_seconds)
                        ).isoformat(),
                    }
                )
                roles = {
                    str(digest): str(role)
                    for digest, role in (profiled.get("artifact_roles") or {}).items()
                }
                artifact_roles.update(roles)
                if str(profiled.get("status")) != "succeeded" or not profiled.get("summary"):
                    raise RuntimeError(json.dumps(profiled, sort_keys=True))
                provenance = dict(profiled.get("provenance") or {})
                workarounds = tuple(provenance.get("workarounds") or ())
                check_status = (
                    CapabilityStatus.AVAILABLE_WITH_WORKAROUND
                    if workarounds
                    else CapabilityStatus.AVAILABLE
                )
                self._record(
                    checks,
                    check_name,
                    CapabilityCheck(
                        status=check_status,
                        duration_ms=(time.monotonic() - started) * 1000,
                        observed={
                            "plan_id": plan_id,
                            "metric_count": len((profiled.get("summary") or {}).get("metrics") or ()),
                            "workarounds": list(workarounds),
                        },
                        artifact_roles=roles,
                    ),
                )
                available_plans.append(plan_id)
            except Exception as error:
                reason = (
                    _reason_from_profile(profiled)
                    if profiled is not None
                    else CapabilityReasonCode.PROBE_CRASH
                )
                self._record(
                    checks,
                    check_name,
                    CapabilityCheck(
                        status=CapabilityStatus.UNAVAILABLE,
                        reason_code=reason,
                        duration_ms=(time.monotonic() - started) * 1000,
                        observed={"plan_id": plan_id, "error_type": type(error).__name__},
                    ),
                )
        if checks.get(PreflightCheckName.NCU_METRICS, unavailable_check()).status is not CapabilityStatus.UNAVAILABLE:
            for plan in profiler_capabilities.get("supported_plans") or ():
                if str(plan).startswith("ncu_") and str(plan) not in available_plans:
                    available_plans.append(str(plan))
        diagnostic_plans = tuple(available_plans)
        return await self._finish(
            run_id=run_id,
            checks=checks,
            capabilities=capabilities,
            diagnostic_plans=diagnostic_plans,
            artifact_roles=artifact_roles,
        )


__all__ = [
    "HARD_CHECKS",
    "PreflightConfiguration",
    "PreflightResult",
    "PreflightRunner",
    "capability_hardware_fingerprint",
]
