# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, public runtime capability contracts used to gate Agent runs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "driver_path",
    "password",
    "secret",
    "seed",
    "token",
)
SAFE_USAGE_KEYS = {
    "cached_tokens",
    "completion_tokens",
    "max_completion_tokens",
    "prompt_tokens",
    "provider_auth_max_total_tokens",
    "reasoning_tokens",
    "total_tokens",
}
SENSITIVE_VALUE = re.compile(r"(?i)\b(?:bearer|authorization)\s+[^\s]+")


class PreflightCheckName(str, Enum):
    PROVIDER_AUTH = "provider_auth"
    GPU_VISIBLE = "gpu_visible"
    COMPILE_TARGET_ARCH = "compile_target_arch"
    KERNEL_LAUNCH = "kernel_launch"
    CUDA_EVENTS = "cuda_events"
    NSYS_CUDA_TRACE = "nsys_cuda_trace"
    NCU_METRICS = "ncu_metrics"
    FREE_VRAM = "free_vram"
    ARTIFACT_STORE = "artifact_store"
    SQLITE_WRITABLE = "sqlite_writable"
    SANDBOX_EXECUTOR = "sandbox_executor"


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    AVAILABLE_WITH_WORKAROUND = "available_with_workaround"
    UNAVAILABLE = "unavailable"


class CapabilityReasonCode(str, Enum):
    NONE = "none"
    AUTH_INVALID = "auth_invalid"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TOOL_MISSING = "tool_missing"
    PERMISSION_DENIED = "permission_denied"
    DRIVER_INCOMPATIBLE = "driver_incompatible"
    ARCHITECTURE_UNSUPPORTED = "architecture_unsupported"
    METRICS_EMPTY = "metrics_empty"
    PROBE_TIMEOUT = "probe_timeout"
    PROBE_CRASH = "probe_crash"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    STORAGE_UNWRITABLE = "storage_unwritable"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    INSUFFICIENT_FREE_VRAM = "insufficient_free_vram"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EXECUTION_FAILED = "execution_failed"
    INTERNAL_ERROR = "internal_error"


class AgentCapabilityMode(str, Enum):
    FULL_DIAGNOSTICS = "full_diagnostics"
    EVENTS_ONLY = "events_only"
    UNAVAILABLE = "unavailable"


class ExecutionBackend(str, Enum):
    SANDBOX = "sandbox"
    TRUSTED_LOCAL = "trusted_local"


class CapabilityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CapabilityStatus
    reason_code: CapabilityReasonCode = CapabilityReasonCode.NONE
    duration_ms: float = Field(default=0.0, ge=0)
    observed: dict[str, Any] = Field(default_factory=dict)
    artifact_roles: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifact_roles")
    @classmethod
    def _valid_artifact_digests(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not DIGEST_PATTERN.fullmatch(digest) for digest in value):
            raise ValueError("artifact roles must be keyed by lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def _status_matches_reason(self) -> "CapabilityCheck":
        if self.status is CapabilityStatus.UNAVAILABLE:
            if self.reason_code is CapabilityReasonCode.NONE:
                raise ValueError("unavailable checks require a reason_code")
        elif self.reason_code is not CapabilityReasonCode.NONE:
            raise ValueError("available checks may not carry a failure reason_code")
        _assert_public(self.observed)
        return self


class CapabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["capability-report/v1"] = "capability-report/v1"
    run_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime
    expires_at: datetime
    execution_backend: ExecutionBackend
    agent_mode: AgentCapabilityMode
    ranking_backend: Literal["cuda_events", "none"]
    hardware_fingerprint: str = Field(min_length=1, max_length=128)
    supervisor_id: str | None = Field(default=None, max_length=128)
    target_arch: str | None = Field(default=None, pattern=r"^sm_[0-9]{2,3}$")
    available_diagnostic_plans: tuple[
        Literal[
            "nsys_timeline_v1",
            "ncu_triage_v1",
            "ncu_memory_v1",
            "ncu_scheduler_v1",
        ],
        ...,
    ] = ()
    checks: dict[PreflightCheckName, CapabilityCheck]
    artifact_roles: dict[str, str] = Field(default_factory=dict)
    budget: dict[str, int] = Field(default_factory=dict)

    @field_validator("artifact_roles")
    @classmethod
    def _valid_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not DIGEST_PATTERN.fullmatch(digest) for digest in value):
            raise ValueError("artifact roles must be keyed by lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def _complete_and_consistent(self) -> "CapabilityReport":
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be later than generated_at")
        if set(self.checks) != set(PreflightCheckName):
            raise ValueError("capability report must contain every fixed preflight check")
        if self.agent_mode is AgentCapabilityMode.UNAVAILABLE:
            if self.ranking_backend != "none":
                raise ValueError("unavailable Agent mode cannot select a ranking backend")
        elif self.ranking_backend != "cuda_events":
            raise ValueError("available Agent modes must rank with CUDA Events")
        _assert_public(self.model_dump(mode="json"))
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate_for_run(
        self,
        *,
        digest: str,
        now: datetime | None = None,
        hardware_fingerprint: str | None = None,
    ) -> None:
        if not DIGEST_PATTERN.fullmatch(digest) or self.sha256() != digest:
            raise ValueError("capability_report_digest_mismatch")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("capability_report_validation_time_must_be_aware")
        if current > self.expires_at:
            raise ValueError("capability_report_expired")
        if hardware_fingerprint and hardware_fingerprint != self.hardware_fingerprint:
            raise ValueError("capability_report_hardware_mismatch")


def _assert_public(value: Any, key: str = "") -> None:
    lowered = key.lower()
    if lowered not in SAFE_USAGE_KEYS and any(
        part in lowered for part in SENSITIVE_KEY_PARTS
    ):
        raise ValueError(f"capability reports may not contain sensitive field {key!r}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _assert_public(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public(child, key)
    elif isinstance(value, str) and SENSITIVE_VALUE.search(value):
        raise ValueError("capability reports may not contain authorization values")


def unavailable_check(
    reason_code: CapabilityReasonCode = CapabilityReasonCode.DEPENDENCY_UNAVAILABLE,
) -> CapabilityCheck:
    return CapabilityCheck(
        status=CapabilityStatus.UNAVAILABLE,
        reason_code=reason_code,
    )


__all__ = [
    "AgentCapabilityMode",
    "CapabilityCheck",
    "CapabilityReasonCode",
    "CapabilityReport",
    "CapabilityStatus",
    "ExecutionBackend",
    "PreflightCheckName",
    "unavailable_check",
]
