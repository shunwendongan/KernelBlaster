# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict request/result contracts for the privileged Profiler Worker."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DIGEST = re.compile(r"^[0-9a-f]{64}$")
KERNEL_FILTER = re.compile(r"^[A-Za-z0-9_:.<>*]{1,128}$")


class ProfilePlanId(str, Enum):
    NSYS_TIMELINE_V1 = "nsys_timeline_v1"
    NCU_TRIAGE_V1 = "ncu_triage_v1"
    NCU_MEMORY_V1 = "ncu_memory_v1"
    NCU_SCHEDULER_V1 = "ncu_scheduler_v1"


class ProfileStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMED_OUT = "timed_out"


class ProfileReasonCode(str, Enum):
    NONE = "none"
    PERMISSION_DENIED = "permission_denied"
    TOOL_MISSING = "tool_missing"
    UNSUPPORTED = "unsupported"
    METRICS_EMPTY = "metrics_empty"
    KERNEL_NOT_FOUND = "kernel_not_found"
    TIMEOUT = "timeout"
    EXECUTION_FAILED = "execution_failed"
    ARTIFACT_NOT_CORRECT = "artifact_not_correct"
    INTERNAL_ERROR = "internal_error"


class ProfileMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: Literal["ns", "us", "bytes", "percent", "count", "cycles"]


class ProfileRequest(BaseModel):
    """The complete public API: no argv, paths, environment, or output options."""

    model_config = ConfigDict(extra="forbid")

    artifact_digest: str
    plan_id: ProfilePlanId
    kernel_filter: str = Field(min_length=1, max_length=128)
    deadline: datetime

    @field_validator("artifact_digest")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        if not DIGEST.fullmatch(value):
            raise ValueError("artifact_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("kernel_filter")
    @classmethod
    def _safe_filter(cls, value: str) -> str:
        if not KERNEL_FILTER.fullmatch(value):
            raise ValueError("kernel_filter contains unsupported characters")
        return value

    @model_validator(mode="after")
    def _future_deadline(self) -> "ProfileRequest":
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        if self.deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise ValueError("deadline_exceeded")
        return self


class ProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["profiler-summary/v1"] = "profiler-summary/v1"
    plan_id: ProfilePlanId
    kernel_name: str
    metrics: tuple[ProfileMetric, ...]
    diagnostic_only: Literal[True] = True
    ranking_source: Literal["cuda_events"] = "cuda_events"


class ProfileProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_artifact_digest: str
    source_digest: str
    tool: Literal["nsys", "ncu"]
    tool_version: str
    gpu_name: str
    driver_version: str
    image_digest: str | None = None
    workarounds: tuple[str, ...] = ()


class ProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["profiler-result/v1"] = "profiler-result/v1"
    status: ProfileStatus
    reason_code: ProfileReasonCode = ProfileReasonCode.NONE
    plan_id: ProfilePlanId
    summary: ProfileSummary | None = None
    provenance: ProfileProvenance | None = None
    artifact_roles: dict[str, str] = Field(default_factory=dict)


class ProfilerCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["profiler-capabilities/v1"] = "profiler-capabilities/v1"
    platform: Literal["wsl", "linux", "windows", "unsupported"]
    cuda_events: Literal[True] = True
    nsys_status: Literal["available", "tool_missing", "unsupported"]
    ncu_status: Literal["available", "permission_denied", "tool_missing", "unsupported"]
    supported_plans: tuple[ProfilePlanId, ...]
    max_concurrent_profiles: Literal[1] = 1
    automatic_execution: bool


def public_profile_feedback(result: ProfileResult) -> dict[str, object]:
    """Return only structured diagnostics suitable for an LLM prompt."""
    return {
        "status": result.status.value,
        "reason_code": result.reason_code.value,
        "summary": result.summary.model_dump(mode="json") if result.summary else None,
        "provenance": result.provenance.model_dump(mode="json") if result.provenance else None,
    }


__all__ = [
    "ProfileMetric",
    "ProfilePlanId",
    "ProfileProvenance",
    "ProfileReasonCode",
    "ProfileRequest",
    "ProfileResult",
    "ProfileStatus",
    "ProfileSummary",
    "ProfilerCapabilities",
    "public_profile_feedback",
]
