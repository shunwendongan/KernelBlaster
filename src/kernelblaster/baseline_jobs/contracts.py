"""Strict contracts for the independent trusted Baseline Worker."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_DIGEST = __import__("re").compile(r"^[0-9a-f]{64}$")


class BaselineProvider(str, Enum):
    UPSTREAM_CUDA = "upstream_cuda"
    PYTORCH_EAGER = "pytorch_eager"
    PYTORCH_COMPILE = "pytorch_compile"
    TRITON = "triton"
    CUBLAS = "cublas"
    CUDNN = "cudnn"
    CUTLASS = "cutlass"


class BaselineStatus(str, Enum):
    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    BLOCKED = "blocked"


class BaselineReasonCode(str, Enum):
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"
    TOOL_MISSING = "tool_missing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CORRECTNESS_FAILED = "correctness_failed"
    EXECUTION_FAILED = "execution_failed"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    HARDWARE_MISMATCH = "hardware_mismatch"
    PROTOCOL_UNSUPPORTED = "protocol_unsupported"
    QUOTA_BLOCKED = "quota_blocked"


class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["baseline-request/v1"] = "baseline-request/v1"
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    task_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    task_spec_digest: str
    case_bundle_digest: str
    evaluation_bundle_digest: str
    provider: BaselineProvider
    hardware_fingerprint: str = Field(min_length=1, max_length=256)
    target_arch: str = Field(pattern=r"^sm_[0-9]{2,3}$")
    protocol_digest: str
    objective: Literal["latency", "throughput"] = "latency"
    deadline: datetime

    @field_validator(
        "task_spec_digest", "case_bundle_digest", "evaluation_bundle_digest", "protocol_digest"
    )
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("baseline artifact fields must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def future_deadline(self) -> "BaselineRequest":
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        if self.deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise ValueError("deadline_exceeded")
        return self

    def cache_key(self, *, image_digest: str) -> str:
        payload = {
            "task_spec_digest": self.task_spec_digest,
            "task_id": self.task_id,
            "case_bundle_digest": self.case_bundle_digest,
            "evaluation_bundle_digest": self.evaluation_bundle_digest,
            "provider": self.provider.value,
            "image_digest": image_digest,
            "hardware_fingerprint": self.hardware_fingerprint,
            "target_arch": self.target_arch,
            "protocol_digest": self.protocol_digest,
            "objective": self.objective,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class BaselineWorkloadMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    workload_id: str
    cache_mode: Literal["hot", "rotating"]
    weight: float = Field(gt=0)
    core: bool
    device_samples_us: tuple[float, ...] = Field(min_length=1)
    host_samples_us: tuple[float, ...] = Field(min_length=1)

    @field_validator("device_samples_us", "host_samples_us")
    @classmethod
    def positive_samples(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value <= 0 for value in values):
            raise ValueError("baseline timing samples must be positive")
        return values


class BaselineProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: BaselineProvider
    provider_version: str
    image_digest: str
    hardware_fingerprint: str
    target_arch: str
    task_spec_digest: str
    case_bundle_digest: str
    protocol_digest: str


class BaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["baseline-result/v1"] = "baseline-result/v1"
    request_id: str
    status: BaselineStatus
    reason_code: BaselineReasonCode = BaselineReasonCode.NONE
    correctness_passed: bool = False
    comparable: bool = False
    cache_key: str
    workloads: tuple[BaselineWorkloadMeasurement, ...] = ()
    provenance: BaselineProvenance
    artifact_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_status(self) -> "BaselineResult":
        if self.comparable != (
            self.status is BaselineStatus.SUCCEEDED
            and self.reason_code is BaselineReasonCode.NONE
            and self.correctness_passed
            and bool(self.workloads)
        ):
            raise ValueError("baseline comparable flag contradicts its result")
        return self


class BaselineCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["baseline-capabilities/v1"] = "baseline-capabilities/v1"
    image_digest: str
    hardware_fingerprint: str
    target_arch: str
    providers: tuple[BaselineProvider, ...]
    max_concurrent_jobs: Literal[1] = 1


__all__ = [
    "BaselineCapabilities",
    "BaselineProvider",
    "BaselineProvenance",
    "BaselineReasonCode",
    "BaselineRequest",
    "BaselineResult",
    "BaselineStatus",
    "BaselineWorkloadMeasurement",
]
