"""Strict public contracts shared by task authors and the trusted Harness."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "fp8_e4m3fn",
    "fp8_e5m2",
    "bf16",
    "fp16",
    "fp32",
    "fp64",
}


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class NumericsClass(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    QUANTIZED = "quantized"


class DeterminismLevel(str, Enum):
    BITWISE = "bitwise"
    BOUNDED = "bounded"


class CacheMode(str, Enum):
    HOT = "hot"
    ROTATING = "rotating"


class CaseTier(str, Enum):
    DEV = "dev"
    FEEDBACK = "feedback"
    AUDIT = "audit"


class AdapterKind(str, Enum):
    DECLARATIVE = "declarative"
    TRUSTED_CODE = "trusted_code"
    LEGACY_DRIVER = "legacy_driver"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TensorSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    dtype: str
    shape: tuple[str | int, ...]
    layout: str = Field(default="contiguous_row_major", min_length=1, max_length=64)
    mutable: bool = False
    differentiable: bool = False
    gradient_of: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    may_alias: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_tensor(self) -> "TensorSpec":
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported tensor dtype: {self.dtype}")
        if any(
            (isinstance(value, str) and not _IDENTIFIER.fullmatch(value))
            or (isinstance(value, int) and (isinstance(value, bool) or value < 0))
            for value in self.shape
        ):
            raise ValueError("tensor shape contains an invalid symbol")
        if self.gradient_of is not None and self.differentiable:
            raise ValueError("a gradient output may not itself be differentiable")
        if self.name in self.may_alias:
            raise ValueError("a tensor may not alias itself")
        return self


class ScalarSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    dtype: Literal["int32", "int64", "float32", "float64", "bool"]
    default: int | float | bool | None = None


class ShapeDimension(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    values: tuple[int, ...] = ()
    multiple_of: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ShapeDimension":
        if self.minimum > self.maximum:
            raise ValueError("shape minimum may not exceed maximum")
        if self.values:
            if len(set(self.values)) != len(self.values):
                raise ValueError("shape values must be unique")
            if any(
                value < self.minimum
                or value > self.maximum
                or value % self.multiple_of != 0
                for value in self.values
            ):
                raise ValueError("shape values must satisfy bounds and multiple_of")
        return self


class WorkloadSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    shape: dict[str, int]
    weight: float = Field(gt=0)
    cache_mode: CacheMode
    core: bool = True


class NumericsPolicy(StrictModel):
    classification: NumericsClass = NumericsClass.EXACT
    reference_dtype: str = "fp32"
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_reference_dtype(self) -> "NumericsPolicy":
        if self.reference_dtype not in _DTYPES:
            raise ValueError("unsupported reference dtype")
        return self


class WorkspacePolicy(StrictModel):
    maximum_bytes: int = Field(default=0, ge=0, le=4 * 1024**3)
    harness_allocated: Literal[True] = True
    timed_initialization: bool = True


class StreamPolicy(StrictModel):
    caller_provided: Literal[True] = True
    maximum_streams: Literal[1] = 1
    graph_capture: Literal["optional", "required", "unsupported"] = "optional"


class TaskSpec(StrictModel):
    schema_version: Literal["harness-task/v1"] = "harness-task/v1"
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    operator: str = Field(min_length=1, max_length=128)
    adapter_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    adapter_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    direction: Direction
    tensors: tuple[TensorSpec, ...]
    scalars: tuple[ScalarSpec, ...] = ()
    dimensions: tuple[ShapeDimension, ...]
    workloads: tuple[WorkloadSpec, ...]
    gradient_targets: tuple[str, ...] = ()
    numerics: NumericsPolicy
    determinism: DeterminismLevel = DeterminismLevel.BITWISE
    workspace: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    stream: StreamPolicy = Field(default_factory=StreamPolicy)
    candidate_backends: tuple[Literal["cuda", "triton"], ...] = ("cuda",)
    disclosure: Literal["adaptive_disclosed"] = "adaptive_disclosed"

    @model_validator(mode="after")
    def validate_task(self) -> "TaskSpec":
        tensor_names = [tensor.name for tensor in self.tensors]
        scalar_names = [scalar.name for scalar in self.scalars]
        dimension_names = [dimension.name for dimension in self.dimensions]
        workload_ids = [workload.id for workload in self.workloads]
        for label, values in (
            ("tensor", tensor_names),
            ("scalar", scalar_names),
            ("dimension", dimension_names),
            ("workload", workload_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} identifier")
        if set(tensor_names) & set(scalar_names):
            raise ValueError("tensor and scalar names must not overlap")
        dimensions = set(dimension_names)
        if any(
            {value for value in tensor.shape if isinstance(value, str)} - dimensions
            for tensor in self.tensors
        ):
            raise ValueError("tensor references an undeclared shape dimension")
        for tensor in self.tensors:
            if set(tensor.may_alias) - set(tensor_names):
                raise ValueError("tensor aliases an unknown tensor")
            if tensor.gradient_of is not None and tensor.gradient_of not in tensor_names:
                raise ValueError("gradient output references an unknown tensor")
        for workload in self.workloads:
            if set(workload.shape) != dimensions:
                raise ValueError("workload shape must bind every declared dimension")
            for dimension in self.dimensions:
                value = workload.shape[dimension.name]
                if (
                    value < dimension.minimum
                    or value > dimension.maximum
                    or value % dimension.multiple_of != 0
                    or (dimension.values and value not in dimension.values)
                ):
                    raise ValueError("workload shape violates dimension constraints")
        if not self.workloads or not any(item.core for item in self.workloads):
            raise ValueError("a task requires at least one core workload")
        differentiable = {
            tensor.name
            for tensor in self.tensors
            if tensor.differentiable and tensor.gradient_of is None
        }
        targets = set(self.gradient_targets)
        if self.direction is Direction.BACKWARD:
            if not differentiable or targets != differentiable:
                raise ValueError("backward must target every differentiable tensor")
            produced = {
                tensor.gradient_of
                for tensor in self.tensors
                if tensor.gradient_of is not None
            }
            if produced != targets:
                raise ValueError("backward must declare one gradient output per target")
        elif targets or any(tensor.gradient_of for tensor in self.tensors):
            raise ValueError("forward tasks may not declare gradient outputs")
        if len(self.candidate_backends) != len(set(self.candidate_backends)):
            raise ValueError("candidate backends must be unique")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class CaseSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    tier: CaseTier
    shape: dict[str, int]
    seed: int
    distribution: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


class CaseBundle(StrictModel):
    schema_version: Literal["harness-case-bundle/v1"] = "harness-case-bundle/v1"
    task_spec_digest: str
    disclosure: Literal["adaptive_disclosed"] = "adaptive_disclosed"
    cases: tuple[CaseSpec, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> "CaseBundle":
        if not _DIGEST.fullmatch(self.task_spec_digest):
            raise ValueError("task_spec_digest must be a lowercase SHA-256")
        identifiers = [case.id for case in self.cases]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("case IDs must be non-empty and unique")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate_for(self, task: TaskSpec) -> None:
        """Verify the bundle binding and every generated shape before execution."""
        if self.task_spec_digest != task.canonical_sha256():
            raise ValueError("case bundle task binding mismatch")
        dimensions = {item.name: item for item in task.dimensions}
        for case in self.cases:
            if set(case.shape) != set(dimensions):
                raise ValueError("case shape must bind every declared dimension")
            for name, value in case.shape.items():
                dimension = dimensions[name]
                if (
                    value < dimension.minimum
                    or value > dimension.maximum
                    or value % dimension.multiple_of != 0
                    or (dimension.values and value not in dimension.values)
                ):
                    raise ValueError("case shape violates dimension constraints")


class TensorError(StrictModel):
    tensor: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    count: int = Field(ge=1)
    mismatch_count: int = Field(ge=0)
    nonfinite_count: int = Field(ge=0)
    max_abs_error: float = Field(ge=0)
    p99_abs_error: float = Field(ge=0)
    max_normalized_error: float = Field(ge=0)
    expected_dtype: str | None = None
    observed_dtype: str | None = None
    expected_shape: tuple[int, ...] = ()
    observed_shape: tuple[int, ...] = ()


class CorrectnessCaseResult(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    tier: CaseTier
    shape: dict[str, int]
    seed: int
    distribution: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    passed: bool
    deterministic: bool
    cuda_error: str | None = Field(default=None, max_length=256)
    violations: tuple[
        Literal[
            "cuda_error",
            "guard_canary",
            "input_mutation",
            "missing_output",
            "nonfinite_output",
            "output_poison",
            "shape_mismatch",
            "dtype_mismatch",
            "nondeterministic",
            "numerical_mismatch",
        ],
        ...,
    ] = ()
    guard_canary_intact: bool = True
    inputs_unmodified: bool = True
    tensors: tuple[TensorError, ...]

    @model_validator(mode="after")
    def validate_outcome(self) -> "CorrectnessCaseResult":
        if not self.tensors:
            raise ValueError("a correctness case requires tensor metrics")
        calculated = (
            self.cuda_error is None
            and not self.violations
            and self.deterministic
            and self.guard_canary_intact
            and self.inputs_unmodified
            and all(
                metric.mismatch_count == 0 and metric.nonfinite_count == 0
                for metric in self.tensors
            )
        )
        if self.passed != calculated:
            raise ValueError("case passed flag contradicts its metrics")
        return self


class CorrectnessResultV2(StrictModel):
    schema_version: Literal["correctness-result/v2"] = "correctness-result/v2"
    protocol_id: Literal["generated-correctness-v2"] = "generated-correctness-v2"
    task_spec_digest: str
    direction: Direction
    passed: bool
    case_count: int = Field(ge=1)
    cases: tuple[CorrectnessCaseResult, ...]

    @model_validator(mode="after")
    def validate_result(self) -> "CorrectnessResultV2":
        if not _DIGEST.fullmatch(self.task_spec_digest):
            raise ValueError("task_spec_digest must be a lowercase SHA-256")
        if self.case_count != len(self.cases):
            raise ValueError("case_count does not match cases")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("correctness cases must be unique")
        calculated = all(case.passed for case in self.cases)
        if self.passed != calculated:
            raise ValueError("result passed flag contradicts case results")
        for case in self.cases:
            for metric in case.tensors:
                if not all(
                    math.isfinite(value)
                    for value in (
                        metric.max_abs_error,
                        metric.p99_abs_error,
                        metric.max_normalized_error,
                    )
                ):
                    raise ValueError("correctness metrics must be finite")
        return self

    def public_feedback(self) -> dict[str, Any]:
        """The agreed adaptive-disclosed policy intentionally returns full cases."""
        return self.model_dump(mode="json")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def parse_correctness_stdout(stdout: bytes | str) -> CorrectnessResultV2:
    """Parse exactly one UTF-8 JSON object and reject log-prefixed false positives."""
    if isinstance(stdout, bytes):
        text = stdout.decode("utf-8", errors="strict")
    else:
        text = stdout
    stripped = text.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise ValueError("correctness stdout must contain exactly one JSON line")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError("correctness stdout is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("correctness stdout must contain a JSON object")
    return CorrectnessResultV2.model_validate(payload)
