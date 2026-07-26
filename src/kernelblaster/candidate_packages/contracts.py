"""CandidatePackage v2 and controlled launch-plan contracts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_FUNCTIONS = {"ceil_div", "min", "max"}


class CandidateBackend(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _validate_expression(source: str, symbols: set[str], *, predicate: bool = False) -> None:
    if len(source) > 256:
        raise ValueError("launch expression is too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError("launch expression syntax is invalid") from error
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.UAdd,
        ast.USub,
        ast.Call,
    )
    if predicate:
        allowed += (
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("launch expression uses a forbidden operation")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, int)
        ):
            raise ValueError("launch expressions accept integer constants only")
        if isinstance(node, ast.Name) and node.id not in symbols | _FUNCTIONS:
            raise ValueError("launch expression references an undeclared symbol")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or node.keywords
        ):
            raise ValueError("launch expression calls a forbidden function")


def _evaluate(source: str, bindings: Mapping[str, int | float]) -> int | bool:
    functions = {
        "ceil_div": lambda left, right: (left + right - 1) // right,
        "min": min,
        "max": max,
    }
    value = eval(  # noqa: S307 - AST is strictly allowlisted above
        compile(ast.parse(source, mode="eval"), "<launch-plan>", "eval"),
        {"__builtins__": {}},
        {**functions, **bindings},
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    raise ValueError("launch expression did not produce an integer or boolean")


class Dimensions(StrictModel):
    x: str
    y: str = "1"
    z: str = "1"


class KernelDeclaration(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    parameters: tuple[str, ...]


class KernelLaunch(StrictModel):
    kernel: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    grid: Dimensions
    block: Dimensions
    dynamic_shared_bytes: str = "0"
    arguments: tuple[str, ...]


class DispatchRule(StrictModel):
    when: str | None = None
    launches: tuple[KernelLaunch, ...] = Field(min_length=1, max_length=16)


class CandidateLaunchPlan(StrictModel):
    schema_version: Literal["candidate-launch-plan/v1"] = "candidate-launch-plan/v1"
    task_spec_digest: str
    backend: CandidateBackend
    shape_symbols: tuple[str, ...]
    tensor_bindings: tuple[str, ...]
    scalar_bindings: tuple[str, ...] = ()
    workspace_bytes: int = Field(default=0, ge=0, le=4 * 1024**3)
    stream: Literal["harness_provided"] = "harness_provided"
    cuda_graph: Literal["disabled", "capture_if_requested", "required"] = "disabled"
    kernels: tuple[KernelDeclaration, ...] = Field(min_length=1, max_length=32)
    dispatch: tuple[DispatchRule, ...] = Field(min_length=1, max_length=32)

    @field_validator("task_spec_digest")
    @classmethod
    def digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("TaskSpec digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> "CandidateLaunchPlan":
        for label, values in (
            ("shape", self.shape_symbols),
            ("tensor", self.tensor_bindings),
            ("scalar", self.scalar_bindings),
            ("kernel", tuple(item.name for item in self.kernels)),
        ):
            if (label != "scalar" and not values) or len(values) != len(set(values)) or any(
                not _NAME.fullmatch(value) for value in values
            ):
                raise ValueError(f"launch plan {label} identifiers must be safe and unique")
        if self.dispatch[-1].when is not None or any(
            rule.when is None for rule in self.dispatch[:-1]
        ):
            raise ValueError("launch dispatch requires exactly one final default rule")
        symbols = set(self.shape_symbols) | set(self.scalar_bindings)
        kernels = {item.name: item for item in self.kernels}
        bindings = (
            set(self.tensor_bindings)
            | set(self.scalar_bindings)
            | set(self.shape_symbols)
            | {"workspace"}
        )
        for rule in self.dispatch:
            if rule.when is not None:
                _validate_expression(rule.when, symbols, predicate=True)
            for launch in rule.launches:
                if launch.kernel not in kernels:
                    raise ValueError("launch references an undeclared kernel")
                for expression in (
                    launch.grid.x,
                    launch.grid.y,
                    launch.grid.z,
                    launch.block.x,
                    launch.block.y,
                    launch.block.z,
                    launch.dynamic_shared_bytes,
                ):
                    _validate_expression(expression, symbols)
                if set(launch.arguments) - bindings:
                    raise ValueError("launch arguments reference an undeclared binding")
                if tuple(launch.arguments) != kernels[launch.kernel].parameters:
                    raise ValueError("launch arguments do not match the kernel declaration")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def select(self, bindings: Mapping[str, int | float]) -> tuple["ResolvedLaunch", ...]:
        if set(bindings) != set(self.shape_symbols) | set(self.scalar_bindings):
            raise ValueError("launch bindings do not match the declared symbols")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in bindings.values()
        ):
            raise ValueError("launch bindings must be finite non-negative numbers")
        selected = next(
            rule
            for rule in self.dispatch
            if rule.when is None or _evaluate(rule.when, bindings) is True
        )
        resolved: list[ResolvedLaunch] = []
        for launch in selected.launches:
            grid = tuple(int(_evaluate(value, bindings)) for value in launch.grid.model_dump().values())
            block = tuple(int(_evaluate(value, bindings)) for value in launch.block.model_dump().values())
            shared = int(_evaluate(launch.dynamic_shared_bytes, bindings))
            if any(value <= 0 or value > 2**31 - 1 for value in grid):
                raise ValueError("resolved grid is outside the bounded launch domain")
            if any(value <= 0 for value in block) or math.prod(block) > 1024:
                raise ValueError("resolved block exceeds CUDA's thread bound")
            if shared < 0 or shared > 96 * 1024:
                raise ValueError("resolved dynamic shared memory exceeds the fixed bound")
            resolved.append(
                ResolvedLaunch(
                    kernel=launch.kernel,
                    grid=grid,
                    block=block,
                    dynamic_shared_bytes=shared,
                    arguments=launch.arguments,
                )
            )
        return tuple(resolved)


@dataclass(frozen=True)
class ResolvedLaunch:
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_bytes: int
    arguments: tuple[str, ...]


class CandidateProvenance(StrictModel):
    generator: str = Field(min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=128)
    parent_candidate_digest: str | None = None

    @field_validator("parent_candidate_digest")
    @classmethod
    def parent_digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.fullmatch(value):
            raise ValueError("parent digest must be lowercase SHA-256")
        return value


class CandidateManifestV2(StrictModel):
    schema_version: Literal["candidate-package/v2"] = "candidate-package/v2"
    task_spec_digest: str
    backend: CandidateBackend
    source_path: Literal["candidate.cu", "candidate.py"]
    source_digest: str
    launch_plan_digest: str
    provenance: CandidateProvenance

    @field_validator("task_spec_digest", "source_digest", "launch_plan_digest")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("candidate manifest digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def extension_matches_backend(self) -> "CandidateManifestV2":
        expected = "candidate.cu" if self.backend is CandidateBackend.CUDA else "candidate.py"
        if self.source_path != expected:
            raise ValueError("candidate source extension does not match its backend")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


class CandidateCapsuleManifest(StrictModel):
    schema_version: Literal["candidate-capsule/v1"] = "candidate-capsule/v1"
    candidate_package_digest: str
    task_spec_digest: str
    launch_plan_digest: str
    module_digest: str
    backend: CandidateBackend
    target_arch: str = Field(pattern=r"^sm_[0-9]{2,3}$")
    compiler_id: str = Field(min_length=1, max_length=128)

    @field_validator(
        "candidate_package_digest", "task_spec_digest", "launch_plan_digest", "module_digest"
    )
    @classmethod
    def valid_capsule_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("capsule digests must be lowercase SHA-256")
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


class CandidateProfilerCapsuleManifest(StrictModel):
    """Binding for the trusted replay artifact released after correctness."""

    schema_version: Literal["candidate-profiler-capsule/v1"] = (
        "candidate-profiler-capsule/v1"
    )
    candidate_capsule_digest: str
    task_spec_digest: str
    case_bundle_digest: str

    @field_validator(
        "candidate_capsule_digest", "task_spec_digest", "case_bundle_digest"
    )
    @classmethod
    def valid_profiler_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("profiler capsule digests must be lowercase SHA-256")
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


__all__ = [
    "CandidateBackend",
    "CandidateCapsuleManifest",
    "CandidateProfilerCapsuleManifest",
    "CandidateLaunchPlan",
    "CandidateManifestV2",
    "CandidateProvenance",
    "Dimensions",
    "DispatchRule",
    "KernelDeclaration",
    "KernelLaunch",
    "ResolvedLaunch",
]
