"""Operator-neutral trusted correctness runtime.

Candidate launchers are injected by an allowlisted Adapter.  They never get to
construct the verdict; this module checks structure, mutation, stability and
numerics and emits the sole correctness-result/v2 document.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    CaseBundle,
    CorrectnessCaseResult,
    CorrectnessResultV2,
    DeterminismLevel,
    TaskSpec,
    TensorError,
)
from .reference import concrete_shape, torch_dtype


@dataclass(frozen=True)
class CandidateRun:
    outputs: Mapping[str, Any]
    guard_canary_intact: bool = True
    output_poison_overwritten: bool = True


@dataclass(frozen=True)
class HarnessContext:
    device: str
    stream: Any | None
    workspace: Any


CandidateCallable = Callable[
    [Mapping[str, Any], Mapping[str, int | float | bool], HarnessContext],
    Mapping[str, Any] | CandidateRun,
]


class ReferenceAdapter(Protocol):
    def create_inputs(self, task: TaskSpec, case: Any, *, device: str) -> dict[str, Any]: ...

    def scalar_values(self, task: TaskSpec) -> dict[str, int | float | bool]: ...

    def reference(
        self,
        task: TaskSpec,
        inputs: Mapping[str, Any],
        scalars: Mapping[str, int | float | bool],
    ) -> dict[str, Any]: ...


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional runtime
        raise RuntimeError("the correctness runtime requires the gpu extra") from error
    return torch


def _clone(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in values.items()}


def _synchronize(device: str) -> None:
    torch = _torch()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _as_run(value: Mapping[str, Any] | CandidateRun) -> CandidateRun:
    return value if isinstance(value, CandidateRun) else CandidateRun(outputs=value)


class CorrectnessHarness:
    """Create deterministic cases and own every byte of the resulting verdict."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device

    def evaluate(
        self,
        task: TaskSpec,
        bundle: CaseBundle,
        *,
        adapter: ReferenceAdapter,
        candidate: CandidateCallable,
    ) -> CorrectnessResultV2:
        bundle.validate_for(task)
        cases = tuple(
            self._evaluate_case(task, case, adapter=adapter, candidate=candidate)
            for case in bundle.cases
        )
        return CorrectnessResultV2(
            task_spec_digest=task.canonical_sha256(),
            direction=task.direction,
            passed=all(case.passed for case in cases),
            case_count=len(cases),
            cases=cases,
        )

    def _evaluate_case(
        self,
        task: TaskSpec,
        case: Any,
        *,
        adapter: ReferenceAdapter,
        candidate: CandidateCallable,
    ) -> CorrectnessCaseResult:
        torch = _torch()
        original = adapter.create_inputs(task, case, device=self.device)
        scalars = adapter.scalar_values(task)
        reference = adapter.reference(task, _clone(original), scalars)
        expected_specs = {
            item.name: item
            for item in task.tensors
            if item.mutable and (task.direction.value == "forward" or item.gradient_of is not None)
        }
        violations: set[str] = set()
        cuda_error: str | None = None
        run = CandidateRun(outputs={})
        repeat = CandidateRun(outputs={})
        first_inputs = _clone(original)
        second_inputs = _clone(original)
        stream = torch.cuda.Stream(device=self.device) if self.device.startswith("cuda") else None
        workspace = torch.empty(
            task.workspace.maximum_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        context = HarnessContext(device=self.device, stream=stream, workspace=workspace)
        try:
            if stream is None:
                run = _as_run(candidate(first_inputs, scalars, context))
                repeat = _as_run(candidate(second_inputs, scalars, context))
            else:
                with torch.cuda.stream(stream):
                    workspace.fill_(0xA5)
                    run = _as_run(candidate(first_inputs, scalars, context))
                stream.synchronize()
                with torch.cuda.stream(stream):
                    workspace.fill_(0xA5)
                    repeat = _as_run(candidate(second_inputs, scalars, context))
                stream.synchronize()
        except Exception as error:  # trusted Harness converts crashes into evidence
            cuda_error = f"{type(error).__name__}: {error}"[:256]
            violations.add("cuda_error")
        inputs_unmodified = all(torch.equal(original[name], first_inputs[name]) for name in original)
        if not inputs_unmodified:
            violations.add("input_mutation")
        guard_intact = run.guard_canary_intact and repeat.guard_canary_intact
        if not guard_intact:
            violations.add("guard_canary")
        if not run.output_poison_overwritten or not repeat.output_poison_overwritten:
            violations.add("output_poison")

        tensor_metrics: list[TensorError] = []
        deterministic = cuda_error is None
        for name, spec in expected_specs.items():
            expected = reference.get(name)
            observed = run.outputs.get(name)
            observed_repeat = repeat.outputs.get(name)
            expected_shape = concrete_shape(spec.shape, case.shape)
            expected_dtype = torch_dtype(spec.dtype)
            if expected is None or observed is None or observed_repeat is None:
                violations.add("missing_output")
                count = math.prod(expected_shape)
                tensor_metrics.append(
                    TensorError(
                        tensor=name,
                        count=count,
                        mismatch_count=count,
                        nonfinite_count=0,
                        max_abs_error=0,
                        p99_abs_error=0,
                        max_normalized_error=0,
                        expected_dtype=str(expected_dtype),
                        expected_shape=expected_shape,
                    )
                )
                deterministic = False
                continue
            observed_shape = tuple(observed.shape)
            if observed_shape != expected_shape:
                violations.add("shape_mismatch")
            if observed.dtype != expected_dtype:
                violations.add("dtype_mismatch")
            if tuple(observed_repeat.shape) != observed_shape or observed_repeat.dtype != observed.dtype:
                deterministic = False
            elif task.determinism is DeterminismLevel.BITWISE:
                deterministic = deterministic and torch.equal(observed, observed_repeat)
            else:
                deterministic = deterministic and torch.allclose(
                    observed.float(),
                    observed_repeat.float(),
                    atol=task.numerics.atol,
                    rtol=task.numerics.rtol,
                )
            if observed_shape != expected_shape:
                count = math.prod(expected_shape)
                mismatch = count
                nonfinite = 0
                maximum = percentile = normalized = 0.0
            else:
                observed_float = observed.float()
                expected_float = expected.float()
                finite = torch.isfinite(observed_float)
                nonfinite = int((~finite).sum().item())
                safe_observed = torch.where(finite, observed_float, torch.zeros_like(observed_float))
                error = (safe_observed - expected_float).abs().reshape(-1)
                tolerance = task.numerics.atol + task.numerics.rtol * expected_float.abs()
                mismatch = int(((error.reshape(expected_float.shape) > tolerance) | ~finite).sum().item())
                maximum = float(error.max().item()) if error.numel() else 0.0
                # torch.quantile rejects very large tensors and sorting the
                # full output would distort the correctness memory budget.
                # A fixed-stride sample is deterministic and bounded.
                stride = max(1, math.ceil(error.numel() / 1_000_000))
                percentile_sample = error[::stride]
                percentile = (
                    float(torch.quantile(percentile_sample, 0.99).item())
                    if percentile_sample.numel()
                    else 0.0
                )
                denominator = (
                    task.numerics.atol + task.numerics.rtol * expected_float.abs()
                ).clamp_min(torch.finfo(torch.float32).tiny)
                normalized = float((error.reshape(expected_float.shape) / denominator).max().item())
                count = observed.numel()
                if nonfinite:
                    violations.add("nonfinite_output")
                if mismatch:
                    violations.add("numerical_mismatch")
            tensor_metrics.append(
                TensorError(
                    tensor=name,
                    count=count,
                    mismatch_count=mismatch,
                    nonfinite_count=nonfinite,
                    max_abs_error=maximum,
                    p99_abs_error=percentile,
                    max_normalized_error=normalized,
                    expected_dtype=str(expected_dtype),
                    observed_dtype=str(observed.dtype),
                    expected_shape=expected_shape,
                    observed_shape=observed_shape,
                )
            )
        if not deterministic:
            violations.add("nondeterministic")
        return CorrectnessCaseResult(
            case_id=case.id,
            tier=case.tier,
            shape=case.shape,
            seed=case.seed,
            distribution=case.distribution,
            passed=not violations,
            deterministic=deterministic,
            cuda_error=cuda_error,
            violations=tuple(sorted(violations)),
            guard_canary_intact=guard_intact,
            inputs_unmodified=inputs_unmodified,
            tensors=tuple(tensor_metrics),
        )


__all__ = [
    "CandidateCallable",
    "CandidateRun",
    "CorrectnessHarness",
    "HarnessContext",
    "ReferenceAdapter",
]
