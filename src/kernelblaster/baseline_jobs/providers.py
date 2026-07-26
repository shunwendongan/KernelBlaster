"""Fixed trusted provider implementations shipped in the Baseline image."""

from __future__ import annotations

import math
import time
from typing import Any

from ..harness.contracts import CaseBundle, CaseSpec, CaseTier, TaskSpec
from ..harness.reference import PyTorchAutogradAdapter, concrete_shape, torch_dtype
from ..harness.runtime import CorrectnessHarness
from .contracts import (
    BaselineProvider,
    BaselineReasonCode,
    BaselineRequest,
    BaselineStatus,
    BaselineWorkloadMeasurement,
)
from .worker import ProviderExecution


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - image contract
        raise RuntimeError("PyTorch baseline provider requires PyTorch") from error
    return torch


class PyTorchEagerProviderRuntime:
    """Correctness-gated PyTorch eager reference with Events and host timing."""

    async def execute(
        self,
        request: BaselineRequest,
        task: TaskSpec,
        cases: CaseBundle,
        evaluation_bundle: bytes,
    ) -> ProviderExecution:
        del evaluation_bundle
        torch = _torch()
        if not torch.cuda.is_available():
            return ProviderExecution(
                status=BaselineStatus.UNAVAILABLE,
                reason_code=BaselineReasonCode.TOOL_MISSING,
                correctness_passed=False,
                provider_version=f"torch:{torch.__version__}",
            )
        adapter = PyTorchAutogradAdapter()
        output_specs = {item.name: item for item in task.tensors if item.mutable}

        def candidate(inputs, scalars, context):
            del context
            return {
                name: value.to(torch_dtype(output_specs[name].dtype))
                for name, value in adapter.reference(task, inputs, scalars).items()
            }

        correctness = CorrectnessHarness(device="cuda").evaluate(
            task, cases, adapter=adapter, candidate=candidate
        )
        if not correctness.passed:
            return ProviderExecution(
                status=BaselineStatus.FAILED,
                reason_code=BaselineReasonCode.CORRECTNESS_FAILED,
                correctness_passed=False,
                provider_version=f"torch:{torch.__version__}",
            )
        measurements = tuple(
            self._measure(task, workload, adapter) for workload in task.workloads
        )
        return ProviderExecution(
            status=BaselineStatus.SUCCEEDED,
            reason_code=BaselineReasonCode.NONE,
            correctness_passed=True,
            provider_version=f"torch:{torch.__version__}",
            workloads=measurements,
        )

    @staticmethod
    def _measure(task: TaskSpec, workload: Any, adapter: PyTorchAutogradAdapter) -> BaselineWorkloadMeasurement:
        torch = _torch()
        case = CaseSpec(
            id=f"baseline-{workload.id}",
            tier=CaseTier.DEV,
            shape=workload.shape,
            seed=20260726,
            distribution="normal",
        )
        inputs = adapter.create_inputs(task, case, device="cuda")
        scalars = adapter.scalar_values(task)
        maximum_elements = max(
            math.prod(concrete_shape(item.shape, workload.shape)) for item in task.tensors
        )
        repeats = 20 if maximum_elements <= 1_000_000 else (5 if maximum_elements <= 32_000_000 else 1)
        bank = [inputs]
        if workload.cache_mode.value == "rotating":
            input_bytes = max(
                1, sum(value.numel() * value.element_size() for value in inputs.values())
            )
            properties = torch.cuda.get_device_properties(0)
            l2_bytes = int(getattr(properties, "L2_cache_size", 4 * 1024 * 1024))
            bank_size = min(16, max(2, math.ceil((2 * l2_bytes) / input_bytes)))
            bank.extend(
                {name: value.clone() for name, value in inputs.items()}
                for _ in range(bank_size - 1)
            )
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for index in range(min(5, repeats)):
                adapter.reference(task, bank[index % len(bank)], scalars)
        stream.synchronize()
        device_samples: list[float] = []
        host_samples: list[float] = []
        for session in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                start.record(stream)
                for index in range(repeats):
                    adapter.reference(task, bank[(session + index) % len(bank)], scalars)
                end.record(stream)
            end.synchronize()
            host_elapsed = (time.perf_counter_ns() - host_start) / 1000.0 / repeats
            device_samples.append(float(start.elapsed_time(end) * 1000.0 / repeats))
            host_samples.append(host_elapsed)
        return BaselineWorkloadMeasurement(
            workload_id=workload.id,
            cache_mode=workload.cache_mode.value,
            weight=workload.weight,
            core=workload.core,
            device_samples_us=tuple(device_samples),
            host_samples_us=tuple(host_samples),
        )


def built_in_provider_runtimes() -> dict[BaselineProvider, Any]:
    """Unavailable providers remain explicit; no implementation fallback is allowed."""
    return {BaselineProvider.PYTORCH_EAGER: PyTorchEagerProviderRuntime()}


__all__ = ["PyTorchEagerProviderRuntime", "built_in_provider_runtimes"]
