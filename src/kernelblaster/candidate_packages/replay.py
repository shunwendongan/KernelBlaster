"""Fixed trusted replay for AOT capsules; candidate Python is never imported."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
import tempfile
import time
from typing import Any, Mapping

from ..harness.contracts import CaseBundle, CaseSpec, CaseTier, TaskSpec
from ..harness.reference import PyTorchAutogradAdapter, concrete_shape, torch_dtype
from ..harness.runtime import CandidateRun, CorrectnessHarness, HarnessContext
from .package import ValidatedCandidateCapsule, validate_candidate_capsule


def _torch() -> Any:
    import torch

    return torch


def _cupy() -> Any:
    import cupy

    return cupy


class CapsuleCandidate:
    def __init__(self, capsule: ValidatedCandidateCapsule, task: TaskSpec, module_path: Path):
        self.capsule = capsule
        self.task = task
        cupy = _cupy()
        self.module = cupy.RawModule(path=str(module_path))
        self.kernels = {
            item.name: self.module.get_function(item.name)
            for item in capsule.launch_plan.kernels
        }

    def _bindings(self, inputs: Mapping[str, Any]) -> dict[str, int]:
        bindings: dict[str, int] = {}
        for spec in self.task.tensors:
            value = inputs.get(spec.name)
            if value is None:
                continue
            for symbol, concrete in zip(spec.shape, value.shape, strict=True):
                if isinstance(symbol, str):
                    previous = bindings.setdefault(symbol, int(concrete))
                    if previous != int(concrete):
                        raise ValueError("capsule input shape binding mismatch")
        return bindings

    def prepare(
        self,
        inputs: Mapping[str, Any],
        scalars: Mapping[str, int | float | bool],
        context: HarnessContext,
    ) -> tuple[dict[str, Any], Any]:
        torch = _torch()
        cupy = _cupy()
        shapes = self._bindings(inputs)
        output_specs = {item.name: item for item in self.task.tensors if item.mutable}
        outputs = {
            name: torch.empty(
                concrete_shape(spec.shape, shapes),
                dtype=torch_dtype(spec.dtype),
                device=context.device,
            )
            for name, spec in output_specs.items()
        }
        arrays = {
            **{name: cupy.from_dlpack(value) for name, value in inputs.items()},
            **{name: cupy.from_dlpack(value) for name, value in outputs.items()},
            "workspace": cupy.from_dlpack(context.workspace),
        }
        scalar_specs = {item.name: item for item in self.task.scalars}
        resolved = self.capsule.launch_plan.select({**shapes, **scalars})
        external = cupy.cuda.ExternalStream(context.stream.cuda_stream)

        def argument(name: str) -> Any:
            if name in arrays:
                return arrays[name]
            if name in shapes:
                return cupy.int64(shapes[name])
            spec = scalar_specs[name]
            return {
                "int32": cupy.int32,
                "int64": cupy.int64,
                "float32": cupy.float32,
                "float64": cupy.float64,
                "bool": cupy.bool_,
            }[spec.dtype](scalars[name])

        def launch() -> None:
            with external:
                for item in resolved:
                    self.kernels[item.kernel](
                        item.grid,
                        item.block,
                        tuple(argument(name) for name in item.arguments),
                        shared_mem=item.dynamic_shared_bytes,
                    )

        return outputs, launch

    def __call__(self, inputs, scalars, context) -> CandidateRun:
        outputs, launch = self.prepare(inputs, scalars, context)
        launch()
        return CandidateRun(outputs=outputs)


def _materialize_capsule(payload: bytes, root: Path) -> Path:
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        stream = archive.extractfile("module.cubin")
        if stream is None:
            raise ValueError("capsule module is missing")
        path = root / "module.cubin"
        path.write_bytes(stream.read())
        return path


def _events(
    capsule: ValidatedCandidateCapsule,
    task: TaskSpec,
    *,
    workload_id: str | None,
    protocol_id: str,
    module_path: Path,
) -> dict[str, object]:
    torch = _torch()
    adapter = PyTorchAutogradAdapter()
    workload = next(
        (item for item in task.workloads if workload_id is None or item.id == workload_id),
        None,
    )
    if workload is None:
        raise ValueError("requested workload is not in TaskSpec")
    case = CaseSpec(
        id="events-workload",
        tier=CaseTier.DEV,
        shape=workload.shape,
        seed=20260726,
        distribution="normal",
    )
    original = adapter.create_inputs(task, case, device="cuda")
    scalars = adapter.scalar_values(task)
    input_bytes = max(1, sum(value.numel() * value.element_size() for value in original.values()))
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 4 * 1024**2))
    bank_size = (
        min(16, max(2, math.ceil((2 * l2) / input_bytes)))
        if workload.cache_mode.value == "rotating"
        else 1
    )
    banks = [original] + [
        {name: value.clone() for name, value in original.items()}
        for _ in range(bank_size - 1)
    ]
    stream = torch.cuda.Stream()
    candidate = CapsuleCandidate(capsule, task, module_path)
    prepared: list[tuple[dict[str, Any], Any]] = []
    for inputs in banks:
        workspace = torch.empty(task.workspace.maximum_bytes, dtype=torch.uint8, device="cuda")
        prepared.append(
            candidate.prepare(
                inputs,
                scalars,
                HarnessContext(device="cuda", stream=stream, workspace=workspace),
            )
        )
    with torch.cuda.stream(stream):
        for index in range(20):
            prepared[index % len(prepared)][1]()
    stream.synchronize()
    device_samples: list[float] = []
    host_samples: list[float] = []
    for session in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            start.record(stream)
            for index in range(100):
                prepared[(session + index) % len(prepared)][1]()
            end.record(stream)
        end.synchronize()
        device_samples.append(float(start.elapsed_time(end) * 10.0))
        host_samples.append((time.perf_counter_ns() - host_start) / 1000.0 / 100)
    return {
        "value": statistics.median(device_samples),
        "unit": "us",
        "source": "cuda_events",
        "samples": device_samples,
        "host_samples_us": host_samples,
        "protocol_id": protocol_id,
        "workload_id": workload.id,
        "cache_mode": workload.cache_mode.value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--mode", choices=("correctness", "events"), required=True)
    parser.add_argument("--protocol", default="candidate-capsule-events-v1")
    parser.add_argument("--workload")
    args = parser.parse_args()
    capsule_payload = args.capsule.read_bytes()
    capsule = validate_candidate_capsule(capsule_payload)
    task = TaskSpec.model_validate_json(args.task.read_bytes())
    cases = CaseBundle.model_validate_json(args.cases.read_bytes())
    if capsule.manifest.task_spec_digest != task.canonical_sha256():
        raise ValueError("capsule TaskSpec binding mismatch")
    cases.validate_for(task)
    temporary_root = Path(os.environ.get("TMPDIR", "/work"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=temporary_root, prefix="capsule-replay-"
    ) as temporary:
        module_path = _materialize_capsule(capsule_payload, Path(temporary))
        if args.mode == "correctness":
            result = CorrectnessHarness(device="cuda").evaluate(
                task,
                cases,
                adapter=PyTorchAutogradAdapter(),
                candidate=CapsuleCandidate(capsule, task, module_path),
            )
            print(result.canonical_bytes().decode("utf-8"))
            return 0 if result.passed else 1
        print(
            json.dumps(
                _events(
                    capsule,
                    task,
                    workload_id=args.workload,
                    protocol_id=args.protocol,
                    module_path=module_path,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CapsuleCandidate"]
