"""Public Core 10 task contracts used to prove the generic Harness surface."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CacheMode,
    DeterminismLevel,
    Direction,
    NumericsPolicy,
    ScalarSpec,
    ShapeDimension,
    TaskSpec,
    TensorSpec,
    WorkloadSpec,
    WorkspacePolicy,
)


CORE10_IDS = ("004", "007", "019", "023", "026", "036", "040", "047", "088", "095")
_TRITON_FORWARD = {"007", "026", "036", "047"}


@dataclass(frozen=True)
class _Definition:
    operator: str
    dimensions: dict[str, tuple[int, ...]]
    inputs: tuple[tuple[str, str, tuple[str | int, ...], bool], ...]
    outputs: tuple[tuple[str, str, tuple[str | int, ...]], ...]
    scalars: tuple[tuple[str, str, int | float | bool], ...] = ()
    atol: float = 0.005
    rtol: float = 0.01
    workspace_bytes: int = 0


_DEFINITIONS: dict[str, _Definition] = {
    "004": _Definition(
        "Matrix-vector multiplication",
        {"M": (1, 17, 33, 256, 257), "K": (1, 31, 65, 129, 131072)},
        (("A", "fp16", ("M", "K"), True), ("B", "fp16", ("K",), True)),
        (("output", "fp16", ("M",)),),
        atol=0.01,
    ),
    "007": _Definition(
        "Small-K matrix multiplication",
        {"M": (1, 17, 65, 16384), "N": (1, 19, 33, 16384), "K": (1, 7, 31, 32)},
        (("A", "fp16", ("M", "K"), True), ("B", "fp16", ("K", "N"), True)),
        (("output", "fp16", ("M", "N")),),
        atol=0.01,
    ),
    "019": _Definition(
        "ReLU",
        {"B": (1, 16), "D": (1, 31, 32, 33, 16384)},
        (("input", "fp16", ("B", "D"), True),),
        (("output", "fp16", ("B", "D")),),
    ),
    "023": _Definition(
        "Softmax",
        {"B": (1, 16), "D": (1, 31, 32, 33, 16384)},
        (("input", "fp16", ("B", "D"), True),),
        (("output", "fp16", ("B", "D")),),
    ),
    "026": _Definition(
        "GELU",
        {"B": (1, 16), "D": (1, 31, 32, 33, 16384)},
        (("input", "fp16", ("B", "D"), True),),
        (("output", "fp16", ("B", "D")),),
    ),
    "036": _Definition(
        "RMSNorm channel-first",
        {
            "B": (1, 2, 16),
            "C": (4, 63, 64, 65),
            "D1": (1, 3, 17, 256),
            "D2": (3, 5, 7, 17, 19, 256),
        },
        (("input", "fp16", ("B", "C", "D1", "D2"), True),),
        (("output", "fp16", ("B", "C", "D1", "D2")),),
        (("eps", "float32", 1e-5),),
    ),
    "040": _Definition(
        "LayerNorm",
        {
            "B": (1, 2, 3, 16),
            "F": (3, 7, 16, 64),
            "D1": (5, 9, 17, 256),
            "D2": (7, 11, 18, 256),
        },
        (
            ("input", "fp16", ("B", "F", "D1", "D2"), True),
            ("weight", "fp16", ("F", "D1", "D2"), True),
            ("bias", "fp16", ("F", "D1", "D2"), True),
        ),
        (("output", "fp16", ("B", "F", "D1", "D2")),),
        workspace_bytes=32896,
    ),
    "047": _Definition(
        "Sum reduction over dimension",
        {"B": (1, 16), "D1": (1, 31, 32, 33, 256), "D2": (1, 31, 32, 33, 256)},
        (("input", "fp16", ("B", "D1", "D2"), True),),
        (("output", "fp16", ("B", 1, "D2")),),
        (("reduce_dim", "int64", 1),),
        atol=0.01,
    ),
    "088": _Definition(
        "MinGPT GELU",
        {"B": (1, 16, 2000), "D": (1, 31, 32, 33, 2000)},
        (("input", "fp16", ("B", "D"), True),),
        (("output", "fp16", ("B", "D")),),
    ),
    "095": _Definition(
        "Cross entropy loss",
        {"B": (1, 17, 65, 257, 4096), "C": (2, 7, 9, 10)},
        (
            ("predictions", "fp16", ("B", "C"), True),
            ("targets", "int64", ("B",), False),
        ),
        (("loss", "fp16", (1,)),),
        atol=0.01,
        workspace_bytes=4096 * 4,
    ),
}


def _dimension(name: str, values: tuple[int, ...]) -> ShapeDimension:
    return ShapeDimension(name=name, minimum=min(values), maximum=max(values), values=values)


def _canonical_shape(definition: _Definition) -> dict[str, int]:
    return {name: values[-1] for name, values in definition.dimensions.items()}


def _workloads(definition: _Definition) -> tuple[WorkloadSpec, ...]:
    shape = _canonical_shape(definition)
    return (
        WorkloadSpec(id="canonical-hot", shape=shape, weight=0.7, cache_mode=CacheMode.HOT),
        WorkloadSpec(
            id="canonical-rotating", shape=shape, weight=0.3, cache_mode=CacheMode.ROTATING
        ),
    )


def _forward(task_id: str, definition: _Definition) -> TaskSpec:
    tensors = tuple(
        TensorSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            differentiable=differentiable,
        )
        for name, dtype, shape, differentiable in definition.inputs
    ) + tuple(
        TensorSpec(name=name, dtype=dtype, shape=shape, mutable=True)
        for name, dtype, shape in definition.outputs
    )
    return TaskSpec(
        id=f"kernelbench.level1.{task_id}.forward",
        operator=definition.operator,
        adapter_id="kernelbench.legacy-driver",
        adapter_version="1.0.0",
        direction=Direction.FORWARD,
        tensors=tensors,
        scalars=tuple(
            ScalarSpec(name=name, dtype=dtype, default=default)
            for name, dtype, default in definition.scalars
        ),
        dimensions=tuple(_dimension(name, values) for name, values in definition.dimensions.items()),
        workloads=_workloads(definition),
        numerics=NumericsPolicy(
            reference_dtype="fp32", atol=definition.atol, rtol=definition.rtol
        ),
        workspace=WorkspacePolicy(maximum_bytes=definition.workspace_bytes),
        candidate_backends=("cuda", "triton") if task_id in _TRITON_FORWARD else ("cuda",),
    )


def _backward(task_id: str, definition: _Definition) -> TaskSpec:
    differentiable = tuple(item for item in definition.inputs if item[3])
    tensors = tuple(
        TensorSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            differentiable=is_differentiable,
        )
        for name, dtype, shape, is_differentiable in definition.inputs
    )
    tensors += tuple(
        TensorSpec(name=f"grad_{name}", dtype=dtype, shape=shape)
        for name, dtype, shape in definition.outputs
    )
    tensors += tuple(
        TensorSpec(
            name=f"grad_{name}",
            dtype=dtype,
            shape=shape,
            mutable=True,
            gradient_of=name,
        )
        for name, dtype, shape, _ in differentiable
    )
    return TaskSpec(
        id=f"kernelbench.level1.{task_id}.backward",
        operator=definition.operator,
        adapter_id="kernelbench.pytorch-autograd-backward",
        adapter_version="1.0.0",
        direction=Direction.BACKWARD,
        tensors=tensors,
        scalars=tuple(
            ScalarSpec(name=name, dtype=dtype, default=default)
            for name, dtype, default in definition.scalars
        ),
        dimensions=tuple(_dimension(name, values) for name, values in definition.dimensions.items()),
        workloads=_workloads(definition),
        gradient_targets=tuple(name for name, *_ in differentiable),
        numerics=NumericsPolicy(
            reference_dtype="fp32", atol=definition.atol, rtol=definition.rtol
        ),
        determinism=DeterminismLevel.BITWISE,
        workspace=WorkspacePolicy(maximum_bytes=max(definition.workspace_bytes, 1024 * 1024)),
        candidate_backends=("cuda",),
    )


def core10_task_specs() -> tuple[TaskSpec, ...]:
    specs: list[TaskSpec] = []
    for task_id in CORE10_IDS:
        definition = _DEFINITIONS[task_id]
        specs.extend((_forward(task_id, definition), _backward(task_id, definition)))
    return tuple(specs)
