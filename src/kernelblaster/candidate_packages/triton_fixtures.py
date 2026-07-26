"""Auditable fixed Triton AOT fixtures for the first four supported tasks."""

from __future__ import annotations

from typing import Callable

from ..harness.contracts import Direction, TaskSpec
from .contracts import (
    CandidateBackend,
    CandidateLaunchPlan,
    CandidateProvenance,
    Dimensions,
    DispatchRule,
    KernelDeclaration,
    KernelLaunch,
)
from .package import BackendUnsupportedError, build_candidate_package


TRITON_AOT_TASK_IDS = frozenset({"007", "026", "036", "047"})

_SOURCES = {
    "007": b'''import triton\nimport triton.language as tl\n\n@triton.jit\ndef kb007_matmul(output, A, B, M, N, K):\n    pid = tl.program_id(0)\n    tile_m = pid // tl.cdiv(N, 16)\n    tile_n = pid % tl.cdiv(N, 16)\n    rows = tile_m * 16 + tl.arange(0, 16)\n    cols = tile_n * 16 + tl.arange(0, 16)\n    value = tl.zeros((16, 16), tl.float32)\n    for depth in range(0, 32):\n        a = tl.load(A + rows * K + depth, mask=(rows < M) & (depth < K), other=0.0).to(tl.float32)\n        b = tl.load(B + depth * N + cols, mask=(depth < K) & (cols < N), other=0.0).to(tl.float32)\n        value += a[:, None] * b[None, :]\n    tl.store(output + rows[:, None] * N + cols[None, :], value, mask=(rows[:, None] < M) & (cols[None, :] < N))\n''',
    "026": b'''import triton\nimport triton.language as tl\n\n@triton.jit\ndef kb026_gelu(output, input, B, D):\n    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)\n    count = B * D\n    x = tl.load(input + offsets, mask=offsets < count, other=0.0).to(tl.float32)\n    value = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))\n    tl.store(output + offsets, value, mask=offsets < count)\n''',
    "036": b'''import triton\nimport triton.language as tl\n\n@triton.jit\ndef kb036_rmsnorm(output, input, eps, B, C, D1, D2):\n    pid = tl.program_id(0)\n    b = pid // (D1 * D2)\n    spatial = pid % (D1 * D2)\n    channel = tl.arange(0, 128)\n    offsets = ((b * C + channel) * D1 * D2) + spatial\n    mask = channel < C\n    value = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)\n    inv = tl.rsqrt(tl.sum(value * value, axis=0) / C + eps)\n    tl.store(output + offsets, value * inv, mask=mask)\n''',
    "047": b'''import triton\nimport triton.language as tl\n\n@triton.jit\ndef kb047_sum(output, input, reduce_dim, B, D1, D2):\n    pid = tl.program_id(0)\n    b = pid // D2\n    column = pid % D2\n    rows = tl.arange(0, 256)\n    offsets = (b * D1 + rows) * D2 + column\n    value = tl.load(input + offsets, mask=rows < D1, other=0.0).to(tl.float32)\n    tl.store(output + b * D2 + column, tl.sum(value, axis=0))\n''',
}


def _task_number(task: TaskSpec) -> str:
    parts = task.id.split(".")
    if len(parts) < 4 or parts[:2] != ["kernelbench", "level1"]:
        raise BackendUnsupportedError("backend_unsupported")
    return parts[2]


def _launch_plan(task: TaskSpec, number: str) -> CandidateLaunchPlan:
    shapes = tuple(item.name for item in task.dimensions)
    tensors = tuple(item.name for item in task.tensors)
    scalars = tuple(item.name for item in task.scalars)
    definitions: dict[str, tuple[str, tuple[str, ...], str]] = {
        "007": ("kb007_matmul", ("output", "A", "B", "M", "N", "K"), "ceil_div(M, 16) * ceil_div(N, 16)"),
        "026": ("kb026_gelu", ("output", "input", "B", "D"), "ceil_div(B * D, 256)"),
        "036": ("kb036_rmsnorm", ("output", "input", "eps", "B", "C", "D1", "D2"), "B * D1 * D2"),
        "047": ("kb047_sum", ("output", "input", "reduce_dim", "B", "D1", "D2"), "B * D2"),
    }
    kernel, arguments, grid_x = definitions[number]
    return CandidateLaunchPlan(
        task_spec_digest=task.canonical_sha256(),
        backend=CandidateBackend.TRITON,
        shape_symbols=shapes,
        tensor_bindings=tensors,
        scalar_bindings=scalars,
        kernels=(KernelDeclaration(name=kernel, parameters=arguments),),
        dispatch=(
            DispatchRule(
                launches=(
                    KernelLaunch(
                        kernel=kernel,
                        grid=Dimensions(x=grid_x),
                        block=Dimensions(x="32"),
                        arguments=arguments,
                    ),
                )
            ),
        ),
    )


def build_fixed_triton_candidate(task: TaskSpec) -> bytes:
    """Return a deterministic package or explicitly reject unsupported tasks."""
    number = _task_number(task)
    if (
        task.direction is not Direction.FORWARD
        or number not in TRITON_AOT_TASK_IDS
        or "triton" not in task.candidate_backends
    ):
        raise BackendUnsupportedError("backend_unsupported")
    return build_candidate_package(
        _SOURCES[number],
        _launch_plan(task, number),
        task=task,
        provenance=CandidateProvenance(generator="kernelblaster-fixed-triton-v1"),
    )


__all__ = ["TRITON_AOT_TASK_IDS", "build_fixed_triton_candidate"]
