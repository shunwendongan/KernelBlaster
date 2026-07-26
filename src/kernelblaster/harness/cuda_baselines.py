"""Trusted launch Adapter for the auditable Core 10 backward CUDA baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import Direction, TaskSpec
from .reference import concrete_shape, torch_dtype
from .runtime import CandidateRun, HarnessContext


_KERNELS = {
    "004": ("kb004_grad_a", "kb004_grad_b"),
    "007": ("kb007_grad_a", "kb007_grad_b"),
    "019": ("kb019_relu_backward",),
    "023": ("kb023_softmax_backward",),
    "026": ("kb026_gelu_backward",),
    "036": ("kb036_rmsnorm_backward",),
    "040": (
        "kb040_layernorm_stats",
        "kb040_layernorm_grad_input",
        "kb040_layernorm_grad_affine",
    ),
    "047": ("kb047_sum_backward",),
    "088": ("kb088_mingpt_gelu_backward",),
    "095": ("kb095_cross_entropy_backward",),
}


def _cupy() -> Any:
    try:
        import cupy
    except ImportError as error:  # pragma: no cover - optional NGC runtime
        raise RuntimeError("the Core 10 CUDA fixture requires CuPy") from error
    return cupy


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional gpu extra
        raise RuntimeError("the Core 10 CUDA fixture requires PyTorch") from error
    return torch


class Core10NaiveCudaBackwardCandidate:
    """Launch fixed repository kernels on the Harness-owned non-default stream."""

    def __init__(self, task: TaskSpec, *, source_root: Path | None = None) -> None:
        if task.direction is not Direction.BACKWARD:
            raise ValueError("the naive CUDA baseline is backward-only")
        self.task = task
        self.number = task.id.split(".")[2]
        root = source_root or (
            Path(__file__).resolve().parents[3] / "portfolio" / "harness" / "core10" / "backward"
        )
        matches = sorted(root.glob(f"{self.number}_*.cu"))
        if len(matches) != 1:
            raise FileNotFoundError(f"missing unique backward baseline for {self.number}")
        cupy = _cupy()
        self.module = cupy.RawModule(
            code=matches[0].read_text(encoding="utf-8"),
            options=("--std=c++17",),
            backend="nvcc",
        )
        self.kernels = {name: self.module.get_function(name) for name in _KERNELS[self.number]}

    def __call__(
        self,
        inputs: Mapping[str, Any],
        scalars: Mapping[str, int | float | bool],
        context: HarnessContext,
    ) -> CandidateRun:
        if context.stream is None or not context.device.startswith("cuda"):
            raise ValueError("the CUDA baseline requires a Harness-owned CUDA stream")
        torch = _torch()
        cupy = _cupy()
        bindings: dict[str, int] = {}
        for spec in self.task.tensors:
            value = inputs.get(spec.name)
            if value is None:
                continue
            for symbol, concrete in zip(spec.shape, value.shape, strict=True):
                if isinstance(symbol, str):
                    previous = bindings.setdefault(symbol, int(concrete))
                    if previous != int(concrete):
                        raise ValueError("candidate input shape binding mismatch")
        output_specs = {item.name: item for item in self.task.tensors if item.mutable}
        outputs = {
            name: torch.empty(
                concrete_shape(spec.shape, bindings),
                dtype=torch_dtype(spec.dtype),
                device=context.device,
            )
            for name, spec in output_specs.items()
        }

        arrays = {
            **{name: cupy.from_dlpack(value) for name, value in inputs.items()},
            **{name: cupy.from_dlpack(value) for name, value in outputs.items()},
        }
        i64 = cupy.int64
        f32 = cupy.float32
        block = 256

        def linear(name: str, count: int, arguments: tuple[Any, ...]) -> None:
            self.kernels[name](((count + block - 1) // block,), (block,), arguments)

        stream = cupy.cuda.ExternalStream(context.stream.cuda_stream)
        with stream:
            if self.number == "004":
                m, k = bindings["M"], bindings["K"]
                linear("kb004_grad_a", m * k, (arrays["grad_A"], arrays["grad_output"], arrays["B"], i64(m), i64(k)))
                linear("kb004_grad_b", k, (arrays["grad_B"], arrays["A"], arrays["grad_output"], i64(m), i64(k)))
            elif self.number == "007":
                m, n, k = bindings["M"], bindings["N"], bindings["K"]
                linear("kb007_grad_a", m * k, (arrays["grad_A"], arrays["grad_output"], arrays["B"], i64(m), i64(n), i64(k)))
                linear("kb007_grad_b", k * n, (arrays["grad_B"], arrays["A"], arrays["grad_output"], i64(m), i64(n), i64(k)))
            elif self.number in {"019", "026", "088"}:
                count = inputs["input"].numel()
                name = _KERNELS[self.number][0]
                linear(name, count, (arrays["grad_input"], arrays["input"], arrays["grad_output"], i64(count)))
            elif self.number == "023":
                batch, dim = bindings["B"], bindings["D"]
                self.kernels["kb023_softmax_backward"](
                    (batch,),
                    (block,),
                    (arrays["grad_input"], arrays["input"], arrays["grad_output"], i64(batch), i64(dim)),
                    shared_mem=block * 4,
                )
            elif self.number == "036":
                batch, channels = bindings["B"], bindings["C"]
                dim1, dim2 = bindings["D1"], bindings["D2"]
                self.kernels["kb036_rmsnorm_backward"](
                    (batch * dim1 * dim2,),
                    (block,),
                    (
                        arrays["grad_input"], arrays["input"], arrays["grad_output"],
                        i64(batch), i64(channels), i64(dim1), i64(dim2), f32(scalars["eps"]),
                    ),
                    shared_mem=block * 8,
                )
            elif self.number == "040":
                batch = bindings["B"]
                normalized = bindings["F"] * bindings["D1"] * bindings["D2"]
                floats = context.workspace[: batch * 8].view(torch.float32)
                means = cupy.from_dlpack(floats[:batch])
                inverse = cupy.from_dlpack(floats[batch : 2 * batch])
                self.kernels["kb040_layernorm_stats"](
                    (batch,), (block,),
                    (means, inverse, arrays["input"], i64(batch), i64(normalized), f32(scalars.get("eps", 1e-5))),
                    shared_mem=block * 8,
                )
                self.kernels["kb040_layernorm_grad_input"](
                    (batch,), (block,),
                    (arrays["grad_input"], arrays["input"], arrays["weight"], arrays["grad_output"], means, inverse, i64(batch), i64(normalized)),
                    shared_mem=block * 8,
                )
                linear(
                    "kb040_layernorm_grad_affine",
                    normalized,
                    (arrays["grad_weight"], arrays["grad_bias"], arrays["input"], arrays["grad_output"], means, inverse, i64(batch), i64(normalized)),
                )
            elif self.number == "047":
                batch, dim1, dim2 = bindings["B"], bindings["D1"], bindings["D2"]
                count = batch * dim1 * dim2
                linear(
                    "kb047_sum_backward", count,
                    (arrays["grad_input"], arrays["grad_output"], i64(batch), i64(dim1), i64(dim2)),
                )
            elif self.number == "095":
                batch, classes = bindings["B"], bindings["C"]
                linear(
                    "kb095_cross_entropy_backward", batch,
                    (arrays["grad_predictions"], arrays["predictions"], arrays["targets"], arrays["grad_loss"], i64(batch), i64(classes)),
                )
            else:  # pragma: no cover - constructor restricts IDs
                raise KeyError("backward CUDA baseline is unsupported")
        return CandidateRun(outputs=outputs)


__all__ = ["Core10NaiveCudaBackwardCandidate"]
