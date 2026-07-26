"""Trusted PyTorch references for the public Core 10 Adapter.

PyTorch is an optional runtime dependency.  It is used here as a correctness
oracle, never as an ABI requirement for third-party TaskSpec plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import CaseSpec, Direction, TaskSpec


_TORCH_DTYPES = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
}


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("the PyTorch correctness Adapter requires the gpu extra") from error
    return torch


def torch_dtype(name: str) -> Any:
    torch = _torch()
    attribute = _TORCH_DTYPES.get(name)
    if attribute is None:
        raise ValueError(f"PyTorch Adapter does not support dtype {name}")
    return getattr(torch, attribute)


def concrete_shape(shape: tuple[str | int, ...], bindings: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(bindings[item] if isinstance(item, str) else item for item in shape)


@dataclass(frozen=True)
class PyTorchAutogradAdapter:
    """Allowlisted Core 10 reference with FP16 inputs and FP32 math/autograd."""

    id: str = "kernelbench.pytorch-autograd-backward"
    version: str = "1.0.0"

    def create_inputs(self, task: TaskSpec, case: CaseSpec, *, device: str) -> dict[str, Any]:
        torch = _torch()
        generator = torch.Generator(device=device)
        generator.manual_seed(case.seed)
        inputs: dict[str, Any] = {}
        for spec in task.tensors:
            if spec.mutable or spec.gradient_of is not None:
                continue
            shape = concrete_shape(spec.shape, case.shape)
            dtype = torch_dtype(spec.dtype)
            if spec.dtype.startswith("int"):
                maximum = case.shape.get("C", 17) if spec.name == "targets" else 17
                tensor = torch.randint(0, maximum, shape, dtype=dtype, device=device, generator=generator)
            elif spec.dtype == "bool":
                tensor = torch.randint(0, 2, shape, dtype=torch.int8, device=device, generator=generator).bool()
            elif case.distribution == "zero":
                tensor = torch.zeros(shape, dtype=dtype, device=device)
            elif case.distribution == "alternating":
                count = 1
                for value in shape:
                    count *= value
                tensor = ((torch.arange(count, device=device) & 1) * 2 - 1).reshape(shape).to(dtype)
            elif case.distribution == "extreme":
                count = 1
                for value in shape:
                    count *= value
                # Exercise large signed values without making a valid FP16
                # reduction overflow solely because of the fixture itself.
                tensor = (((torch.arange(count, device=device) & 1) * 2 - 1) * 4).reshape(shape).to(dtype)
            elif case.distribution == "constant":
                tensor = torch.full(shape, 0.125, dtype=dtype, device=device)
            else:
                tensor = torch.randn(shape, dtype=dtype, device=device, generator=generator)
            inputs[spec.name] = tensor
        return inputs

    @staticmethod
    def scalar_values(task: TaskSpec) -> dict[str, int | float | bool]:
        return {item.name: item.default for item in task.scalars if item.default is not None}

    def reference(
        self,
        task: TaskSpec,
        inputs: Mapping[str, Any],
        scalars: Mapping[str, int | float | bool],
    ) -> dict[str, Any]:
        if not task.id.startswith("kernelbench.level1."):
            raise KeyError("reference_adapter_task_unsupported")
        task_number = task.id.split(".")[2]
        if task.direction is Direction.FORWARD:
            output = self._forward(task_number, inputs, scalars)
            output_name = next(item.name for item in task.tensors if item.mutable)
            return {output_name: output}

        differentiable = [
            item for item in task.tensors if item.differentiable and item.gradient_of is None
        ]
        promoted: dict[str, Any] = {}
        targets: list[Any] = []
        for name, value in inputs.items():
            spec = next(item for item in task.tensors if item.name == name)
            if spec.differentiable:
                value = value.detach().to(torch_dtype(task.numerics.reference_dtype))
                value.requires_grad_(True)
                targets.append(value)
            promoted[name] = value
        output = self._forward(task_number, promoted, scalars)
        output_specs = [item for item in task.tensors if item.name.startswith("grad_") and not item.mutable]
        if len(output_specs) != 1:
            raise ValueError("Core 10 backward Adapter requires exactly one output cotangent")
        grad_output = promoted[output_specs[0].name].to(output.dtype)
        gradients = _torch().autograd.grad(
            output,
            targets,
            grad_outputs=grad_output.reshape(output.shape),
            allow_unused=False,
            create_graph=False,
        )
        return {
            f"grad_{spec.name}": gradient.detach()
            for spec, gradient in zip(differentiable, gradients, strict=True)
        }

    @staticmethod
    def _forward(
        task_number: str,
        inputs: Mapping[str, Any],
        scalars: Mapping[str, int | float | bool],
    ) -> Any:
        torch = _torch()
        functional = torch.nn.functional

        def fp32(name: str) -> Any:
            value = inputs[name]
            return value.float() if value.is_floating_point() else value

        if task_number == "004":
            return fp32("A") @ fp32("B")
        if task_number == "007":
            return fp32("A") @ fp32("B")
        if task_number == "019":
            return functional.relu(fp32("input"))
        if task_number == "023":
            return functional.softmax(fp32("input"), dim=-1)
        if task_number == "026":
            return functional.gelu(fp32("input"), approximate="none")
        if task_number == "036":
            value = fp32("input")
            eps = float(scalars["eps"])
            return value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + eps)
        if task_number == "040":
            value = fp32("input")
            normalized_shape = value.shape[1:]
            return functional.layer_norm(
                value,
                normalized_shape,
                fp32("weight"),
                fp32("bias"),
                eps=float(scalars.get("eps", 1e-5)),
            )
        if task_number == "047":
            return fp32("input").sum(dim=int(scalars["reduce_dim"]), keepdim=True)
        if task_number == "088":
            return functional.gelu(fp32("input"), approximate="tanh")
        if task_number == "095":
            return functional.cross_entropy(
                fp32("predictions"), inputs["targets"], reduction="mean"
            ).reshape(1)
        raise KeyError("reference_adapter_task_unsupported")


__all__ = ["PyTorchAutogradAdapter", "concrete_shape", "torch_dtype"]
