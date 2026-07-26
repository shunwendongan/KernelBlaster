#!/usr/bin/env python3
"""Compile one validated Triton DSL kernel into a device-only cubin."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernelblaster.candidate_packages import CandidateBackend, CandidateLaunchPlan
from src.kernelblaster.candidate_packages.package import validate_source
from src.kernelblaster.harness.contracts import TaskSpec


ARCH = re.compile(r"^sm_([0-9]{2,3})$")
TRITON_TYPES = {
    "fp16": "*fp16",
    "bf16": "*bf16",
    "fp32": "*fp32",
    "fp64": "*fp64",
    "int8": "*i8",
    "int16": "*i16",
    "int32": "*i32",
    "int64": "*i64",
    "bool": "*i1",
}
SCALAR_TYPES = {
    "fp16": "fp16",
    "bf16": "bf16",
    "fp32": "fp32",
    "fp64": "fp64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float64": "fp64",
}


def _load_kernel(source: Path, kernel_name: str) -> Any:
    spec = importlib.util.spec_from_file_location("kernelblaster_triton_candidate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Triton candidate module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        raise ValueError("Triton candidate kernel is missing")
    return kernel


def _signature(plan: CandidateLaunchPlan, task: TaskSpec) -> dict[str, str]:
    tensors = {item.name: item for item in task.tensors}
    scalars = {item.name: item for item in task.scalars}
    signature: dict[str, str] = {}
    for name in plan.kernels[0].parameters:
        if name in tensors:
            signature[name] = TRITON_TYPES[tensors[name].dtype]
        elif name in scalars:
            signature[name] = SCALAR_TYPES[scalars[name].dtype]
        elif name in plan.shape_symbols:
            signature[name] = "i64"
        elif name == "workspace":
            signature[name] = "*i8"
        else:  # validated plans should make this unreachable
            raise ValueError("Triton signature contains an unknown binding")
    return signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--target-arch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    match = ARCH.fullmatch(args.target_arch)
    if match is None:
        raise ValueError("target architecture is invalid")
    source = args.source.read_bytes()
    plan_payload = args.launch_plan.read_bytes()
    task_payload = args.task.read_bytes()
    plan = CandidateLaunchPlan.model_validate_json(plan_payload)
    task = TaskSpec.model_validate_json(task_payload)
    if plan.canonical_bytes() != plan_payload or task.canonical_bytes() != task_payload:
        raise ValueError("Triton compile inputs must use canonical encoding")
    if plan.backend is not CandidateBackend.TRITON:
        raise ValueError("Triton compiler received a non-Triton package")
    if plan.task_spec_digest != task.canonical_sha256():
        raise ValueError("Triton compile TaskSpec binding mismatch")
    if plan.backend.value not in task.candidate_backends or len(plan.kernels) != 1:
        raise ValueError("backend_unsupported")
    validate_source(source, plan)

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    block_threads = int(plan.dispatch[0].launches[0].block.x)
    kernel = _load_kernel(args.source, plan.kernels[0].name)
    compiled = triton.compile(
        ASTSource(kernel, signature=_signature(plan, task)),
        target=GPUTarget("cuda", int(match.group(1)), 32),
        options={"num_warps": block_threads // 32, "num_stages": 2},
    )
    cubin = compiled.asm.get("cubin")
    if not isinstance(cubin, bytes) or not cubin:
        raise RuntimeError("Triton compiler did not produce a cubin")
    args.output.write_bytes(cubin)
    print(
        json.dumps(
            {
                "kernel": plan.kernels[0].name,
                "num_warps": int(compiled.metadata.num_warps),
                "shared_bytes": int(compiled.metadata.shared),
                "target_arch": args.target_arch,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
