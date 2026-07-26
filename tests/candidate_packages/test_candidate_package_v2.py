from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from src.kernelblaster.candidate_packages import (
    BackendUnsupportedError,
    CandidateLaunchPlan,
    CandidateProvenance,
    CudaWinnerQualification,
    SanitizerPlan,
    SanitizerResult,
    build_candidate_capsule,
    build_candidate_package,
    build_fixed_cuda_candidate,
    build_fixed_triton_candidate,
    build_profiler_replay_capsule,
    validate_candidate_capsule,
    validate_candidate_package,
    validate_profiler_replay_capsule,
)
from src.kernelblaster.harness import build_development_case_bundle, core10_task_specs


def _task():
    return next(item for item in core10_task_specs() if item.id.endswith("019.forward"))


def _plan(**updates):
    task = _task()
    payload = {
        "task_spec_digest": task.canonical_sha256(),
        "backend": "cuda",
        "shape_symbols": ["B", "D"],
        "tensor_bindings": ["input", "output"],
        "workspace_bytes": 0,
        "kernels": [
            {
                "name": "relu_kernel",
                "parameters": ["output", "input", "B", "D"],
            }
        ],
        "dispatch": [
            {
                "when": None,
                "launches": [
                    {
                        "kernel": "relu_kernel",
                        "grid": {"x": "ceil_div(B * D, 256)"},
                        "block": {"x": "256"},
                        "arguments": ["output", "input", "B", "D"],
                    }
                ],
            }
        ],
    }
    payload.update(updates)
    return CandidateLaunchPlan.model_validate(payload)


CUDA = b'''#include <cuda_fp16.h>
extern "C" __global__ void relu_kernel(half* output, const half* input, long B, long D) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B * D) output[i] = __hgt(input[i], __float2half(0.0f)) ? input[i] : __float2half(0.0f);
}
'''


def test_cuda_package_and_capsule_are_canonical_digest_bound_and_device_only():
    task = _task()
    plan = _plan()
    payload = build_candidate_package(
        CUDA,
        plan,
        task=task,
        provenance=CandidateProvenance(generator="fixed-test"),
    )
    package = validate_candidate_package(payload, task=task)
    assert package.digest == hashlib.sha256(payload).hexdigest()
    capsule_payload = build_candidate_capsule(
        package,
        module=b"ELF-cubin",
        target_arch="sm_86",
        compiler_id="nvcc:test",
    )
    capsule = validate_candidate_capsule(capsule_payload)
    assert capsule.manifest.candidate_package_digest == package.digest
    assert capsule.launch_plan.select({"B": 2, "D": 33})[0].grid == (1, 1, 1)


@pytest.mark.parametrize(
    "addition",
    [
        b"\nint main() { return 0; }\n",
        b"\nvoid x() { cudaMalloc(nullptr, 1); }\n",
        b"\n#include <cublas_v2.h>\n",
        b"\nvoid launch_gpu_implementation() {}\n",
    ],
)
def test_cuda_candidate_cannot_supply_host_launcher_allocator_or_vendor_library(addition):
    with pytest.raises(ValueError, match="forbidden"):
        build_candidate_package(
            CUDA + addition,
            _plan(),
            task=_task(),
            provenance=CandidateProvenance(generator="mutant"),
        )


def test_launch_plan_has_bounded_dispatch_and_no_arbitrary_expression_execution():
    plan = _plan(
        dispatch=[
            {
                "when": "D <= 32",
                "launches": [
                    {
                        "kernel": "relu_kernel",
                        "grid": {"x": "ceil_div(B * D, 128)"},
                        "block": {"x": "128"},
                        "arguments": ["output", "input", "B", "D"],
                    }
                ],
            },
            {
                "when": None,
                "launches": [
                    {
                        "kernel": "relu_kernel",
                        "grid": {"x": "ceil_div(B * D, 256)"},
                        "block": {"x": "256"},
                        "arguments": ["output", "input", "B", "D"],
                    }
                ],
            },
        ]
    )
    assert plan.select({"B": 1, "D": 31})[0].block == (128, 1, 1)
    assert plan.select({"B": 1, "D": 33})[0].block == (256, 1, 1)
    with pytest.raises(ValidationError, match="forbidden"):
        _plan(
            dispatch=[
                {
                    "when": None,
                    "launches": [
                        {
                            "kernel": "relu_kernel",
                            "grid": {"x": "__import__('os').system('id')"},
                            "block": {"x": "256"},
                            "arguments": ["output", "input", "B", "D"],
                        }
                    ],
                }
            ]
        )


def test_triton_python_is_compile_only_and_rejects_top_level_execution():
    task = next(item for item in core10_task_specs() if item.id.endswith("026.forward"))
    plan = CandidateLaunchPlan.model_validate(
        {
            "task_spec_digest": task.canonical_sha256(),
            "backend": "triton",
            "shape_symbols": ["B", "D"],
            "tensor_bindings": ["input", "output"],
            "kernels": [
                {"name": "gelu_kernel", "parameters": ["output", "input", "B", "D"]}
            ],
            "dispatch": [
                {
                    "launches": [
                        {
                            "kernel": "gelu_kernel",
                            "grid": {"x": "ceil_div(B * D, 256)"},
                            "block": {"x": "256"},
                            "arguments": ["output", "input", "B", "D"],
                        }
                    ]
                }
            ],
        }
    )
    valid = b"import triton\nimport triton.language as tl\n@triton.jit\ndef gelu_kernel(output, input, B: tl.constexpr, D: tl.constexpr):\n    return\n"
    build_candidate_package(
        valid,
        plan,
        task=task,
        provenance=CandidateProvenance(generator="fixed-triton"),
    )
    with pytest.raises(ValueError, match="forbidden|top-level"):
        build_candidate_package(
            valid + b"\nprint('candidate runtime')\n",
            plan,
            task=task,
            provenance=CandidateProvenance(generator="mutant"),
        )


def test_only_all_four_sanitizers_qualify_cuda_and_profiler_status_is_irrelevant():
    passed = tuple(SanitizerResult(plan, "succeeded", "none") for plan in SanitizerPlan)
    qualification = CudaWinnerQualification(
        backend="cuda",
        correctness_passed=True,
        events_gate_passed=True,
        sanitizer_results=passed,
        nsys_status="unavailable",
        ncu_status="permission_denied",
    )
    assert qualification.qualified
    assert not CudaWinnerQualification(
        backend="cuda",
        correctness_passed=True,
        events_gate_passed=True,
        sanitizer_results=passed[:-1],
    ).qualified
    assert not CudaWinnerQualification(
        backend="triton",
        correctness_passed=True,
        events_gate_passed=True,
        sanitizer_results=passed,
    ).qualified


def test_profiler_replay_capsule_is_digest_bound_to_candidate_task_and_cases():
    task = _task()
    package = validate_candidate_package(
        build_candidate_package(
            CUDA,
            _plan(),
            task=task,
            provenance=CandidateProvenance(generator="replay-test"),
        ),
        task=task,
    )
    capsule = build_candidate_capsule(
        package, module=b"ELF-cubin", target_arch="sm_86", compiler_id="nvcc:test"
    )
    cases = build_development_case_bundle(task)
    replay = build_profiler_replay_capsule(
        capsule,
        task_payload=task.canonical_bytes(),
        case_payload=cases.canonical_bytes(),
    )
    validated = validate_profiler_replay_capsule(replay)
    assert validated.manifest.candidate_capsule_digest == hashlib.sha256(capsule).hexdigest()
    assert validated.task == task
    assert validated.cases == cases


def test_core10_cuda_packages_cover_both_directions_and_triton_fails_explicitly():
    tasks = core10_task_specs()
    assert len(
        [
            validate_candidate_package(build_fixed_cuda_candidate(task), task=task)
            for task in tasks
        ]
    ) == 20
    triton = [
        task
        for task in tasks
        if task.direction.value == "forward" and "triton" in task.candidate_backends
    ]
    assert {task.id.split(".")[2] for task in triton} == {"007", "026", "036", "047"}
    for task in triton:
        validate_candidate_package(build_fixed_triton_candidate(task), task=task)
    with pytest.raises(BackendUnsupportedError, match="backend_unsupported"):
        build_fixed_triton_candidate(next(task for task in tasks if task.id.endswith("019.forward")))
