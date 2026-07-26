#!/usr/bin/env python3
"""Run fixed valid and mutant Core 10 packages through the trusted Harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.harness import (  # noqa: E402
    CandidateRun,
    CaseBundle,
    CorrectnessHarness,
    Core10NaiveCudaBackwardCandidate,
    PyTorchAutogradAdapter,
    build_development_case_bundle,
    core10_task_specs,
)
from src.kernelblaster.harness.reference import torch_dtype  # noqa: E402
from src.kernelblaster.harness.reference import concrete_shape  # noqa: E402


def _candidate(task, adapter):
    outputs = {item.name: item for item in task.tensors if item.mutable}

    def run(inputs, scalars, context):
        return {
            name: value.to(torch_dtype(outputs[name].dtype))
            for name, value in adapter.reference(task, inputs, scalars).items()
        }

    return run


def _bounded(task, bundle: CaseBundle, limit: int) -> CaseBundle:
    cases = tuple(
        case
        for case in bundle.cases
        if max(
            __import__("math").prod(concrete_shape(tensor.shape, case.shape))
            for tensor in task.tensors
        )
        <= limit
    )
    if not cases:
        cases = (bundle.cases[0],)
    return CaseBundle(task_spec_digest=bundle.task_spec_digest, cases=cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tensor-elements", type=int, default=300 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backward-cuda",
        action="store_true",
        help="Use the audited naive CUDA sources instead of the oracle as backward fixtures",
    )
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    adapter = PyTorchAutogradAdapter()
    harness = CorrectnessHarness(device=args.device)
    evidence: list[dict[str, object]] = []
    for task in core10_task_specs():
        bundle = _bounded(
            task,
            build_development_case_bundle(task),
            args.max_tensor_elements,
        )
        candidate = (
            Core10NaiveCudaBackwardCandidate(task)
            if args.backward_cuda and task.direction.value == "backward"
            else _candidate(task, adapter)
        )
        result = harness.evaluate(
            task,
            bundle,
            adapter=adapter,
            candidate=candidate,
        )
        if not result.passed:
            raise RuntimeError(f"valid fixture failed: {task.id}")
        evidence.append(
            {
                "task_id": task.id,
                "task_spec_digest": task.canonical_sha256(),
                "case_bundle_digest": bundle.canonical_sha256(),
                "correctness_result_digest": hashlib.sha256(result.canonical_bytes()).hexdigest(),
                "case_count": result.case_count,
            }
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    mutation_task = next(task for task in core10_task_specs() if task.id.endswith("019.forward"))
    mutation_bundle = _bounded(mutation_task, build_development_case_bundle(mutation_task), 1024)

    def mutant(inputs, scalars, context):
        inputs["input"].add_(1)
        return CandidateRun(
            outputs={"output": torch.full_like(inputs["input"], float("nan"))},
            guard_canary_intact=False,
            output_poison_overwritten=False,
        )

    mutant_result = harness.evaluate(
        mutation_task,
        mutation_bundle,
        adapter=adapter,
        candidate=mutant,
    )
    if mutant_result.passed:
        raise RuntimeError("mutation fixture was not rejected")
    payload = {
        "schema_version": "core10-harness-smoke/v1",
        "device": args.device,
        "torch_version": torch.__version__,
        "hardware": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "compute_capability": (
            ".".join(str(item) for item in torch.cuda.get_device_capability(0))
            if args.device.startswith("cuda")
            else None
        ),
        "tasks": evidence,
        "mutant_violations": list(mutant_result.cases[0].violations),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
