from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from src.kernelblaster.candidate_packages import (
    build_candidate_capsule,
    build_fixed_cuda_candidate,
    build_fixed_triton_candidate,
    validate_candidate_capsule,
    validate_candidate_package,
)
from src.kernelblaster.candidate_packages.archive import archive_files
from src.kernelblaster.candidate_packages.replay import CapsuleCandidate
from src.kernelblaster.harness import (
    CaseBundle,
    CorrectnessHarness,
    PyTorchAutogradAdapter,
    build_development_case_bundle,
    core10_task_specs,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gpu
@pytest.mark.parametrize("number", ("007", "026", "036", "047"))
def test_fixed_triton_package_aot_replays_without_candidate_python(
    number: str, tmp_path: Path
) -> None:
    task = next(
        item
        for item in core10_task_specs()
        if item.id.endswith(f"{number}.forward")
    )
    package_payload = build_fixed_triton_candidate(task)
    package = validate_candidate_package(package_payload, task=task)
    files = archive_files(package_payload)
    source = tmp_path / "candidate.py"
    plan = tmp_path / "launch-plan.json"
    task_path = tmp_path / "task-spec.json"
    module = tmp_path / "module.cubin"
    source.write_bytes(files["candidate.py"])
    plan.write_bytes(files["launch-plan.json"])
    task_path.write_bytes(task.canonical_bytes())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "triton_aot_compile.py"),
            "--source",
            str(source),
            "--launch-plan",
            str(plan),
            "--task",
            str(task_path),
            "--target-arch",
            "sm_86",
            "--output",
            str(module),
        ],
        check=True,
        timeout=180,
    )
    capsule = build_candidate_capsule(
        package,
        module=module.read_bytes(),
        target_arch="sm_86",
        compiler_id="triton:gpu-test",
    )
    capsule_path = tmp_path / "candidate.capsule"
    capsule_path.write_bytes(capsule)

    # The odd disclosed fixture is small but non-zero and exercises masking.
    full = build_development_case_bundle(task)
    cases = CaseBundle(task_spec_digest=full.task_spec_digest, cases=(full.cases[2],))
    validated = validate_candidate_capsule(capsule)
    result = CorrectnessHarness(device="cuda").evaluate(
        task,
        cases,
        adapter=PyTorchAutogradAdapter(),
        candidate=CapsuleCandidate(validated, task, module),
    )
    assert result.passed, result.model_dump(mode="json")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "task_id", [task.id for task in core10_task_specs()]
)
def test_core10_cuda_device_module_replays_through_the_generic_harness(
    task_id: str, tmp_path: Path
) -> None:
    task = next(item for item in core10_task_specs() if item.id == task_id)
    package_payload = build_fixed_cuda_candidate(task)
    package = validate_candidate_package(package_payload, task=task)
    files = archive_files(package_payload)
    source = tmp_path / "candidate.cu"
    module = tmp_path / "module.cubin"
    source.write_bytes(files["candidate.cu"])
    subprocess.run(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "--cubin",
            "-arch=sm_86",
            str(source),
            "-o",
            str(module),
        ],
        check=True,
        timeout=180,
    )
    capsule = build_candidate_capsule(
        package,
        module=module.read_bytes(),
        target_arch="sm_86",
        compiler_id="nvcc:gpu-test",
    )
    full = build_development_case_bundle(task)
    cases = CaseBundle(task_spec_digest=full.task_spec_digest, cases=(full.cases[2],))
    result = CorrectnessHarness(device="cuda").evaluate(
        task,
        cases,
        adapter=PyTorchAutogradAdapter(),
        candidate=CapsuleCandidate(
            validate_candidate_capsule(capsule), task, module
        ),
    )
    assert result.passed, result.model_dump(mode="json")
