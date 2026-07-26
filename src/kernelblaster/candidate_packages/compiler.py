"""Fixed AOT compiler commands; candidate data never controls argv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from .contracts import CandidateBackend
from .package import ValidatedCandidatePackage
from ..harness.contracts import TaskSpec


_ARCH = re.compile(r"^sm_[0-9]{2,3}$")


@dataclass(frozen=True)
class AotCompilation:
    module: bytes
    stdout: bytes
    stderr: bytes
    compiler_id: str


def fixed_compile_command(
    package: ValidatedCandidatePackage,
    *,
    source_path: Path,
    output_path: Path,
    target_arch: str,
    task_path: Path | None = None,
) -> list[str]:
    if not _ARCH.fullmatch(target_arch):
        raise ValueError("target architecture is invalid")
    if package.manifest.backend is CandidateBackend.CUDA:
        return [
            "nvcc",
            "-O3",
            "-std=c++17",
            "--cubin",
            f"-arch={target_arch}",
            "--ptxas-options=-v",
            str(source_path),
            "-o",
            str(output_path),
        ]
    if task_path is None:
        raise ValueError("Triton AOT requires a trusted TaskSpec path")
    return [
        "python",
        "/opt/kernelblaster/scripts/triton_aot_compile.py",
        "--source",
        str(source_path),
        "--launch-plan",
        str(source_path.parent / "launch-plan.json"),
        "--task",
        str(task_path),
        "--target-arch",
        target_arch,
        "--output",
        str(output_path),
    ]


def compile_aot(
    package: ValidatedCandidatePackage,
    *,
    root: Path,
    target_arch: str,
    timeout_seconds: int = 180,
    task: TaskSpec | None = None,
) -> AotCompilation:
    source = root / package.manifest.source_path
    output = root / "module.cubin"
    source.write_bytes(package.source)
    (root / "launch-plan.json").write_bytes(package.launch_plan.canonical_bytes())
    task_path: Path | None = None
    if package.manifest.backend is CandidateBackend.TRITON:
        if task is None or task.canonical_sha256() != package.manifest.task_spec_digest:
            raise ValueError("Triton AOT requires the package-bound TaskSpec")
        task_path = root / "task-spec.json"
        task_path.write_bytes(task.canonical_bytes())
    command = fixed_compile_command(
        package,
        source_path=source,
        output_path=output,
        target_arch=target_arch,
        task_path=task_path,
    )
    result = subprocess.run(
        command,
        cwd=root,
        env={
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(root),
            "TMPDIR": str(root),
            "CUDA_VISIBLE_DEVICES": "0",
        },
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError("AOT compilation failed")
    version = subprocess.run(
        ["nvcc", "--version"] if package.manifest.backend is CandidateBackend.CUDA else ["python", "-c", "import triton; print(triton.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().splitlines()[-1]
    return AotCompilation(
        module=output.read_bytes(),
        stdout=result.stdout,
        stderr=result.stderr,
        compiler_id=f"{package.manifest.backend.value}:{version}"[:128],
    )


__all__ = ["AotCompilation", "compile_aot", "fixed_compile_command"]
