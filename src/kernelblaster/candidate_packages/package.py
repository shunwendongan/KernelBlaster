"""Deterministic CandidatePackage/capsule construction and hostile-source rejection."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import re

from ..harness.contracts import CaseBundle, TaskSpec
from .archive import archive_files, build_archive
from .contracts import (
    CandidateBackend,
    CandidateCapsuleManifest,
    CandidateLaunchPlan,
    CandidateManifestV2,
    CandidateProfilerCapsuleManifest,
    CandidateProvenance,
)


_CUDA_KERNEL = re.compile(
    r'extern\s+"C"\s+__global__\s+(?:[A-Za-z_][A-Za-z0-9_:<>,*&]*\s+)+'
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_CUDA_FORBIDDEN = (
    r"\bmain\s*\(",
    r"\blaunch_gpu_implementation\s*\(",
    r"\b__host__\b",
    r"__attribute__\s*\(\s*\(\s*constructor",
    r"#\s*include\s*[<\"](?:cublas|cudnn|cutlass)",
    r"\b(?:cublas|cudnn|cutlass)[A-Za-z0-9_]*\s*\(",
    r"\bcuda(?:Malloc|Free|StreamCreate|StreamDestroy|DeviceReset)\s*\(",
    r"\b(?:fopen|open|system|popen|fork|exec[lvpe]*|socket|connect)\s*\(",
    r"\b(?:asm|__asm__)\b",
)
_CUDA_ALLOWED_INCLUDES = {
    "cuda_fp16.h",
    "cuda_bf16.h",
    "float.h",
    "math.h",
    "math_constants.h",
    "stdint.h",
}


@dataclass(frozen=True)
class ValidatedCandidatePackage:
    digest: str
    manifest: CandidateManifestV2
    launch_plan: CandidateLaunchPlan
    source: bytes


@dataclass(frozen=True)
class ValidatedCandidateCapsule:
    digest: str
    manifest: CandidateCapsuleManifest
    launch_plan: CandidateLaunchPlan
    module: bytes


@dataclass(frozen=True)
class ValidatedProfilerReplayCapsule:
    digest: str
    manifest: CandidateProfilerCapsuleManifest
    candidate: ValidatedCandidateCapsule
    task: TaskSpec
    cases: CaseBundle


def _validate_plan_for_task(plan: CandidateLaunchPlan, task: TaskSpec) -> None:
    if plan.task_spec_digest != task.canonical_sha256():
        raise ValueError("launch plan TaskSpec binding mismatch")
    if set(plan.shape_symbols) != {item.name for item in task.dimensions}:
        raise ValueError("launch plan shape symbols do not match TaskSpec")
    if set(plan.tensor_bindings) != {item.name for item in task.tensors}:
        raise ValueError("launch plan tensor bindings do not match TaskSpec")
    if set(plan.scalar_bindings) != {item.name for item in task.scalars}:
        raise ValueError("launch plan scalar bindings do not match TaskSpec")
    if plan.workspace_bytes > task.workspace.maximum_bytes:
        raise ValueError("candidate workspace exceeds TaskSpec")
    if plan.backend.value not in task.candidate_backends:
        raise BackendUnsupportedError("backend_unsupported")
    if plan.backend is CandidateBackend.TRITON and len(plan.kernels) != 1:
        raise BackendUnsupportedError("Triton AOT v1 supports one kernel per package")
    if plan.backend is CandidateBackend.TRITON:
        blocks = {
            (launch.block.x, launch.block.y, launch.block.z)
            for rule in plan.dispatch
            for launch in rule.launches
        }
        if len(blocks) != 1:
            raise ValueError("Triton AOT requires one fixed block configuration")
        block_x, block_y, block_z = next(iter(blocks))
        if block_y != "1" or block_z != "1" or block_x not in {
            "32",
            "64",
            "128",
            "256",
        }:
            raise ValueError("Triton AOT block size must be a fixed whole-warp value")
        if any(
            launch.dynamic_shared_bytes != "0"
            for rule in plan.dispatch
            for launch in rule.launches
        ):
            raise ValueError("Triton AOT owns dynamic shared-memory allocation")
    if task.stream.graph_capture == "required" and plan.cuda_graph != "required":
        raise ValueError("TaskSpec requires CUDA Graph replay")
    if task.stream.graph_capture == "unsupported" and plan.cuda_graph != "disabled":
        raise ValueError("TaskSpec forbids CUDA Graph replay")


class BackendUnsupportedError(ValueError):
    """A valid TaskSpec explicitly excludes the requested candidate backend."""


def _validate_cuda_source(source: str, plan: CandidateLaunchPlan) -> None:
    for pattern in _CUDA_FORBIDDEN:
        if re.search(pattern, source, flags=re.IGNORECASE):
            raise ValueError("CUDA candidate contains forbidden host or vendor-library code")
    includes = re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", source)
    if any(header not in _CUDA_ALLOWED_INCLUDES for header in includes):
        raise ValueError("CUDA candidate includes a non-allowlisted header")
    if re.search(r"#\s*(?:include_next|import|line|pragma)\b", source):
        raise ValueError("CUDA candidate uses a forbidden preprocessor directive")
    declared = {item.name for item in plan.kernels}
    defined = set(_CUDA_KERNEL.findall(source))
    if defined != declared:
        raise ValueError("CUDA source kernels do not exactly match the launch plan")


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _validate_triton_source(source: str, plan: CandidateLaunchPlan) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Triton candidate syntax is invalid") from error
    imported_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            if any(name not in {"triton", "triton.language"} for name in names):
                raise ValueError("Triton candidate imports a forbidden module")
            imported_aliases.update(item.asname or item.name.split(".")[0] for item in node.names)
        if isinstance(node, ast.Call):
            root = _root_name(node.func)
            is_tensor_cast = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and isinstance(node.func.value, ast.Call)
                and _root_name(node.func.value.func) in imported_aliases | {"tl"}
            )
            if (
                root not in imported_aliases | {"tl", "triton", "range"}
                and not is_tensor_cast
            ):
                raise ValueError("Triton candidate calls forbidden Python code")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Triton candidate uses a forbidden dunder attribute")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            raise ValueError("Triton candidate has executable top-level Python")
    kernels = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "jit"
            and _root_name(decorator) in imported_aliases | {"triton"}
            for decorator in node.decorator_list
        )
    }
    if kernels != {item.name for item in plan.kernels}:
        raise ValueError("Triton kernels do not exactly match the launch plan")


def validate_source(source: bytes, plan: CandidateLaunchPlan) -> None:
    if not source or len(source) > 2 * 1024 * 1024 or b"\x00" in source:
        raise ValueError("candidate source size or encoding is invalid")
    text = source.decode("utf-8", errors="strict")
    if plan.backend is CandidateBackend.CUDA:
        _validate_cuda_source(text, plan)
    else:
        _validate_triton_source(text, plan)


def build_candidate_package(
    source: bytes,
    launch_plan: CandidateLaunchPlan,
    *,
    task: TaskSpec,
    provenance: CandidateProvenance,
) -> bytes:
    _validate_plan_for_task(launch_plan, task)
    validate_source(source, launch_plan)
    source_path = (
        "candidate.cu" if launch_plan.backend is CandidateBackend.CUDA else "candidate.py"
    )
    manifest = CandidateManifestV2(
        task_spec_digest=task.canonical_sha256(),
        backend=launch_plan.backend,
        source_path=source_path,
        source_digest=hashlib.sha256(source).hexdigest(),
        launch_plan_digest=launch_plan.canonical_sha256(),
        provenance=provenance,
    )
    return build_archive(
        {
            "candidate-manifest.json": manifest.canonical_bytes(),
            "launch-plan.json": launch_plan.canonical_bytes(),
            source_path: source,
        }
    )


def validate_candidate_package(payload: bytes, *, task: TaskSpec) -> ValidatedCandidatePackage:
    files = archive_files(payload)
    if set(files) not in (
        {"candidate-manifest.json", "launch-plan.json", "candidate.cu"},
        {"candidate-manifest.json", "launch-plan.json", "candidate.py"},
    ):
        raise ValueError("CandidatePackage contains non-allowlisted files")
    manifest = CandidateManifestV2.model_validate_json(files["candidate-manifest.json"])
    plan = CandidateLaunchPlan.model_validate_json(files["launch-plan.json"])
    if manifest.canonical_bytes() != files["candidate-manifest.json"]:
        raise ValueError("candidate manifest is not canonical")
    if plan.canonical_bytes() != files["launch-plan.json"]:
        raise ValueError("launch plan is not canonical")
    source = files.get(manifest.source_path)
    if source is None or hashlib.sha256(source).hexdigest() != manifest.source_digest:
        raise ValueError("candidate source digest mismatch")
    if plan.canonical_sha256() != manifest.launch_plan_digest:
        raise ValueError("candidate launch-plan digest mismatch")
    if manifest.task_spec_digest != task.canonical_sha256() or plan.task_spec_digest != task.canonical_sha256():
        raise ValueError("CandidatePackage TaskSpec binding mismatch")
    if manifest.backend is not plan.backend:
        raise ValueError("candidate backend mismatch")
    _validate_plan_for_task(plan, task)
    validate_source(source, plan)
    return ValidatedCandidatePackage(
        digest=hashlib.sha256(payload).hexdigest(),
        manifest=manifest,
        launch_plan=plan,
        source=source,
    )


def build_candidate_capsule(
    package: ValidatedCandidatePackage,
    *,
    module: bytes,
    target_arch: str,
    compiler_id: str,
) -> bytes:
    if not module:
        raise ValueError("compiled device module is empty")
    manifest = CandidateCapsuleManifest(
        candidate_package_digest=package.digest,
        task_spec_digest=package.manifest.task_spec_digest,
        launch_plan_digest=package.launch_plan.canonical_sha256(),
        module_digest=hashlib.sha256(module).hexdigest(),
        backend=package.manifest.backend,
        target_arch=target_arch,
        compiler_id=compiler_id,
    )
    return build_archive(
        {
            "capsule-manifest.json": manifest.canonical_bytes(),
            "launch-plan.json": package.launch_plan.canonical_bytes(),
            "module.cubin": module,
        }
    )


def validate_candidate_capsule(payload: bytes) -> ValidatedCandidateCapsule:
    files = archive_files(payload)
    if set(files) != {"capsule-manifest.json", "launch-plan.json", "module.cubin"}:
        raise ValueError("candidate capsule contains non-allowlisted files")
    manifest = CandidateCapsuleManifest.model_validate_json(files["capsule-manifest.json"])
    plan = CandidateLaunchPlan.model_validate_json(files["launch-plan.json"])
    if manifest.canonical_bytes() != files["capsule-manifest.json"]:
        raise ValueError("candidate capsule manifest is not canonical")
    if plan.canonical_bytes() != files["launch-plan.json"]:
        raise ValueError("candidate capsule launch plan is not canonical")
    module = files["module.cubin"]
    if hashlib.sha256(module).hexdigest() != manifest.module_digest:
        raise ValueError("candidate capsule module digest mismatch")
    if plan.canonical_sha256() != manifest.launch_plan_digest:
        raise ValueError("candidate capsule launch-plan digest mismatch")
    if plan.backend is not manifest.backend or plan.task_spec_digest != manifest.task_spec_digest:
        raise ValueError("candidate capsule contract binding mismatch")
    return ValidatedCandidateCapsule(
        digest=hashlib.sha256(payload).hexdigest(),
        manifest=manifest,
        launch_plan=plan,
        module=module,
    )


def build_profiler_replay_capsule(
    candidate_payload: bytes,
    *,
    task_payload: bytes,
    case_payload: bytes,
) -> bytes:
    """Create the only generated-v2 artifact the Profiler may execute."""
    candidate = validate_candidate_capsule(candidate_payload)
    task = TaskSpec.model_validate_json(task_payload)
    cases = CaseBundle.model_validate_json(case_payload)
    if task.canonical_bytes() != task_payload or cases.canonical_bytes() != case_payload:
        raise ValueError("profiler replay TaskSpec and cases must be canonical")
    if candidate.manifest.task_spec_digest != task.canonical_sha256():
        raise ValueError("profiler replay TaskSpec binding mismatch")
    cases.validate_for(task)
    manifest = CandidateProfilerCapsuleManifest(
        candidate_capsule_digest=hashlib.sha256(candidate_payload).hexdigest(),
        task_spec_digest=task.canonical_sha256(),
        case_bundle_digest=cases.canonical_sha256(),
    )
    return build_archive(
        {
            "replay-manifest.json": manifest.canonical_bytes(),
            "candidate.capsule": candidate_payload,
            "task-spec.json": task_payload,
            "case-bundle.json": case_payload,
        }
    )


def validate_profiler_replay_capsule(payload: bytes) -> ValidatedProfilerReplayCapsule:
    files = archive_files(payload)
    if set(files) != {
        "replay-manifest.json",
        "candidate.capsule",
        "task-spec.json",
        "case-bundle.json",
    }:
        raise ValueError("profiler replay capsule contains non-allowlisted files")
    manifest = CandidateProfilerCapsuleManifest.model_validate_json(
        files["replay-manifest.json"]
    )
    if manifest.canonical_bytes() != files["replay-manifest.json"]:
        raise ValueError("profiler replay manifest is not canonical")
    candidate = validate_candidate_capsule(files["candidate.capsule"])
    task = TaskSpec.model_validate_json(files["task-spec.json"])
    cases = CaseBundle.model_validate_json(files["case-bundle.json"])
    if task.canonical_bytes() != files["task-spec.json"]:
        raise ValueError("profiler replay TaskSpec is not canonical")
    if cases.canonical_bytes() != files["case-bundle.json"]:
        raise ValueError("profiler replay case bundle is not canonical")
    if hashlib.sha256(files["candidate.capsule"]).hexdigest() != manifest.candidate_capsule_digest:
        raise ValueError("profiler replay candidate digest mismatch")
    if task.canonical_sha256() != manifest.task_spec_digest:
        raise ValueError("profiler replay TaskSpec digest mismatch")
    if cases.canonical_sha256() != manifest.case_bundle_digest:
        raise ValueError("profiler replay case digest mismatch")
    if candidate.manifest.task_spec_digest != manifest.task_spec_digest:
        raise ValueError("profiler replay candidate TaskSpec binding mismatch")
    cases.validate_for(task)
    return ValidatedProfilerReplayCapsule(
        digest=hashlib.sha256(payload).hexdigest(),
        manifest=manifest,
        candidate=candidate,
        task=task,
        cases=cases,
    )


__all__ = [
    "BackendUnsupportedError",
    "ValidatedCandidateCapsule",
    "ValidatedCandidatePackage",
    "ValidatedProfilerReplayCapsule",
    "build_candidate_capsule",
    "build_candidate_package",
    "build_profiler_replay_capsule",
    "validate_candidate_capsule",
    "validate_candidate_package",
    "validate_profiler_replay_capsule",
    "validate_source",
]
