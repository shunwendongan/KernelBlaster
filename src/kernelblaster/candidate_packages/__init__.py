"""Generated v2 device-only candidate contracts and immutable capsules."""

from .compiler import AotCompilation, compile_aot, fixed_compile_command
from .cuda_fixtures import build_fixed_cuda_candidate
from .contracts import (
    CandidateBackend,
    CandidateCapsuleManifest,
    CandidateLaunchPlan,
    CandidateManifestV2,
    CandidateProfilerCapsuleManifest,
    CandidateProvenance,
    Dimensions,
    DispatchRule,
    KernelDeclaration,
    KernelLaunch,
    ResolvedLaunch,
)
from .package import (
    BackendUnsupportedError,
    ValidatedCandidateCapsule,
    ValidatedCandidatePackage,
    ValidatedProfilerReplayCapsule,
    build_candidate_capsule,
    build_candidate_package,
    build_profiler_replay_capsule,
    validate_candidate_capsule,
    validate_candidate_package,
    validate_profiler_replay_capsule,
    validate_source,
)
from .qualification import (
    CudaWinnerQualification,
    SanitizerPlan,
    SanitizerResult,
    run_sanitizer,
    sanitizer_command,
)
from .triton_fixtures import TRITON_AOT_TASK_IDS, build_fixed_triton_candidate

__all__ = [
    "AotCompilation",
    "BackendUnsupportedError",
    "CandidateBackend",
    "CandidateCapsuleManifest",
    "CandidateLaunchPlan",
    "CandidateManifestV2",
    "CandidateProfilerCapsuleManifest",
    "CandidateProvenance",
    "CudaWinnerQualification",
    "Dimensions",
    "DispatchRule",
    "KernelDeclaration",
    "KernelLaunch",
    "ResolvedLaunch",
    "SanitizerPlan",
    "SanitizerResult",
    "TRITON_AOT_TASK_IDS",
    "ValidatedCandidateCapsule",
    "ValidatedCandidatePackage",
    "ValidatedProfilerReplayCapsule",
    "build_candidate_capsule",
    "build_candidate_package",
    "build_fixed_cuda_candidate",
    "build_fixed_triton_candidate",
    "build_profiler_replay_capsule",
    "compile_aot",
    "fixed_compile_command",
    "run_sanitizer",
    "sanitizer_command",
    "validate_candidate_capsule",
    "validate_candidate_package",
    "validate_profiler_replay_capsule",
    "validate_source",
]
