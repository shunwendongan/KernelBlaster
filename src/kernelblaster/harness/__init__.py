"""Versioned, operator-neutral contracts for trusted kernel evaluation."""

from .contracts import (
    AdapterKind,
    CacheMode,
    CaseBundle,
    CaseSpec,
    CaseTier,
    CorrectnessCaseResult,
    CorrectnessResultV2,
    DeterminismLevel,
    Direction,
    NumericsClass,
    ShapeDimension,
    TaskSpec,
    TensorError,
    TensorSpec,
    WorkloadSpec,
    parse_correctness_stdout,
)
from .core10 import CORE10_IDS, core10_task_specs
from .cuda_baselines import Core10NaiveCudaBackwardCandidate
from .cases import build_development_case_bundle
from .plugins import (
    AdapterPluginManifest,
    AdapterPluginAllowlist,
    AllowedAdapterPlugin,
    PluginAdapter,
    PluginFile,
    TrustedAdapterKey,
    TrustedAdapterKeys,
    build_signed_plugin,
    verify_signed_plugin,
    verify_allowlisted_plugin,
)
from .registry import AdapterDescriptor, AdapterRegistry, LegacyDriverAdapter
from .reference import PyTorchAutogradAdapter
from .runtime import CandidateRun, CorrectnessHarness, HarnessContext

__all__ = [
    "AdapterDescriptor",
    "AdapterKind",
    "AdapterPluginManifest",
    "AdapterPluginAllowlist",
    "AllowedAdapterPlugin",
    "AdapterRegistry",
    "CORE10_IDS",
    "CacheMode",
    "CaseBundle",
    "CaseSpec",
    "CaseTier",
    "CandidateRun",
    "CorrectnessCaseResult",
    "CorrectnessResultV2",
    "CorrectnessHarness",
    "Core10NaiveCudaBackwardCandidate",
    "HarnessContext",
    "DeterminismLevel",
    "Direction",
    "LegacyDriverAdapter",
    "NumericsClass",
    "PluginFile",
    "PluginAdapter",
    "PyTorchAutogradAdapter",
    "ShapeDimension",
    "TaskSpec",
    "TensorError",
    "TensorSpec",
    "TrustedAdapterKey",
    "TrustedAdapterKeys",
    "WorkloadSpec",
    "build_signed_plugin",
    "build_development_case_bundle",
    "core10_task_specs",
    "parse_correctness_stdout",
    "verify_signed_plugin",
    "verify_allowlisted_plugin",
]
