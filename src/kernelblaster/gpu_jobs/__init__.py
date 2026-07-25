"""Versioned, hardware-aware GPU Job contracts."""

from .bundles import build_deterministic_bundle, validate_bundle
from .capabilities import detect_gpu_capabilities
from .contracts import (
    GpuCapabilities,
    GpuJobManifest,
    GpuJobResult,
    GpuReasonCode,
    GpuJobStage,
    GpuJobStatus,
    ResourceLimits,
)

__all__ = [
    "GpuCapabilities",
    "GpuJobManifest",
    "GpuJobResult",
    "GpuReasonCode",
    "GpuJobStage",
    "GpuJobStatus",
    "ResourceLimits",
    "build_deterministic_bundle",
    "detect_gpu_capabilities",
    "validate_bundle",
]
