"""Standalone-instance portability, migration, and aggregate-report helpers."""

from .contracts import (
    AGGREGATE_REPORT_SCHEMA,
    HARDWARE_IDENTITY_SCHEMA,
    INSTANCE_IDENTITY_SCHEMA,
    RUN_BUNDLE_SCHEMA,
    HardwareIdentity,
    InstanceIdentity,
    canonical_bytes,
    sha256,
)
from .identity import detect_hardware_identity, load_or_create_instance_identity, rotate_instance_identity
from .profile import load_profile
from .report import build_aggregate_report, write_aggregate_report
from .targets import SshTarget, load_targets, run_explicit_target

__all__ = [
    "AGGREGATE_REPORT_SCHEMA",
    "HARDWARE_IDENTITY_SCHEMA",
    "INSTANCE_IDENTITY_SCHEMA",
    "RUN_BUNDLE_SCHEMA",
    "HardwareIdentity",
    "InstanceIdentity",
    "canonical_bytes",
    "detect_hardware_identity",
    "load_or_create_instance_identity",
    "load_profile",
    "build_aggregate_report",
    "write_aggregate_report",
    "SshTarget",
    "load_targets",
    "run_explicit_target",
    "rotate_instance_identity",
    "sha256",
]
