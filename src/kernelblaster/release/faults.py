"""Bounded, opt-in release fault plans; dangerous actions are never default."""

from __future__ import annotations

from typing import Any


_FAULTS = {
    "control_restart": {"dangerous": False, "expect": "control_recovers_without_duplicate_terminal_state"},
    "supervisor_restart": {"dangerous": False, "expect": "lease_recovers_once_and_smoke_passes"},
    "compile_error": {"dangerous": False, "expect": "compile_failed_and_smoke_passes"},
    "timeout": {"dangerous": False, "expect": "timed_out_and_smoke_passes"},
    "oom": {"dangerous": True, "expect": "gpu_oom_and_smoke_passes"},
    "illegal_access": {"dangerous": True, "expect": "gpu_recovery_failed_or_execution_failed_and_smoke_passes"},
    "log_flood": {"dangerous": False, "expect": "bounded_output_and_smoke_passes"},
    "cas_corruption": {"dangerous": False, "expect": "corruption_detected_before_use"},
    "lease_expiry": {"dangerous": False, "expect": "one_expiry_recovery_and_unique_attempt_history"},
    "nsys_empty": {"dangerous": False, "expect": "metrics_empty"},
    "ncu_permission_denied": {"dangerous": False, "expect": "events_only_permission_denied"},
    "ncu_tool_missing": {"dangerous": False, "expect": "events_only_tool_missing"},
    "disk_full": {"dangerous": True, "expect": "atomic_failure_and_retryable_cleanup"},
    "transfer_interrupted": {"dangerous": False, "expect": "no_partial_import_and_retryable_transfer"},
}


def fault_plan(*, allow_real_faults: bool = False) -> list[dict[str, Any]]:
    """List the release fault matrix; every action is simulated by default."""
    plan = []
    for name, spec in _FAULTS.items():
        plan.append({
            "name": name,
            "mode": "real" if allow_real_faults else "simulated",
            "dangerous": spec["dangerous"],
            "expected": spec["expect"],
        })
    return plan
