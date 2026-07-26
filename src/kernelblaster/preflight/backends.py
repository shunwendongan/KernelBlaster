# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed runtime backend selection for Agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ControlPlaneClient
from .contracts import AgentCapabilityMode, CapabilityReport, ExecutionBackend


@dataclass(frozen=True)
class RuntimeBackendBundle:
    execution_backend: ExecutionBackend
    capability_report: CapabilityReport | None = None
    control: ControlPlaneClient | None = None

    def create_events_backend(
        self,
        *,
        driver_path: Path,
        gpu: Any,
        logger: Any,
        work_dir: Path,
    ) -> Any:
        if self.execution_backend is ExecutionBackend.SANDBOX:
            raise RuntimeError(
                "sandbox CandidateEvaluator is required; trusted-local fallback is disabled"
            )
        from ..profiling import CudaEventsRunner, EventsProfilerBackend

        return EventsProfilerBackend(
            CudaEventsRunner(
                driver_path=driver_path,
                gpu=gpu,
                logger=logger,
                work_dir=work_dir,
            )
        )


def build_backend_bundle(
    *,
    requested: ExecutionBackend,
    report: CapabilityReport | None = None,
    control: ControlPlaneClient | None = None,
) -> RuntimeBackendBundle:
    if requested is ExecutionBackend.TRUSTED_LOCAL:
        return RuntimeBackendBundle(execution_backend=requested)
    if report is None or control is None:
        raise ValueError("sandbox backend requires a validated capability report and Control")
    if report.execution_backend is not ExecutionBackend.SANDBOX:
        raise ValueError("capability report does not authorize the sandbox backend")
    if report.agent_mode is AgentCapabilityMode.UNAVAILABLE:
        raise ValueError("capability report marks the Agent unavailable")
    return RuntimeBackendBundle(
        execution_backend=requested,
        capability_report=report,
        control=control,
    )


__all__ = ["RuntimeBackendBundle", "build_backend_bundle"]
