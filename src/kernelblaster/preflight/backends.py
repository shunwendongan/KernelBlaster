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
    private_evaluation_profile_id: str | None = None
    benchmark_protocol_id: str = "generated-agent-v1"

    def create_candidate_evaluator(
        self,
        *,
        run_id: str,
        driver_path: Path,
        gpu: Any,
        logger: Any,
        work_dir: Path,
    ) -> Any:
        if self.execution_backend is ExecutionBackend.SANDBOX:
            from ..evaluation import SandboxCandidateEvaluator

            assert self.control is not None and self.capability_report is not None
            assert self.private_evaluation_profile_id is not None
            return SandboxCandidateEvaluator(
                self.control,
                self.capability_report,
                run_id=run_id,
                private_evaluation_profile_id=self.private_evaluation_profile_id,
                benchmark_protocol_id=self.benchmark_protocol_id,
            )
        from ..evaluation import TrustedLocalCandidateEvaluator
        from ..profiling import CudaEventsRunner, EventsProfilerBackend

        return TrustedLocalCandidateEvaluator(
            EventsProfilerBackend(
                CudaEventsRunner(
                    driver_path=driver_path,
                    gpu=gpu,
                    logger=logger,
                    work_dir=work_dir,
                )
            )
        )


def build_backend_bundle(
    *,
    requested: ExecutionBackend,
    report: CapabilityReport | None = None,
    control: ControlPlaneClient | None = None,
    private_evaluation_profile_id: str | None = None,
    benchmark_protocol_id: str = "generated-agent-v1",
) -> RuntimeBackendBundle:
    if requested is ExecutionBackend.TRUSTED_LOCAL:
        return RuntimeBackendBundle(execution_backend=requested)
    if report is None or control is None:
        raise ValueError("sandbox backend requires a validated capability report and Control")
    if report.execution_backend is not ExecutionBackend.SANDBOX:
        raise ValueError("capability report does not authorize the sandbox backend")
    if report.agent_mode is AgentCapabilityMode.UNAVAILABLE:
        raise ValueError("capability report marks the Agent unavailable")
    if not report.target_arch:
        raise ValueError("capability report has no target architecture")
    if not private_evaluation_profile_id:
        raise ValueError("sandbox backend requires a private evaluation profile ID")
    return RuntimeBackendBundle(
        execution_backend=requested,
        capability_report=report,
        control=control,
        private_evaluation_profile_id=private_evaluation_profile_id,
        benchmark_protocol_id=benchmark_protocol_id,
    )


__all__ = ["RuntimeBackendBundle", "build_backend_bundle"]
