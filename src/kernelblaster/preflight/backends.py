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
        """为经典可信路径创建 Events backend。

        sandbox 候选不能携带或读取本地 ``driver_path``。它们必须由结构化
        CandidateEvaluator 通过 Control/GPU Job 契约执行，因此这里明确拒绝
        “安全后端失败后改用本地 Driver”的隐式回退。
        """
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
