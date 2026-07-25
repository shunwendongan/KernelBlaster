# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import pytest
from pydantic import ValidationError

from src.kernelblaster.profiler_jobs.contracts import (
    ProfilePlanId,
    ProfileReasonCode,
    ProfileRequest,
    ProfileStatus,
    ProfilerCapabilities,
    public_profile_feedback,
)
from src.kernelblaster.profiler_jobs.worker import (
    FixedPlanProfiler,
    ToolExecution,
    parse_profile_csv,
    profile_commands,
)
import src.kernelblaster.profiler_jobs.worker as worker_module


def _request(**updates) -> ProfileRequest:
    payload = {
        "artifact_digest": "a" * 64,
        "plan_id": "ncu_triage_v1",
        "kernel_filter": "vector_add_kernel",
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    payload.update(updates)
    return ProfileRequest.model_validate(payload)


def _capabilities() -> ProfilerCapabilities:
    return ProfilerCapabilities(
        platform="linux",
        nsys_status="available",
        ncu_status="available",
        supported_plans=tuple(ProfilePlanId),
        automatic_execution=True,
    )


def test_request_rejects_arbitrary_executable_argv_paths_and_environment():
    for field, value in (
        ("executable", "/tmp/candidate"),
        ("argv", ["--set", "full"]),
        ("output_path", "../../escape"),
        ("environment", {"TOKEN": "secret"}),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _request(**{field: value})
    with pytest.raises(ValidationError, match="unsupported characters"):
        _request(kernel_filter="--import /etc/passwd")


def test_fixed_plan_commands_do_not_accept_caller_argv_or_output_paths(tmp_path):
    request = _request()
    profile, export, report = profile_commands(request, tmp_path / "candidate", tmp_path / "report")
    assert profile[:7] == [
        "ncu",
        "--section",
        "SpeedOfLight",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
    ]
    assert profile[profile.index("--kernel-name") + 1] == "regex:vector_add_kernel"
    assert profile[-3:] == [str(tmp_path / "candidate"), "--mode", "events"]
    assert export == ["ncu", "--import", str(report), "--csv", "--page", "raw"]


def test_parsers_distinguish_empty_metrics_and_missing_target_kernel():
    kernel, metrics, reason = parse_profile_csv(
        ProfilePlanId.NSYS_TIMELINE_V1, b"", "vector_add_kernel"
    )
    assert (kernel, metrics, reason) == ("", (), ProfileReasonCode.METRICS_EMPTY)
    payload = (
        b'Name,Total Time (ns),Avg (ns),Instances\n'
        b'other_kernel,1000,100,10\n'
    )
    assert parse_profile_csv(
        ProfilePlanId.NSYS_TIMELINE_V1, payload, "vector_add_kernel"
    )[2] is ProfileReasonCode.KERNEL_NOT_FOUND


def test_wsl_defaults_to_events_nsys_and_permission_blocked_ncu(monkeypatch):
    monkeypatch.setattr(worker_module, "_runtime_platform", lambda: "wsl")
    monkeypatch.setattr(worker_module.shutil, "which", lambda _tool: "/usr/bin/tool")
    monkeypatch.setenv("KERNELBLASTER_NCU_PREFLIGHT_STATUS", "auto")
    capabilities = worker_module.detect_capabilities()
    assert capabilities.platform == "wsl"
    assert capabilities.cuda_events is True
    assert capabilities.nsys_status == "available"
    assert capabilities.ncu_status == "permission_denied"
    assert capabilities.supported_plans == (ProfilePlanId.NSYS_TIMELINE_V1,)


def test_autodl_linux_enables_ncu_plans_after_preflight(monkeypatch):
    monkeypatch.setattr(worker_module, "_runtime_platform", lambda: "linux")
    monkeypatch.setattr(worker_module.shutil, "which", lambda _tool: "/usr/bin/tool")
    monkeypatch.setenv("KERNELBLASTER_NCU_PREFLIGHT_STATUS", "available")
    capabilities = worker_module.detect_capabilities()
    assert capabilities.ncu_status == "available"
    assert set(capabilities.supported_plans) == set(ProfilePlanId)


@pytest.mark.asyncio
async def test_tool_missing_and_windows_unsupported_are_distinct_blockers():
    executable = b"candidate"
    request = _request(artifact_digest=hashlib.sha256(executable).hexdigest())
    missing = _capabilities().model_copy(
        update={"ncu_status": "tool_missing", "supported_plans": ()}
    )
    result = await FixedPlanProfiler(_Control(executable), capabilities=missing).profile(request)
    assert result.reason_code is ProfileReasonCode.TOOL_MISSING
    windows = _capabilities().model_copy(
        update={
            "platform": "windows",
            "ncu_status": "unsupported",
            "nsys_status": "unsupported",
            "supported_plans": (),
            "automatic_execution": False,
        }
    )
    result = await FixedPlanProfiler(_Control(executable), capabilities=windows).profile(request)
    assert result.reason_code is ProfileReasonCode.UNSUPPORTED


class _Control:
    def __init__(self, executable: bytes):
        self.executable = executable
        self.uploads: list[bytes] = []

    async def download(self, digest):
        assert hashlib.sha256(self.executable).hexdigest() == digest
        return self.executable, "b" * 64

    async def upload(self, payload, **_kwargs):
        self.uploads.append(payload)
        return {"digest": hashlib.sha256(payload).hexdigest()}


class _Runner:
    def __init__(self, execution: ToolExecution):
        self.execution = execution

    async def run(self, request, executable, root, timeout):
        assert executable.read_bytes()
        return self.execution


def _execution(*, stderr: bytes = b"", csv_output: bytes | None = None) -> ToolExecution:
    csv_output = csv_output if csv_output is not None else (
        b'ID,Process ID,Process Name,Host Name,Kernel Name,Context,Stream,Section Name,Metric Name,Metric Unit,Metric Value\n'
        b'1,1,candidate,host,vector_add_kernel,1,1,SpeedOfLight,gpu__time_duration.sum,nsecond,1234\n'
    )
    return ToolExecution(
        tool="ncu",
        version="NVIDIA Nsight Compute 2026.1",
        returncode=1 if stderr else 0,
        stdout=b"raw profiler stdout",
        stderr=stderr,
        csv_output=csv_output,
        report=b"raw-report",
    )


@pytest.mark.asyncio
async def test_permission_failure_is_separate_and_raw_stdout_is_not_prompt_feedback():
    executable = b"candidate"
    request = _request(artifact_digest=hashlib.sha256(executable).hexdigest())
    worker = FixedPlanProfiler(
        _Control(executable),
        capabilities=_capabilities(),
        runner=_Runner(_execution(stderr=b"ERR_NVGPUCTRPERM")),
    )
    result = await worker.profile(request)
    assert result.status is ProfileStatus.BLOCKED
    assert result.reason_code is ProfileReasonCode.PERMISSION_DENIED
    assert "raw profiler stdout" not in str(public_profile_feedback(result))


@pytest.mark.asyncio
async def test_success_has_whitelisted_units_provenance_and_diagnostic_only_summary():
    executable = b"candidate"
    request = _request(artifact_digest=hashlib.sha256(executable).hexdigest())
    control = _Control(executable)
    result = await FixedPlanProfiler(
        control, capabilities=_capabilities(), runner=_Runner(_execution())
    ).profile(request)
    assert result.status is ProfileStatus.SUCCEEDED
    assert result.summary is not None
    assert result.summary.diagnostic_only is True
    assert result.summary.ranking_source == "cuda_events"
    assert result.summary.metrics[0].name == "gpu_time"
    assert result.summary.metrics[0].unit == "ns"
    assert result.provenance is not None
    assert result.provenance.source_digest == "b" * 64
    assert set(result.artifact_roles.values()) == {
        "raw_report",
        "raw_csv",
        "tool_log",
        "structured_summary",
    }


@pytest.mark.asyncio
async def test_ncu_profiles_are_strictly_single_session():
    executable = b"candidate"
    request = _request(artifact_digest=hashlib.sha256(executable).hexdigest())
    active = 0
    maximum = 0

    class Runner:
        async def run(self, request, executable, root, timeout):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return _execution()

    profiler = FixedPlanProfiler(_Control(executable), capabilities=_capabilities(), runner=Runner())
    await asyncio.gather(profiler.profile(request), profiler.profile(request))
    assert maximum == 1
