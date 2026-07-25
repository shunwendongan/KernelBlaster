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
    probe_ncu_counters,
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
    profile, export, report = profile_commands(
        request,
        tmp_path / "candidate",
        tmp_path / "report",
        benchmark_protocol_id="trusted-smoke-v1",
    )
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
    assert profile[-5:] == [
        str(tmp_path / "candidate"),
        "--mode",
        "events",
        "--protocol",
        "trusted-smoke-v1",
    ]
    assert export == ["ncu", "--import", str(report), "--csv", "--page", "raw"]

    _profile, nsys_export, nsys_report = profile_commands(
        _request(plan_id="nsys_timeline_v1"),
        tmp_path / "candidate",
        tmp_path / "nsys-report",
        benchmark_protocol_id="trusted-smoke-v1",
    )
    assert nsys_export[-2:] == ["--force-export=true", str(nsys_report)]


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


def test_ncu_parser_accepts_recent_wide_raw_csv_and_normalizes_units():
    payload = (
        b'"Kernel Name","gpu__time_duration.sum",'
        b'"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
        b'"launch__occupancy_limit_blocks",'
        b'"sm__warps_active.avg.pct_of_peak_sustained_active"\n'
        b'"","us","%","block","%"\n'
        b'"vector_add_kernel(const float *, int)","7.84","2.79","16","49.81"\n'
    )

    kernel, metrics, reason = parse_profile_csv(
        ProfilePlanId.NCU_TRIAGE_V1, payload, "vector_add_kernel"
    )

    assert kernel == "vector_add_kernel(const float *, int)"
    assert reason is ProfileReasonCode.NONE
    assert {metric.name: metric.value for metric in metrics} == {
        "gpu_time": 7840.0,
        "sm_throughput": 2.79,
        "occupancy_limit_blocks": 16.0,
        "achieved_occupancy": 49.81,
    }
    assert {metric.name: metric.unit for metric in metrics} == {
        "gpu_time": "ns",
        "sm_throughput": "percent",
        "occupancy_limit_blocks": "count",
        "achieved_occupancy": "percent",
    }


def test_ncu_wide_parser_distinguishes_missing_kernel_from_empty_metrics():
    payload = (
        b'"Kernel Name","gpu__time_duration.sum"\n'
        b'"","us"\n'
        b'"other_kernel","7.84"\n'
    )
    assert parse_profile_csv(
        ProfilePlanId.NCU_TRIAGE_V1, payload, "vector_add_kernel"
    )[2] is ProfileReasonCode.KERNEL_NOT_FOUND

    empty_value = payload.replace(b'"other_kernel","7.84"', b'"vector_add_kernel",""')
    assert parse_profile_csv(
        ProfilePlanId.NCU_TRIAGE_V1, empty_value, "vector_add_kernel"
    )[2] is ProfileReasonCode.METRICS_EMPTY


def test_wsl_defaults_to_events_nsys_and_permission_blocked_ncu(monkeypatch):
    monkeypatch.setattr(worker_module, "_runtime_platform", lambda: "wsl")
    monkeypatch.setattr(worker_module.shutil, "which", lambda _tool: "/usr/bin/tool")
    monkeypatch.setattr(worker_module, "probe_ncu_counters", lambda: "permission_denied")
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


def test_auto_ncu_preflight_uses_a_fixed_bounded_probe(monkeypatch, tmp_path):
    executable = tmp_path / "ncu-preflight"
    executable.write_bytes(b"fixed-probe")
    monkeypatch.setenv("KERNELBLASTER_NCU_PREFLIGHT_BINARY", str(executable))
    monkeypatch.setattr(worker_module.shutil, "which", lambda _tool: "/usr/bin/ncu")

    def available(command, **kwargs):
        assert command[:3] == ["ncu", "--metrics", "gpu__time_duration.sum"]
        assert command[-1] == str(executable)
        assert kwargs["timeout"] == worker_module.NCU_PREFLIGHT_TIMEOUT
        report_base = command[command.index("--export") + 1]
        worker_module.Path(report_base).with_suffix(".ncu-rep").write_bytes(b"report")
        return worker_module.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(worker_module.subprocess, "run", available)
    assert probe_ncu_counters() == "available"

    def denied(command, **_kwargs):
        return worker_module.subprocess.CompletedProcess(
            command, 1, b"", b"ERR_NVGPUCTRPERM"
        )

    monkeypatch.setattr(worker_module.subprocess, "run", denied)
    assert probe_ncu_counters() == "permission_denied"


@pytest.mark.asyncio
async def test_wsl_nsys_empty_gpu_rows_use_official_timestamp_config(
    monkeypatch, tmp_path
):
    request = _request(plan_id="nsys_timeline_v1")
    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"candidate")
    calls = 0

    async def run(command, *, root, timeout):
        nonlocal calls
        calls += 1
        assert timeout > 0
        if command[1] == "profile":
            report_base = worker_module.Path(command[command.index("--output") + 1])
            report_base.with_suffix(".nsys-rep").write_bytes(b"raw-report")
            if calls > 2:
                config = (
                    root
                    / ".config"
                    / "NVIDIA Corporation"
                    / "nsys-config.ini"
                )
                assert config.read_text(encoding="utf-8") == (
                    "CuptiUseRawGpuTimestamps=false\n"
                )
            return 0, b"", b"", False
        if calls == 2:
            return 1, b"", b"empty CUDA kernel report", False
        return (
            0,
            b"Name,Total Time (ns),Avg (ns),Instances\n"
            b"vector_add_kernel,2105,2105,1\n",
            b"",
            False,
        )

    monkeypatch.setattr(worker_module, "_run_process", run)
    monkeypatch.setattr(worker_module, "_runtime_platform", lambda: "wsl")
    monkeypatch.setattr(
        worker_module.subprocess,
        "run",
        lambda command, **_kwargs: worker_module.subprocess.CompletedProcess(
            command, 0, "NVIDIA Nsight Systems version test", ""
        ),
    )
    runner = worker_module.FixedToolRunner()
    execution = await runner.run(
        request,
        candidate,
        "trusted-smoke-v1",
        tmp_path,
        30,
    )
    assert execution.returncode == 0
    assert execution.used_timestamp_workaround is True
    assert b"vector_add_kernel" in execution.csv_output


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
        return self.executable, "b" * 64, "trusted-smoke-v1"

    async def upload(self, payload, **_kwargs):
        self.uploads.append(payload)
        return {"digest": hashlib.sha256(payload).hexdigest()}


class _Runner:
    def __init__(self, execution: ToolExecution):
        self.execution = execution

    async def run(self, request, executable, benchmark_protocol_id, root, timeout):
        assert executable.read_bytes()
        assert benchmark_protocol_id == "trusted-smoke-v1"
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
        async def run(self, request, executable, benchmark_protocol_id, root, timeout):
            nonlocal active, maximum
            assert benchmark_protocol_id == "trusted-smoke-v1"
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return _execution()

    profiler = FixedPlanProfiler(_Control(executable), capabilities=_capabilities(), runner=Runner())
    await asyncio.gather(profiler.profile(request), profiler.profile(request))
    assert maximum == 1
