# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent, single-session Profiler Worker with fixed NSYS/NCU plans."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import tempfile
from typing import Any

from fastapi import Depends, FastAPI, Header
import uvicorn

from .client import ControlProfilerClient
from .contracts import (
    ProfileMetric,
    ProfilePlanId,
    ProfileProvenance,
    ProfileReasonCode,
    ProfileRequest,
    ProfileResult,
    ProfileStatus,
    ProfileSummary,
    ProfilerCapabilities,
)


PLAN_SECTIONS: dict[ProfilePlanId, tuple[str, ...]] = {
    ProfilePlanId.NCU_TRIAGE_V1: ("SpeedOfLight", "LaunchStats", "Occupancy"),
    ProfilePlanId.NCU_MEMORY_V1: ("MemoryWorkloadAnalysis",),
    ProfilePlanId.NCU_SCHEDULER_V1: ("SchedulerStats", "WarpStateStats"),
}
PLAN_TIMEOUTS = {
    ProfilePlanId.NSYS_TIMELINE_V1: 300,
    ProfilePlanId.NCU_TRIAGE_V1: 600,
    ProfilePlanId.NCU_MEMORY_V1: 600,
    ProfilePlanId.NCU_SCHEDULER_V1: 600,
}
OUTPUT_LIMIT = 8 * 1024 * 1024
NCU_PREFLIGHT_TIMEOUT = 60
NCU_PREFLIGHT_BINARY = Path("/opt/kernelblaster/bin/ncu-preflight")

NCU_METRICS: dict[str, tuple[str, str]] = {
    "gpu__time_duration.sum": ("gpu_time", "ns"),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_throughput", "percent"),
    "launch__occupancy_limit_blocks": ("occupancy_limit_blocks", "count"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved_occupancy", "percent"),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput", "percent"),
    "l1tex__t_bytes.sum": ("l1_bytes", "bytes"),
    "smsp__issue_active.avg.pct_of_peak_sustained_active": ("issue_active", "percent"),
    "smsp__warps_active.avg.pct_of_peak_sustained_active": ("warp_active", "percent"),
}

_TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "nsecond": 1.0,
    "nseconds": 1.0,
    "us": 1_000.0,
    "usecond": 1_000.0,
    "useconds": 1_000.0,
    "ms": 1_000_000.0,
    "msecond": 1_000_000.0,
    "mseconds": 1_000_000.0,
    "s": 1_000_000_000.0,
    "second": 1_000_000_000.0,
    "seconds": 1_000_000_000.0,
}

_BYTE_UNIT_TO_BYTES = {
    "byte": 1.0,
    "bytes": 1.0,
    "kbyte": 1_000.0,
    "kbytes": 1_000.0,
    "mbyte": 1_000_000.0,
    "mbytes": 1_000_000.0,
    "gbyte": 1_000_000_000.0,
    "gbytes": 1_000_000_000.0,
}


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    version: str
    returncode: int
    stdout: bytes
    stderr: bytes
    csv_output: bytes
    report: bytes
    timed_out: bool = False
    used_timestamp_workaround: bool = False


def _runtime_platform() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system != "linux":
        return "unsupported"
    release = platform.release().lower()
    proc_version = ""
    try:
        proc_version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        pass
    return "wsl" if "microsoft" in release or "microsoft" in proc_version else "linux"


def detect_capabilities() -> ProfilerCapabilities:
    runtime = _runtime_platform()
    if runtime in {"windows", "unsupported"}:
        return ProfilerCapabilities(
            platform=runtime,
            nsys_status="unsupported",
            ncu_status="unsupported",
            supported_plans=(),
            automatic_execution=False,
        )
    nsys = "available" if shutil.which("nsys") else "tool_missing"
    configured_ncu = os.getenv("KERNELBLASTER_NCU_PREFLIGHT_STATUS", "auto").lower()
    if not shutil.which("ncu"):
        ncu = "tool_missing"
    elif configured_ncu == "auto":
        ncu = probe_ncu_counters()
    elif configured_ncu in {"available", "permission_denied", "unsupported"}:
        ncu = configured_ncu
    else:
        raise RuntimeError(
            "KERNELBLASTER_NCU_PREFLIGHT_STATUS must be auto, available, "
            "permission_denied, or unsupported"
        )
    plans: list[ProfilePlanId] = []
    if nsys == "available":
        plans.append(ProfilePlanId.NSYS_TIMELINE_V1)
    if ncu == "available":
        plans.extend(PLAN_SECTIONS)
    return ProfilerCapabilities(
        platform=runtime,
        nsys_status=nsys,
        ncu_status=ncu,
        supported_plans=tuple(plans),
        automatic_execution=True,
    )


def probe_ncu_counters() -> str:
    """Run a fixed, bounded counter probe as the Profiler service identity."""
    if not shutil.which("ncu"):
        return "tool_missing"
    executable = Path(
        os.getenv("KERNELBLASTER_NCU_PREFLIGHT_BINARY", str(NCU_PREFLIGHT_BINARY))
    )
    if not executable.is_file():
        return "unsupported"
    try:
        with tempfile.TemporaryDirectory(prefix="kernelblaster-ncu-preflight-") as temporary:
            root = Path(temporary)
            report_base = root / "preflight"
            result = subprocess.run(
                [
                    "ncu",
                    "--metrics",
                    "gpu__time_duration.sum",
                    "--kernel-name-base",
                    "demangled",
                    "--kernel-name",
                    "regex:kernelblaster_ncu_preflight",
                    "--launch-count",
                    "1",
                    "--cache-control",
                    "none",
                    "--clock-control",
                    "none",
                    "--export",
                    str(report_base),
                    "--force-overwrite",
                    str(executable),
                ],
                cwd=root,
                env=_child_environment(root),
                check=False,
                capture_output=True,
                timeout=NCU_PREFLIGHT_TIMEOUT,
            )
            output = (result.stdout + result.stderr).decode(
                "utf-8", errors="replace"
            )
            if "ERR_NVGPUCTRPERM" in output:
                return "permission_denied"
            if result.returncode == 0 and report_base.with_suffix(".ncu-rep").is_file():
                return "available"
    except FileNotFoundError:
        return "tool_missing"
    except (OSError, subprocess.TimeoutExpired):
        return "unsupported"
    return "unsupported"


def profile_commands(
    request: ProfileRequest,
    executable: Path,
    report_base: Path,
    *,
    benchmark_protocol_id: str,
) -> tuple[list[str], list[str], Path]:
    """Return fixed argv; caller data occupies values, never command structure."""
    if request.plan_id is ProfilePlanId.NSYS_TIMELINE_V1:
        report = report_base.with_suffix(".nsys-rep")
        profile = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output",
            str(report_base),
            str(executable),
            "--mode",
            "events",
            "--protocol",
            benchmark_protocol_id,
        ]
        export = [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            "--force-export=true",
            str(report),
        ]
        return profile, export, report
    report = report_base.with_suffix(".ncu-rep")
    sections = [value for section in PLAN_SECTIONS[request.plan_id] for value in ("--section", section)]
    profile = [
        "ncu",
        *sections,
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{request.kernel_filter}",
        "--export",
        str(report_base),
        "--force-overwrite",
        str(executable),
        "--mode",
        "events",
        "--protocol",
        benchmark_protocol_id,
    ]
    export = ["ncu", "--import", str(report), "--csv", "--page", "raw"]
    return profile, export, report


def _child_environment(root: Path) -> dict[str, str]:
    environment = {
        "HOME": str(root),
        "PATH": os.getenv("PATH", "/usr/local/cuda/bin:/usr/bin:/bin"),
        "TMPDIR": str(root),
        "CUDA_VISIBLE_DEVICES": os.getenv("KERNELBLASTER_GPU_DEVICE", "0"),
    }
    return environment


def _configure_wsl_nsys_timestamps(root: Path) -> None:
    config = root / ".config" / "NVIDIA Corporation" / "nsys-config.ini"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("CuptiUseRawGpuTimestamps=false\n", encoding="utf-8")


async def _run_process(
    command: list[str], *, root: Path, timeout: float
) -> tuple[int, bytes, bytes, bool]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        env=_child_environment(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        return -1, b"", b"profile timed out", True
    return process.returncode or 0, stdout[:OUTPUT_LIMIT], stderr[:OUTPUT_LIMIT], False


class FixedToolRunner:
    async def run(
        self,
        request: ProfileRequest,
        executable: Path,
        benchmark_protocol_id: str,
        root: Path,
        timeout: float,
    ) -> ToolExecution:
        profile, export, report_path = profile_commands(
            request,
            executable,
            root / "profile",
            benchmark_protocol_id=benchmark_protocol_id,
        )
        tool = "nsys" if request.plan_id is ProfilePlanId.NSYS_TIMELINE_V1 else "ncu"
        version_process = subprocess.run(
            [tool, "--version"], check=False, capture_output=True, text=True, timeout=10
        )
        version = (version_process.stdout or version_process.stderr).strip().splitlines()[0]
        code, stdout, stderr, timed_out = await _run_process(
            profile, root=root, timeout=timeout
        )
        csv_output = b""
        workaround = False
        if code == 0 and not timed_out:
            export_code, csv_output, export_stderr, export_timeout = await _run_process(
                export, root=root, timeout=min(timeout, 120)
            )
            stderr += b"\n" + export_stderr
            timed_out = export_timeout
            code = export_code
        if (
            tool == "nsys"
            and report_path.is_file()
            and _runtime_platform() == "wsl"
            and not _csv_rows(
                csv_output.decode("utf-8", errors="replace"), "Total Time"
            )
            and not timed_out
        ):
            workaround = True
            _configure_wsl_nsys_timestamps(root)
            code, retry_stdout, retry_stderr, timed_out = await _run_process(
                profile, root=root, timeout=timeout
            )
            stdout += b"\n" + retry_stdout
            stderr += b"\n" + retry_stderr
            if code == 0 and not timed_out:
                code, csv_output, export_stderr, export_timeout = await _run_process(
                    export, root=root, timeout=min(timeout, 120)
                )
                stderr += b"\n" + export_stderr
                timed_out = export_timeout
        report = report_path.read_bytes() if report_path.is_file() else b""
        return ToolExecution(
            tool=tool,
            version=version or "unknown",
            returncode=code,
            stdout=stdout,
            stderr=stderr,
            csv_output=csv_output,
            report=report,
            timed_out=timed_out,
            used_timestamp_workaround=workaround,
        )


def _number(value: str) -> float:
    cleaned = value.strip().replace(",", "").replace("%", "")
    return float(cleaned)


def _canonical_metric_value(value: str, source_unit: str, target_unit: str) -> float:
    number = _number(value)
    unit = source_unit.strip().lower().replace("\u00b5", "u")
    if target_unit == "ns":
        return number * _TIME_UNIT_TO_NS[unit]
    if target_unit == "bytes":
        return number * _BYTE_UNIT_TO_BYTES[unit]
    return number


def _configured_metric(
    raw_name: str, raw_value: str, raw_unit: str
) -> ProfileMetric | None:
    configured = NCU_METRICS.get(raw_name.strip())
    if not configured or not raw_value.strip():
        return None
    name, unit = configured
    try:
        value = _canonical_metric_value(raw_value, raw_unit, unit)
    except (KeyError, ValueError):
        return None
    return ProfileMetric(name=name, value=value, unit=unit)


def _csv_rows(text: str, marker: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if marker in line), None)
    if start is None:
        return []
    return list(csv.DictReader(StringIO("\n".join(lines[start:]))))


def _parse_ncu_wide_csv(
    text: str, kernel_filter: str
) -> tuple[str, tuple[ProfileMetric, ...], ProfileReasonCode] | None:
    """Parse the wide raw CSV emitted by recent Nsight Compute releases."""
    reader = csv.reader(StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return None
    try:
        kernel_index = header.index("Kernel Name")
    except ValueError:
        return None
    metric_indexes = [
        (index, raw_name)
        for index, raw_name in enumerate(header)
        if raw_name in NCU_METRICS
    ]
    if not metric_indexes:
        return None
    records = list(reader)
    if not records:
        return "", (), ProfileReasonCode.METRICS_EMPTY

    # The first record in NCU's wide form is a units row with no kernel name.
    has_units_row = (
        len(records[0]) > kernel_index and not records[0][kernel_index].strip()
    )
    units = records[0] if has_units_row else [""] * len(header)
    data_rows = records[1:] if has_units_row else records
    matching = [
        row
        for row in data_rows
        if len(row) > kernel_index and kernel_filter in row[kernel_index]
    ]
    if not matching:
        return "", (), (
            ProfileReasonCode.KERNEL_NOT_FOUND
            if data_rows
            else ProfileReasonCode.METRICS_EMPTY
        )

    metrics: list[ProfileMetric] = []
    for row in matching:
        for index, raw_name in metric_indexes:
            if index >= len(row):
                continue
            source_unit = units[index] if index < len(units) else ""
            metric = _configured_metric(raw_name, row[index], source_unit)
            if metric is not None:
                metrics.append(metric)
    return matching[0][kernel_index], tuple(metrics), (
        ProfileReasonCode.NONE if metrics else ProfileReasonCode.METRICS_EMPTY
    )


def parse_profile_csv(
    plan_id: ProfilePlanId, csv_payload: bytes, kernel_filter: str
) -> tuple[str, tuple[ProfileMetric, ...], ProfileReasonCode]:
    try:
        text = csv_payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "", (), ProfileReasonCode.METRICS_EMPTY
    if plan_id is ProfilePlanId.NSYS_TIMELINE_V1:
        rows = _csv_rows(text, "Total Time")
        if not rows:
            return "", (), ProfileReasonCode.METRICS_EMPTY
        matching = [row for row in rows if kernel_filter in str(row.get("Name", ""))]
        if not matching:
            return "", (), ProfileReasonCode.KERNEL_NOT_FOUND
        row = matching[0]
        metrics: list[ProfileMetric] = []
        for column, name, unit in (
            ("Total Time (ns)", "gpu_time_total", "ns"),
            ("Avg (ns)", "gpu_time_average", "ns"),
            ("Instances", "instances", "count"),
        ):
            if row.get(column, "").strip():
                metrics.append(ProfileMetric(name=name, value=_number(row[column]), unit=unit))
        return str(row.get("Name", "")), tuple(metrics), (
            ProfileReasonCode.NONE if metrics else ProfileReasonCode.METRICS_EMPTY
        )
    rows = _csv_rows(text, "Metric Name")
    if not rows:
        wide = _parse_ncu_wide_csv(text, kernel_filter)
        return wide or ("", (), ProfileReasonCode.METRICS_EMPTY)
    matching = [
        row for row in rows if kernel_filter in str(row.get("Kernel Name", row.get("KernelName", "")))
    ]
    if not matching:
        return "", (), ProfileReasonCode.KERNEL_NOT_FOUND
    metrics: list[ProfileMetric] = []
    for row in matching:
        raw_name = str(row.get("Metric Name", "")).strip()
        raw_value = str(row.get("Metric Value", "")).strip()
        raw_unit = str(row.get("Metric Unit", "")).strip()
        metric = _configured_metric(raw_name, raw_value, raw_unit)
        if metric is not None:
            metrics.append(metric)
    kernel_name = str(matching[0].get("Kernel Name", matching[0].get("KernelName", "")))
    return kernel_name, tuple(metrics), (
        ProfileReasonCode.NONE if metrics else ProfileReasonCode.METRICS_EMPTY
    )


def _hardware() -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        name, driver = [field.strip() for field in result.stdout.strip().split(",", 1)]
        return name, driver
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown", "unknown"


class FixedPlanProfiler:
    def __init__(
        self,
        control: Any,
        *,
        capabilities: ProfilerCapabilities,
        runner: Any | None = None,
    ) -> None:
        self.control = control
        self.capabilities = capabilities
        self.runner = runner or FixedToolRunner()
        self._gpu = asyncio.Semaphore(1)

    def _blocked_reason(self, plan: ProfilePlanId) -> ProfileReasonCode | None:
        if not self.capabilities.automatic_execution:
            return ProfileReasonCode.UNSUPPORTED
        if plan is ProfilePlanId.NSYS_TIMELINE_V1:
            return (
                None
                if self.capabilities.nsys_status == "available"
                else ProfileReasonCode.TOOL_MISSING
            )
        return {
            "available": None,
            "permission_denied": ProfileReasonCode.PERMISSION_DENIED,
            "tool_missing": ProfileReasonCode.TOOL_MISSING,
            "unsupported": ProfileReasonCode.UNSUPPORTED,
        }[self.capabilities.ncu_status]

    async def _upload(
        self,
        payload: bytes,
        *,
        media_type: str,
        schema: str,
        source_digest: str,
    ) -> str:
        expected = hashlib.sha256(payload).hexdigest()
        uploaded = await self.control.upload(
            payload,
            media_type=media_type,
            schema=schema,
            source_digest=source_digest,
        )
        if uploaded.get("digest") != expected:
            raise ValueError("Control returned an invalid profiler artifact digest")
        return expected

    async def profile(self, request: ProfileRequest) -> ProfileResult:
        blocked = self._blocked_reason(request.plan_id)
        if blocked is not None:
            return ProfileResult(
                status=ProfileStatus.BLOCKED, reason_code=blocked, plan_id=request.plan_id
            )
        remaining = (request.deadline.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
        timeout = min(float(PLAN_TIMEOUTS[request.plan_id]), remaining)
        if timeout <= 0:
            return ProfileResult(
                status=ProfileStatus.TIMED_OUT,
                reason_code=ProfileReasonCode.TIMEOUT,
                plan_id=request.plan_id,
            )
        async with self._gpu:
            try:
                executable, source_digest, benchmark_protocol_id = (
                    await self.control.download(request.artifact_digest)
                )
                if hashlib.sha256(executable).hexdigest() != request.artifact_digest:
                    raise ValueError("candidate artifact digest mismatch")
                with tempfile.TemporaryDirectory(prefix="kernelblaster-profile-") as temporary:
                    root = Path(temporary)
                    candidate = root / "candidate"
                    candidate.write_bytes(executable)
                    candidate.chmod(0o500)
                    execution = await self.runner.run(
                        request,
                        candidate,
                        benchmark_protocol_id,
                        root,
                        timeout,
                    )
                artifacts: dict[str, str] = {}
                for payload, role, media_type, schema in (
                    (execution.report, "raw_report", "application/octet-stream", "profiler-report/v1"),
                    (execution.csv_output, "raw_csv", "text/csv", "profiler-csv/v1"),
                    (execution.stdout + b"\n" + execution.stderr, "tool_log", "text/plain", "profiler-log/v1"),
                ):
                    if payload:
                        digest = await self._upload(
                            payload,
                            media_type=media_type,
                            schema=schema,
                            source_digest=source_digest,
                        )
                        artifacts[digest] = role
                if execution.timed_out:
                    return ProfileResult(
                        status=ProfileStatus.TIMED_OUT,
                        reason_code=ProfileReasonCode.TIMEOUT,
                        plan_id=request.plan_id,
                        artifact_roles=artifacts,
                    )
                combined = (execution.stderr + b"\n" + execution.stdout).decode(
                    "utf-8", errors="replace"
                )
                if "ERR_NVGPUCTRPERM" in combined:
                    reason = ProfileReasonCode.PERMISSION_DENIED
                elif execution.returncode != 0:
                    reason = ProfileReasonCode.EXECUTION_FAILED
                elif not execution.report:
                    reason = ProfileReasonCode.EXECUTION_FAILED
                else:
                    _kernel, _metrics, reason = parse_profile_csv(
                        request.plan_id, execution.csv_output, request.kernel_filter
                    )
                if reason is not ProfileReasonCode.NONE:
                    status = (
                        ProfileStatus.BLOCKED
                        if reason in {ProfileReasonCode.PERMISSION_DENIED, ProfileReasonCode.TOOL_MISSING, ProfileReasonCode.UNSUPPORTED}
                        else ProfileStatus.FAILED
                    )
                    return ProfileResult(
                        status=status,
                        reason_code=reason,
                        plan_id=request.plan_id,
                        artifact_roles=artifacts,
                    )
                kernel, metrics, _reason = parse_profile_csv(
                    request.plan_id, execution.csv_output, request.kernel_filter
                )
                summary = ProfileSummary(
                    plan_id=request.plan_id, kernel_name=kernel, metrics=metrics
                )
                gpu_name, driver_version = _hardware()
                provenance = ProfileProvenance(
                    candidate_artifact_digest=request.artifact_digest,
                    source_digest=source_digest,
                    tool=execution.tool,
                    tool_version=execution.version,
                    gpu_name=gpu_name,
                    driver_version=driver_version,
                    image_digest=os.getenv("KERNELBLASTER_PROFILER_IMAGE_DIGEST") or None,
                    workarounds=(
                        ("nsys_timestamp_retry",)
                        if execution.used_timestamp_workaround
                        else ()
                    ),
                )
                summary_payload = json.dumps(
                    {
                        "summary": summary.model_dump(mode="json"),
                        "provenance": provenance.model_dump(mode="json"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                summary_digest = await self._upload(
                    summary_payload,
                    media_type="application/json",
                    schema="profiler-summary/v1",
                    source_digest=source_digest,
                )
                artifacts[summary_digest] = "structured_summary"
                return ProfileResult(
                    status=ProfileStatus.SUCCEEDED,
                    plan_id=request.plan_id,
                    summary=summary,
                    provenance=provenance,
                    artifact_roles=artifacts,
                )
            except FileNotFoundError:
                reason = ProfileReasonCode.TOOL_MISSING
            except PermissionError:
                reason = ProfileReasonCode.PERMISSION_DENIED
            except Exception:
                reason = ProfileReasonCode.INTERNAL_ERROR
            return ProfileResult(
                status=(
                    ProfileStatus.BLOCKED
                    if reason in {
                        ProfileReasonCode.TOOL_MISSING,
                        ProfileReasonCode.PERMISSION_DENIED,
                    }
                    else ProfileStatus.FAILED
                ),
                reason_code=reason,
                plan_id=request.plan_id,
            )


APP = FastAPI(title="KernelBlaster Profiler Worker")


async def _require_profiler_token(
    authorization: str | None = Header(default=None),
) -> None:
    from ..servers.auth import require_profiler_token

    await require_profiler_token(authorization)


def _profiler() -> FixedPlanProfiler:
    worker = getattr(APP.state, "profiler", None)
    if worker is None:
        from ..servers.auth import validate_profiler_token

        token = validate_profiler_token()
        control_url = os.getenv("KERNELBLASTER_CONTROL_URL", "").strip()
        if not control_url:
            raise RuntimeError("KERNELBLASTER_CONTROL_URL is required")
        capabilities = detect_capabilities()
        worker = FixedPlanProfiler(
            ControlProfilerClient(control_url, token), capabilities=capabilities
        )
        APP.state.profiler = worker
    return worker


@APP.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "profiler-worker"}


@APP.get("/ready")
async def ready(
    _authorized: None = Depends(_require_profiler_token),
) -> dict[str, str]:
    _profiler()
    return {"status": "ready", "service": "profiler-worker"}


@APP.get("/v1/capabilities")
async def capabilities(
    _authorized: None = Depends(_require_profiler_token),
) -> dict[str, object]:
    return _profiler().capabilities.model_dump(mode="json")


@APP.post("/v1/profiles")
async def profile(
    request: ProfileRequest,
    _authorized: None = Depends(_require_profiler_token),
) -> dict[str, object]:
    return (await _profiler().profile(request)).model_dump(mode="json")


def main() -> None:
    from ..servers.auth import validate_profiler_token

    parser = argparse.ArgumentParser(description="Run the fixed-plan Profiler Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2003)
    args = parser.parse_args()
    validate_profiler_token()
    _profiler()
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = [
    "APP",
    "FixedPlanProfiler",
    "FixedToolRunner",
    "ToolExecution",
    "detect_capabilities",
    "parse_profile_csv",
    "probe_ncu_counters",
    "profile_commands",
]
