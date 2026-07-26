# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate evaluation contracts and the task-end diagnostic funnel."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Protocol
import uuid

from .gpu_jobs import build_deterministic_bundle
from .harness.contracts import CorrectnessResultV2
from .measurements import Measurement, MeasurementSource, MeasurementUnit
from .observability import record_event
from .outcomes import DiagnosticStatus
from .preflight.client import ControlPlaneClient
from .preflight.contracts import CapabilityReport
from .profiling import PerformanceGateResult, evaluate_performance_gate


class CandidateStageStatus(str, Enum):
    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CompilationMetrics:
    registers: int | None = None
    spill_load_bytes: int = 0
    spill_store_bytes: int = 0
    stack_frame_bytes: int = 0
    shared_memory_bytes: int = 0
    constant_memory_bytes: int = 0
    kernel_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "registers": self.registers,
            "spill_load_bytes": self.spill_load_bytes,
            "spill_store_bytes": self.spill_store_bytes,
            "stack_frame_bytes": self.stack_frame_bytes,
            "shared_memory_bytes": self.shared_memory_bytes,
            "constant_memory_bytes": self.constant_memory_bytes,
            "kernel_names": list(self.kernel_names),
        }


@dataclass
class CandidateEvaluation:
    candidate_id: str
    source_digest: str
    executable_digest: str | None = None
    execution_status: CandidateStageStatus = CandidateStageStatus.NOT_RUN
    correctness_status: CandidateStageStatus = CandidateStageStatus.NOT_RUN
    events_status: CandidateStageStatus = CandidateStageStatus.NOT_RUN
    measurement: Measurement | None = None
    compilation_metrics: CompilationMetrics = field(default_factory=CompilationMetrics)
    artifact_roles: dict[str, str] = field(default_factory=dict)
    reason_code: str = "none"
    discovery_samples_us: tuple[float, ...] = ()
    confirmation_samples_us: tuple[float, ...] = ()
    diagnostic_status: DiagnosticStatus = DiagnosticStatus.NOT_REQUESTED
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    correctness: dict[str, Any] | None = None

    @property
    def rankable(self) -> bool:
        measurement = self.measurement
        return bool(
            self.execution_status is CandidateStageStatus.SUCCEEDED
            and self.correctness_status is CandidateStageStatus.SUCCEEDED
            and self.events_status is CandidateStageStatus.SUCCEEDED
            and measurement is not None
            and measurement.source is MeasurementSource.CUDA_EVENTS
            and measurement.unit is MeasurementUnit.MICROSECONDS
            and math.isfinite(measurement.value)
            and measurement.value > 0
        )

    def public_feedback(self) -> dict[str, Any]:
        """Return the only candidate feedback that may enter an Agent prompt."""
        return {
            "execution_status": self.execution_status.value,
            "correctness_status": self.correctness_status.value,
            "events_status": self.events_status.value,
            "reason_code": self.reason_code,
            "compilation_metrics": self.compilation_metrics.to_dict(),
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "correctness": self.correctness,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_digest": self.source_digest,
            "executable_digest": self.executable_digest,
            "execution_status": self.execution_status.value,
            "correctness_status": self.correctness_status.value,
            "events_status": self.events_status.value,
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "compilation_metrics": self.compilation_metrics.to_dict(),
            "artifact_roles": dict(self.artifact_roles),
            "reason_code": self.reason_code,
            "discovery_samples_us": list(self.discovery_samples_us),
            "confirmation_samples_us": list(self.confirmation_samples_us),
            "diagnostic_status": self.diagnostic_status.value,
            "diagnostics": dict(self.diagnostics),
            "correctness": self.correctness,
        }


class CandidateEvaluationError(RuntimeError):
    def __init__(self, evaluation: CandidateEvaluation) -> None:
        self.evaluation = evaluation
        super().__init__(json.dumps(evaluation.public_feedback(), sort_keys=True))


class CandidateEvaluator(Protocol):
    async def evaluate(self, filepath: Path) -> CandidateEvaluation: ...

    async def finalize(
        self,
        baseline: CandidateEvaluation,
        candidates: list[CandidateEvaluation],
    ) -> "FunnelResult": ...


@dataclass(frozen=True)
class FunnelDecision:
    eligible_source_digests: tuple[str, ...]
    confirmation_source_digests: tuple[str, ...]
    anomaly_source_digest: str | None


@dataclass
class FunnelResult:
    decision: FunnelDecision
    candidates: tuple[CandidateEvaluation, ...]
    winner_source_digest: str | None = None
    confirmed_source_digests: tuple[str, ...] = ()
    performance_gate: PerformanceGateResult | None = None
    discovery_calls: int = 0
    confirmation_calls: int = 0
    nsys_calls: int = 0
    ncu_calls: int = 0

    @property
    def winner(self) -> CandidateEvaluation | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.source_digest == self.winner_source_digest
            ),
            None,
        )


class FunnelPolicy:
    def __init__(self, *, anomaly_ratio: float = 1.10, ncu_top2_ratio: float = 1.01):
        self.anomaly_ratio = anomaly_ratio
        self.ncu_top2_ratio = ncu_top2_ratio

    @staticmethod
    def deduplicate(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        unique: dict[str, CandidateEvaluation] = {}
        for candidate in candidates:
            unique.setdefault(candidate.source_digest, candidate)
        return list(unique.values())

    @staticmethod
    def confirmation_count(correct_candidate_count: int) -> int:
        if correct_candidate_count <= 0:
            return 0
        return min(3, max(1, math.ceil(correct_candidate_count * 0.10)))

    def decide(self, candidates: list[CandidateEvaluation]) -> FunnelDecision:
        eligible = sorted(
            (candidate for candidate in self.deduplicate(candidates) if candidate.rankable),
            key=lambda candidate: candidate.measurement.value,  # type: ignore[union-attr]
        )
        confirmation_count = self.confirmation_count(len(eligible))
        anomaly: CandidateEvaluation | None = None
        anomalies: list[tuple[float, CandidateEvaluation]] = []
        for candidate in eligible:
            samples = candidate.discovery_samples_us
            if not samples or min(samples) <= 0:
                continue
            ratio = max(samples) / min(samples)
            if ratio > self.anomaly_ratio:
                anomalies.append((ratio, candidate))
        if anomalies:
            anomaly = max(anomalies, key=lambda item: item[0])[1]
        return FunnelDecision(
            eligible_source_digests=tuple(item.source_digest for item in eligible),
            confirmation_source_digests=tuple(
                item.source_digest for item in eligible[:confirmation_count]
            ),
            anomaly_source_digest=anomaly.source_digest if anomaly else None,
        )

    def ncu_targets(
        self, confirmed: list[CandidateEvaluation]
    ) -> tuple[CandidateEvaluation, ...]:
        ranked = sorted(
            (candidate for candidate in confirmed if candidate.confirmation_samples_us),
            key=lambda candidate: statistics.median(candidate.confirmation_samples_us),
        )
        if not ranked:
            return ()
        targets = [ranked[0]]
        if len(ranked) > 1:
            first = statistics.median(ranked[0].confirmation_samples_us)
            second = statistics.median(ranked[1].confirmation_samples_us)
            if second / first <= self.ncu_top2_ratio:
                targets.append(ranked[1])
        return tuple(targets)


_GLOBAL_KERNEL = re.compile(
    r"__global__\s+(?:[A-Za-z_][A-Za-z0-9_:<>,*&]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_SAFE_KERNEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def candidate_kernel_names(source: str) -> tuple[str, ...]:
    without_bounds = re.sub(r"__launch_bounds__\s*\([^)]*\)", "", source)
    return tuple(dict.fromkeys(_GLOBAL_KERNEL.findall(without_bounds)))


def parse_ptxas_metrics(
    raw_log: str,
    *,
    candidate_kernels: tuple[str, ...],
) -> CompilationMetrics:
    """Parse numeric ptxas output without returning paths or source excerpts."""
    relevant = raw_log
    compiled_entries = re.findall(
        r"ptxas info\s*: Compiling entry function '([^']+)'(.*?)(?=ptxas info\s*: Compiling entry function|'?\Z)",
        raw_log,
        flags=re.DOTALL,
    )
    matched_names: list[str] = []
    relevant_blocks: list[str] = []
    for mangled, block in compiled_entries:
        names = [name for name in candidate_kernels if name in mangled]
        if names:
            matched_names.extend(names)
            relevant_blocks.append(block)
    if relevant_blocks:
        relevant = "\n".join(relevant_blocks)
    elif candidate_kernels:
        # A single source kernel is still a safe attribution when older ptxas
        # versions omit the entry-function banner.
        matched_names = list(candidate_kernels) if len(candidate_kernels) == 1 else []

    def maximum(pattern: str) -> int:
        values = [int(value) for value in re.findall(pattern, relevant, re.IGNORECASE)]
        return max(values, default=0)

    constant_values = [
        int(value)
        for value in re.findall(r"(\d+)\s+bytes\s+cmem\[\d+\]", relevant, re.IGNORECASE)
    ]
    registers = maximum(r"Used\s+(\d+)\s+registers")
    return CompilationMetrics(
        registers=registers or None,
        spill_load_bytes=maximum(r"(\d+)\s+bytes\s+spill loads"),
        spill_store_bytes=maximum(r"(\d+)\s+bytes\s+spill stores"),
        stack_frame_bytes=maximum(r"(\d+)\s+bytes\s+stack frame"),
        shared_memory_bytes=maximum(r"(\d+)\s+bytes\s+(?:smem|shared memory)"),
        constant_memory_bytes=sum(constant_values),
        kernel_names=tuple(
            name for name in dict.fromkeys(matched_names) if _SAFE_KERNEL.fullmatch(name)
        ),
    )


def safe_kernel_base_name(value: str) -> str | None:
    if any(marker in value for marker in ("/", "\\", "--", ";", "\n", "\r")):
        return None
    prefix = value.split("(", 1)[0].strip()
    prefix = re.sub(r"^(?:void|int|float|double)\s+", "", prefix)
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>]{0,128}>)?$", prefix)
    if not match:
        return None
    candidate = match.group(1)
    return candidate if _SAFE_KERNEL.fullmatch(candidate) else None


def _artifact_roles(payload: dict[str, Any]) -> dict[str, str]:
    result = dict(payload.get("result") or {})
    return {str(key): str(value) for key, value in (result.get("artifact_roles") or {}).items()}


def _artifact_for_role(payload: dict[str, Any], role: str) -> str | None:
    return next(
        (digest for digest, actual in _artifact_roles(payload).items() if actual == role),
        None,
    )


def _job_status(payload: dict[str, Any]) -> CandidateStageStatus:
    return {
        "succeeded": CandidateStageStatus.SUCCEEDED,
        "timed_out": CandidateStageStatus.TIMED_OUT,
        "blocked": CandidateStageStatus.BLOCKED,
    }.get(str(payload.get("status")), CandidateStageStatus.FAILED)


def _job_reason(payload: dict[str, Any]) -> str:
    return str((payload.get("result") or {}).get("reason_code") or "execution_failed")


class SandboxCandidateEvaluator:
    """Evaluate generated candidates only through Control and ephemeral GPU Jobs."""

    def __init__(
        self,
        control: ControlPlaneClient,
        report: CapabilityReport,
        *,
        run_id: str,
        private_evaluation_profile_id: str,
        benchmark_protocol_id: str,
        discovery_sessions: int = 3,
        confirmation_sessions: int = 5,
        timeout_seconds: float = 10 * 60,
        policy: FunnelPolicy | None = None,
        trusted_bundle_kind: str = "generated_v1",
        task: Any | None = None,
    ) -> None:
        if not private_evaluation_profile_id.strip():
            raise ValueError("sandbox CandidateEvaluator requires a private profile ID")
        self.control = control
        self.report = report
        self.run_id = run_id
        self.private_evaluation_profile_id = private_evaluation_profile_id
        self.benchmark_protocol_id = benchmark_protocol_id
        self.discovery_sessions = discovery_sessions
        self.confirmation_sessions = confirmation_sessions
        self.timeout_seconds = timeout_seconds
        self.policy = policy or FunnelPolicy()
        if trusted_bundle_kind not in {"generated_v1", "generated_v2"}:
            raise ValueError("sandbox evaluator supports generated_v1 or generated_v2")
        self.trusted_bundle_kind = trusted_bundle_kind
        self.task = task
        self._cache: dict[str, CandidateEvaluation] = {}
        self._source_kernels: dict[str, tuple[str, ...]] = {}
        self._lock = asyncio.Lock()
        self._run_ready = False

    async def _ensure_run(self) -> None:
        if self._run_ready:
            return
        try:
            await self.control.create_run(
                self.run_id,
                {
                    "kind": "agent-candidate-funnel",
                    "schema_version": "candidate-evaluation/v1",
                },
            )
        except Exception as error:
            # A retry may observe the run created by the first attempt. Any
            # other failure remains fail-closed after the required read-back.
            if getattr(error, "status", None) != 409:
                raise
        loaded = await self.control.get_run(self.run_id)
        if loaded.get("id") != self.run_id:
            raise ValueError("candidate evaluation run did not round-trip")
        self._run_ready = True

    async def _job(
        self,
        evaluation: CandidateEvaluation,
        *,
        stage: str,
        phase: str,
        index: int,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        manifest: dict[str, Any] = {
            "schema_version": "gpu-job/v1",
            "job_id": job_id,
            "run_id": self.run_id,
            "idempotency_key": (
                f"candidate:{evaluation.source_digest}:{stage}:{phase}:{index}"
            ),
            "stage": stage,
            "source_bundle_digest": evaluation.source_digest,
            "target_arch": self.report.target_arch,
            "benchmark_protocol_id": self.benchmark_protocol_id,
            "deadline": (
                datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
            ).isoformat(),
            "trusted_bundle_kind": self.trusted_bundle_kind,
            "private_evaluation_profile_id": self.private_evaluation_profile_id,
        }
        if stage != "compile":
            manifest["executable_digest"] = evaluation.executable_digest
        await self.control.submit_gpu_job(manifest)
        return await self.control.wait_gpu_job(
            job_id,
            timeout_seconds=self.timeout_seconds,
        )

    async def _event_value(
        self,
        evaluation: CandidateEvaluation,
        *,
        phase: str,
        index: int,
    ) -> float:
        payload = await self._job(
            evaluation,
            stage="events",
            phase=phase,
            index=index,
        )
        evaluation.artifact_roles.update(_artifact_roles(payload))
        if _job_status(payload) is not CandidateStageStatus.SUCCEEDED:
            raise CandidateEvaluationError(
                CandidateEvaluation(
                    candidate_id=evaluation.candidate_id,
                    source_digest=evaluation.source_digest,
                    executable_digest=evaluation.executable_digest,
                    execution_status=evaluation.execution_status,
                    correctness_status=evaluation.correctness_status,
                    events_status=_job_status(payload),
                    reason_code=_job_reason(payload),
                )
            )
        measurement = dict((payload.get("result") or {}).get("measurement") or {})
        value = measurement.get("value")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or measurement.get("source") != "cuda_events"
            or measurement.get("unit") != "us"
            or measurement.get("protocol_id") != self.benchmark_protocol_id
        ):
            raise ValueError("events measurement contract mismatch")
        return float(value)

    async def evaluate(self, filepath: Path) -> CandidateEvaluation:
        if self.trusted_bundle_kind == "generated_v2":
            if self.task is None:
                raise ValueError("generated_v2 requires a bound TaskSpec")
            return await self.evaluate_package(filepath, task=self.task)
        source = filepath.read_bytes()
        raw_digest = hashlib.sha256(source).hexdigest()
        bundle = build_deterministic_bundle({"candidate.cu": source})
        return await self._evaluate_bundle(
            raw_digest=raw_digest,
            bundle=bundle,
            schema="gpu-source-bundle/v1",
            kernel_names=candidate_kernel_names(source.decode("utf-8", errors="replace")),
        )

    async def evaluate_package(self, filepath: Path, *, task: Any) -> CandidateEvaluation:
        if self.trusted_bundle_kind != "generated_v2":
            raise ValueError("CandidatePackage v2 requires the generated_v2 evaluator")
        from .candidate_packages import validate_candidate_package

        bundle = filepath.read_bytes()
        package = validate_candidate_package(bundle, task=task)
        return await self._evaluate_bundle(
            raw_digest=package.digest,
            bundle=bundle,
            schema="candidate-package/v2",
            kernel_names=tuple(item.name for item in package.launch_plan.kernels),
        )

    async def _evaluate_bundle(
        self,
        *,
        raw_digest: str,
        bundle: bytes,
        schema: str,
        kernel_names: tuple[str, ...],
    ) -> CandidateEvaluation:
        async with self._lock:
            cached = self._cache.get(raw_digest)
            if cached is not None:
                return cached
            await self._ensure_run()
            uploaded = await self.control.upload(
                bundle,
                media_type="application/x-tar",
                schema=schema,
            )
            source_digest = str(uploaded.get("digest") or "")
            if source_digest != hashlib.sha256(bundle).hexdigest():
                raise ValueError("candidate source artifact hash mismatch")
            evaluation = CandidateEvaluation(
                candidate_id=raw_digest,
                source_digest=source_digest,
            )
            self._source_kernels[source_digest] = kernel_names

            compile_job = await self._job(
                evaluation,
                stage="compile",
                phase="evaluation",
                index=0,
            )
            evaluation.artifact_roles.update(_artifact_roles(compile_job))
            evaluation.execution_status = _job_status(compile_job)
            evaluation.reason_code = _job_reason(compile_job)
            evaluation.executable_digest = _artifact_for_role(
                compile_job, "executable"
            ) or _artifact_for_role(compile_job, "candidate_capsule")
            compile_log = _artifact_for_role(compile_job, "compile_log")
            if compile_log:
                raw_log_bytes = await self.control.download(compile_log)
                if hashlib.sha256(raw_log_bytes).hexdigest() != compile_log:
                    raise ValueError("compile log artifact hash mismatch")
                raw_log = raw_log_bytes.decode("utf-8", errors="replace")
                evaluation.compilation_metrics = parse_ptxas_metrics(
                    raw_log,
                    candidate_kernels=self._source_kernels[source_digest],
                )
            if (
                evaluation.execution_status is not CandidateStageStatus.SUCCEEDED
                or not evaluation.executable_digest
            ):
                self._cache[raw_digest] = evaluation
                self._record_evaluation(evaluation)
                return evaluation

            correctness_job = await self._job(
                evaluation,
                stage="correctness",
                phase="evaluation",
                index=0,
            )
            evaluation.artifact_roles.update(_artifact_roles(correctness_job))
            evaluation.correctness_status = _job_status(correctness_job)
            evaluation.reason_code = _job_reason(correctness_job)
            raw_correctness = (correctness_job.get("result") or {}).get("correctness")
            if raw_correctness is not None:
                try:
                    evaluation.correctness = CorrectnessResultV2.model_validate(
                        raw_correctness
                    ).public_feedback()
                except ValueError:
                    evaluation.correctness_status = CandidateStageStatus.FAILED
                    evaluation.reason_code = "correctness_failed"
            if evaluation.correctness_status is not CandidateStageStatus.SUCCEEDED:
                self._cache[raw_digest] = evaluation
                self._record_evaluation(evaluation)
                return evaluation

            samples: list[float] = []
            try:
                for index in range(self.discovery_sessions):
                    samples.append(
                        await self._event_value(
                            evaluation,
                            phase="discovery",
                            index=index,
                        )
                    )
                evaluation.discovery_samples_us = tuple(samples)
                evaluation.events_status = CandidateStageStatus.SUCCEEDED
                evaluation.reason_code = "none"
                evaluation.measurement = Measurement(
                    value=statistics.median(samples),
                    unit=MeasurementUnit.MICROSECONDS,
                    source=MeasurementSource.CUDA_EVENTS,
                    samples=tuple(samples),
                    protocol_id=self.benchmark_protocol_id,
                    hardware_fingerprint=self.report.hardware_fingerprint,
                )
            except CandidateEvaluationError as error:
                evaluation.events_status = error.evaluation.events_status
                evaluation.reason_code = error.evaluation.reason_code
            except Exception:
                evaluation.events_status = CandidateStageStatus.FAILED
                evaluation.reason_code = "events_failed"
            self._cache[raw_digest] = evaluation
            self._record_evaluation(evaluation)
            return evaluation

    def _record_evaluation(self, evaluation: CandidateEvaluation) -> None:
        record_event(
            "candidate_evaluation_completed",
            status="ok" if evaluation.rankable else "error",
            candidate_id=evaluation.candidate_id,
            data={
                **evaluation.to_dict(),
                "discovery_sessions": len(evaluation.discovery_samples_us),
            },
        )

    async def _profile(
        self,
        evaluation: CandidateEvaluation,
        *,
        plan_id: str,
        kernel_filter: str,
    ) -> dict[str, Any]:
        try:
            artifact_digest = evaluation.executable_digest
            if self.trusted_bundle_kind == "generated_v2":
                artifact_digest = next(
                    (
                        digest
                        for digest, role in evaluation.artifact_roles.items()
                        if role == "profiler_replay"
                    ),
                    None,
                )
                if artifact_digest is None:
                    raise ValueError("correctness-gated profiler replay is unavailable")
            payload = await self.control.profile(
                {
                    "artifact_digest": artifact_digest,
                    "plan_id": plan_id,
                    "kernel_filter": kernel_filter,
                    "deadline": (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=self.timeout_seconds)
                    ).isoformat(),
                }
            )
            roles = {
                str(key): str(value)
                for key, value in (payload.get("artifact_roles") or {}).items()
            }
            evaluation.artifact_roles.update(roles)
            summary = (
                payload.get("summary")
                if isinstance(payload.get("summary"), dict)
                else None
            )
            public = {
                "status": str(payload.get("status") or "failed"),
                "reason_code": str(payload.get("reason_code") or "internal_error"),
                "summary": summary,
            }
        except Exception:
            public = {
                "status": "failed",
                "reason_code": "internal_error",
                "summary": None,
            }
        record_event(
            "diagnostic_profile_completed",
            status="ok" if public["status"] == "succeeded" else "error",
            candidate_id=evaluation.candidate_id,
            data={
                "plan_id": plan_id,
                "status": public["status"],
                "reason_code": public["reason_code"],
            },
        )
        return public

    @staticmethod
    def _diagnostic_status(results: list[dict[str, Any]]) -> DiagnosticStatus:
        succeeded = sum(item.get("status") == "succeeded" for item in results)
        if not results or succeeded == 0:
            return DiagnosticStatus.UNAVAILABLE
        if succeeded == len(results):
            return DiagnosticStatus.AVAILABLE
        return DiagnosticStatus.PARTIAL

    async def finalize(
        self,
        baseline: CandidateEvaluation,
        candidates: list[CandidateEvaluation],
    ) -> FunnelResult:
        unique = self.policy.deduplicate(candidates)
        decision = self.policy.decide(unique)
        by_digest = {candidate.source_digest: candidate for candidate in unique}
        confirmed: list[CandidateEvaluation] = []
        gates: dict[str, PerformanceGateResult] = {}
        confirmation_calls = 0
        for digest in decision.confirmation_source_digests:
            candidate = by_digest[digest]
            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            try:
                for index in range(self.confirmation_sessions):
                    order = (
                        ((baseline, baseline_samples), (candidate, candidate_samples))
                        if index % 2 == 0
                        else ((candidate, candidate_samples), (baseline, baseline_samples))
                    )
                    for subject, destination in order:
                        destination.append(
                            await self._event_value(
                                subject,
                                phase=f"confirmation:{candidate.source_digest}",
                                index=confirmation_calls,
                            )
                        )
                        confirmation_calls += 1
                candidate.confirmation_samples_us = tuple(candidate_samples)
                gate = evaluate_performance_gate(
                    baseline_samples,
                    candidate_samples,
                    minimum_sessions=self.confirmation_sessions,
                )
                gates[digest] = gate
                confirmed.append(candidate)
            except Exception:
                continue

        confirmed.sort(key=lambda item: statistics.median(item.confirmation_samples_us))
        winner = confirmed[0] if confirmed else None
        performance_gate = gates.get(winner.source_digest) if winner else None
        ncu_targets = self.policy.ncu_targets(confirmed)

        nsys_targets: list[CandidateEvaluation] = []
        if winner is not None:
            nsys_targets.append(winner)
        anomaly = by_digest.get(decision.anomaly_source_digest or "")
        if anomaly is not None and anomaly not in nsys_targets:
            nsys_targets.append(anomaly)
        diagnostic_results: list[dict[str, Any]] = []
        selectors: dict[str, str] = {}
        nsys_calls = 0
        ncu_calls = 0
        for candidate in nsys_targets:
            kernel_names = candidate.compilation_metrics.kernel_names
            if len(kernel_names) != 1:
                result = {
                    "status": "failed",
                    "reason_code": "kernel_not_found",
                    "summary": None,
                }
            else:
                result = await self._profile(
                    candidate,
                    plan_id="nsys_timeline_v1",
                    kernel_filter=kernel_names[0],
                )
                nsys_calls += 1
            candidate.diagnostics["nsys_timeline_v1"] = result
            diagnostic_results.append(result)
            summary = result.get("summary") or {}
            selector = safe_kernel_base_name(str(summary.get("kernel_name") or ""))
            if selector is None and len(kernel_names) == 1:
                selector = kernel_names[0]
            if selector:
                selectors[candidate.source_digest] = selector

        winner_selector = selectors.get(winner.source_digest) if winner else None
        for candidate in ncu_targets:
            selector = selectors.get(candidate.source_digest) or winner_selector
            if selector is None:
                result = {
                    "status": "failed",
                    "reason_code": "kernel_not_found",
                    "summary": None,
                }
            else:
                result = await self._profile(
                    candidate,
                    plan_id="ncu_triage_v1",
                    kernel_filter=selector,
                )
                ncu_calls += 1
            candidate.diagnostics["ncu_triage_v1"] = result
            diagnostic_results.append(result)

        diagnostic_status = self._diagnostic_status(diagnostic_results)
        diagnostic_candidates = self.policy.deduplicate(
            [*nsys_targets, *ncu_targets]
        )
        for candidate in diagnostic_candidates:
            candidate.diagnostic_status = diagnostic_status

        record_event(
            "funnel_decision",
            status="ok" if winner else "error",
            data={
                "eligible_candidates": len(decision.eligible_source_digests),
                "confirmation_candidates": len(decision.confirmation_source_digests),
                "confirmed_candidates": len(confirmed),
                "anomaly_selected": decision.anomaly_source_digest is not None,
                "winner_source_digest": winner.source_digest if winner else None,
                "discovery_calls": sum(
                    len(candidate.discovery_samples_us) for candidate in unique
                ),
                "confirmation_calls": confirmation_calls,
                "nsys_calls": nsys_calls,
                "ncu_calls": ncu_calls,
            },
        )
        return FunnelResult(
            decision=decision,
            candidates=tuple(unique),
            winner_source_digest=winner.source_digest if winner else None,
            confirmed_source_digests=tuple(item.source_digest for item in confirmed),
            performance_gate=performance_gate,
            discovery_calls=sum(len(item.discovery_samples_us) for item in unique),
            confirmation_calls=confirmation_calls,
            nsys_calls=nsys_calls,
            ncu_calls=ncu_calls,
        )


class TrustedLocalCandidateEvaluator:
    """Explicit trusted-local compatibility adapter; never a sandbox fallback."""

    def __init__(self, profiler_backend: Any, *, policy: FunnelPolicy | None = None) -> None:
        self.profiler_backend = profiler_backend
        self.policy = policy or FunnelPolicy()
        self._cache: dict[str, CandidateEvaluation] = {}
        self._paths: dict[str, Path] = {}

    async def evaluate(self, filepath: Path) -> CandidateEvaluation:
        digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        result = await self.profiler_backend.profile(filepath)
        evaluation = CandidateEvaluation(
            candidate_id=digest,
            source_digest=digest,
            execution_status=(
                CandidateStageStatus.SUCCEEDED
                if result.available
                else CandidateStageStatus.FAILED
            ),
            correctness_status=(
                CandidateStageStatus.SUCCEEDED
                if result.available
                else CandidateStageStatus.NOT_RUN
            ),
            events_status=(
                CandidateStageStatus.SUCCEEDED
                if result.available
                else CandidateStageStatus.FAILED
            ),
            measurement=result.measurement,
            discovery_samples_us=(
                tuple(result.measurement.samples)
                if result.measurement and result.measurement.samples
                else ()
            ),
            reason_code="none" if result.available else "events_failed",
        )
        self._cache[digest] = evaluation
        self._paths[digest] = filepath
        return evaluation

    async def finalize(
        self,
        baseline: CandidateEvaluation,
        candidates: list[CandidateEvaluation],
    ) -> FunnelResult:
        decision = self.policy.decide(candidates)
        by_digest = {item.source_digest: item for item in candidates}
        confirmed: list[CandidateEvaluation] = []
        gates: dict[str, PerformanceGateResult] = {}
        baseline_path = self._paths.get(baseline.source_digest)
        confirmation_calls = 0
        for digest in decision.confirmation_source_digests:
            candidate = by_digest.get(digest)
            candidate_path = self._paths.get(digest)
            if candidate is None or candidate_path is None or baseline_path is None:
                continue
            confirm_pair = getattr(self.profiler_backend, "confirm_pair", None)
            if not callable(confirm_pair):
                continue
            try:
                gates[digest] = await confirm_pair(baseline_path, candidate_path)
                sessions = int(
                    getattr(self.profiler_backend, "confirmation_sessions", 5)
                )
                candidate.confirmation_samples_us = (candidate.measurement.value,) * sessions  # type: ignore[union-attr]
                confirmation_calls += sessions * 2
                confirmed.append(candidate)
            except Exception:
                continue
        confirmed.sort(key=lambda item: item.measurement.value)  # type: ignore[union-attr]
        winner = confirmed[0] if confirmed else None
        return FunnelResult(
            decision=decision,
            candidates=tuple(self.policy.deduplicate(candidates)),
            winner_source_digest=winner.source_digest if winner else None,
            confirmed_source_digests=tuple(item.source_digest for item in confirmed),
            performance_gate=gates.get(winner.source_digest) if winner else None,
            discovery_calls=sum(len(item.discovery_samples_us) for item in candidates),
            confirmation_calls=confirmation_calls,
        )


__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluationError",
    "CandidateEvaluator",
    "CandidateStageStatus",
    "CompilationMetrics",
    "FunnelDecision",
    "FunnelPolicy",
    "FunnelResult",
    "SandboxCandidateEvaluator",
    "TrustedLocalCandidateEvaluator",
    "candidate_kernel_names",
    "parse_ptxas_metrics",
    "safe_kernel_base_name",
]
