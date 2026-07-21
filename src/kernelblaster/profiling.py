from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Protocol


class ProfilingMode(str, Enum):
    NCU = "ncu"
    EVENTS_ONLY = "events_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ProfilingResult:
    mode: ProfilingMode
    elapsed_cycles: int | None = None
    elapsed_us: float | None = None
    annotated_source: str = ""
    raw_output: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and (
            self.elapsed_cycles is not None or self.elapsed_us is not None
        )


class ProfilerBackend(Protocol):
    mode: ProfilingMode

    async def profile(self, filepath: Path) -> ProfilingResult: ...


@dataclass(frozen=True)
class PerformanceGateResult:
    passed: bool
    median_speedup: float | None
    bootstrap_95_lower: float | None
    bootstrap_95_upper: float | None
    session_speedups: tuple[float, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "median_speedup": self.median_speedup,
            "bootstrap_95_lower": self.bootstrap_95_lower,
            "bootstrap_95_upper": self.bootstrap_95_upper,
            "session_speedups": list(self.session_speedups),
            "reason": self.reason,
        }


def paired_bootstrap_interval(
    speedups: list[float],
    *,
    seed: int = 20260719,
    resamples: int = 10_000,
) -> tuple[float, float] | None:
    if len(speedups) < 2:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(speedups, k=len(speedups)))
        for _ in range(resamples)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )


def evaluate_performance_gate(
    baseline_session_us: list[float],
    candidate_session_us: list[float],
    *,
    minimum_sessions: int = 5,
    minimum_speedup: float = 1.01,
) -> PerformanceGateResult:
    if len(baseline_session_us) != len(candidate_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Baseline and candidate confirmation sessions are not paired.",
        )
    if len(baseline_session_us) < minimum_sessions:
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason=f"At least {minimum_sessions} confirmation sessions are required.",
        )
    if any(value <= 0 or not math.isfinite(value) for value in baseline_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Baseline sessions contain a non-positive or non-finite latency.",
        )
    if any(value <= 0 or not math.isfinite(value) for value in candidate_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Candidate sessions contain a non-positive or non-finite latency.",
        )

    speedups = [
        baseline / candidate
        for baseline, candidate in zip(
            baseline_session_us, candidate_session_us, strict=True
        )
    ]
    interval = paired_bootstrap_interval(speedups)
    median_speedup = statistics.median(speedups)
    lower, upper = interval if interval is not None else (None, None)
    passed = bool(
        median_speedup >= minimum_speedup
        and lower is not None
        and lower > 1.0
    )
    reason = None
    if not passed:
        reason = (
            f"Performance gate failed: median speedup={median_speedup:.6f}, "
            f"bootstrap_95_lower={lower}."
        )
    return PerformanceGateResult(
        passed=passed,
        median_speedup=median_speedup,
        bootstrap_95_lower=lower,
        bootstrap_95_upper=upper,
        session_speedups=tuple(speedups),
        reason=reason,
    )


class EventsRunner(Protocol):
    async def __call__(
        self,
        filepath: Path,
        *,
        sessions: int,
    ) -> list[float]: ...


class EventsProfilerBackend:
    """Search and confirmation backend driven by CUDA Events session medians."""

    mode = ProfilingMode.EVENTS_ONLY

    def __init__(
        self,
        runner: EventsRunner,
        *,
        discovery_sessions: int = 3,
        confirmation_sessions: int = 5,
    ) -> None:
        self.runner = runner
        self.discovery_sessions = discovery_sessions
        self.confirmation_sessions = confirmation_sessions

    async def profile(self, filepath: Path) -> ProfilingResult:
        samples = await self.runner(filepath, sessions=self.discovery_sessions)
        if len(samples) != self.discovery_sessions:
            return ProfilingResult(
                mode=self.mode,
                error="CUDA Events runner returned an unexpected session count.",
            )
        median_us = statistics.median(samples)
        return ProfilingResult(
            mode=self.mode,
            elapsed_us=median_us,
            metrics={"session_medians_us": samples, "phase": "discovery"},
        )

    async def confirm_pair(
        self,
        baseline: Path,
        candidate: Path,
    ) -> PerformanceGateResult:
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for session in range(self.confirmation_sessions):
            order = (
                ((baseline, baseline_samples), (candidate, candidate_samples))
                if session % 2 == 0
                else ((candidate, candidate_samples), (baseline, baseline_samples))
            )
            for filepath, destination in order:
                values = await self.runner(filepath, sessions=1)
                if len(values) != 1:
                    raise ProfilerUnavailable(
                        "CUDA Events confirmation runner did not return one session."
                    )
                destination.append(values[0])
        return evaluate_performance_gate(
            baseline_samples,
            candidate_samples,
            minimum_sessions=self.confirmation_sessions,
        )


class CudaEventsRunner:
    """Compile an instrumented correctness driver and execute separate sessions."""

    def __init__(
        self,
        *,
        driver_path: Path,
        gpu: Any,
        logger: Any,
        work_dir: Path,
        warmup: int = 20,
        repetitions: int = 100,
        seed: int = 20260719,
        timeout: float = 1200,
    ) -> None:
        self.driver_path = driver_path
        self.gpu = gpu
        self.logger = logger
        self.work_dir = work_dir
        self.warmup = warmup
        self.repetitions = repetitions
        self.seed = seed
        self.timeout = timeout
        self._counter = 0
        self._compiled: dict[tuple[str, str], Path] = {}

    async def __call__(self, filepath: Path, *, sessions: int) -> list[float]:
        from .agents.utils import (
            NamedTimer,
            compile_and_run_cu_file,
            run_gpu_executable,
        )
        from .benchmarking import BENCHMARK_MARKER, instrument_driver

        source_digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        cache_key = (str(filepath.resolve()), source_digest)
        compiled = self._compiled.get(cache_key)
        if compiled is None:
            self._counter += 1
            driver = instrument_driver(
                self.driver_path.read_text(encoding="utf-8"),
                seed=self.seed,
                warmup=self.warmup,
                repetitions=self.repetitions,
                inner_loops=0,
            )
            instrumented_path = self.work_dir / (
                f"events_driver_{self._counter}_{filepath.stem}.cpp"
            )
            instrumented_path.parent.mkdir(parents=True, exist_ok=True)
            instrumented_path.write_text(driver, encoding="utf-8")
            stdout, _stderr, binary, success = await compile_and_run_cu_file(
                instrumented_path,
                filepath,
                self.gpu,
                NamedTimer(),
                self.logger,
                persistent_artifacts=True,
                timeout=self.timeout,
                num_runs=sessions,
                passed_keyword="passed",
            )
            if not success:
                raise ProfilerUnavailable("CUDA Events correctness execution failed.")
            compiled = Path(binary)
            self._compiled[cache_key] = compiled
        else:
            raw_stdout, _stderr = await run_gpu_executable(
                compiled,
                self.gpu,
                self.timeout,
                job_name=f"{filepath} (CUDA Events confirmation)",
                n_runs=sessions,
            )
            stdout = [raw_stdout] if isinstance(raw_stdout, str) else raw_stdout
        values: list[float] = []
        for session_stdout in stdout:
            markers = [
                line
                for line in session_stdout.splitlines()
                if line.startswith(BENCHMARK_MARKER)
            ]
            if len(markers) != 1:
                raise ProfilerUnavailable(
                    "CUDA Events output did not contain exactly one benchmark marker."
                )
            payload = json.loads(markers[0][len(BENCHMARK_MARKER) :])
            samples = [float(value) for value in payload["samples_us"]]
            if not samples:
                raise ProfilerUnavailable("CUDA Events session contained no samples.")
            values.append(statistics.median(samples))
        return values


class NCUFallbackProfilerBackend:
    """Use NCU when available and downgrade only permission failures to Events."""

    def __init__(
        self,
        ncu_backend: ProfilerBackend,
        events_backend: EventsProfilerBackend,
    ) -> None:
        self.ncu_backend = ncu_backend
        self.events_backend = events_backend
        self.mode = ncu_backend.mode

    async def profile(self, filepath: Path) -> ProfilingResult:
        result = await self.ncu_backend.profile(filepath)
        combined_error = "\n".join(
            value for value in (result.error, result.stderr, result.raw_output) if value
        )
        if result.available or not ncu_permission_blocked(combined_error):
            self.mode = result.mode
            return result
        self.mode = ProfilingMode.EVENTS_ONLY
        return await self.events_backend.profile(filepath)

    async def confirm_pair(
        self,
        baseline: Path,
        candidate: Path,
    ) -> PerformanceGateResult:
        return await self.events_backend.confirm_pair(baseline, candidate)


class ProfilerUnavailable(RuntimeError):
    pass


def ncu_permission_blocked(output: str) -> bool:
    return "ERR_NVGPUCTRPERM" in output


__all__ = [
    "CudaEventsRunner",
    "EventsProfilerBackend",
    "NCUFallbackProfilerBackend",
    "PerformanceGateResult",
    "ProfilerBackend",
    "ProfilerUnavailable",
    "ProfilingMode",
    "ProfilingResult",
    "evaluate_performance_gate",
    "ncu_permission_blocked",
    "paired_bootstrap_interval",
]
