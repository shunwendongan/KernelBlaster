"""Device-time-only multi-workload ranking and strict confirmation gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Literal


@dataclass(frozen=True)
class PairedWorkload:
    workload_id: str
    weight: float
    core: bool
    baseline_device_us: tuple[float, ...]
    candidate_device_us: tuple[float, ...]
    baseline_host_us: tuple[float, ...] = ()
    candidate_host_us: tuple[float, ...] = ()


@dataclass(frozen=True)
class WorkloadGate:
    workload_id: str
    median_baseline_device_us: float
    median_candidate_device_us: float
    median_speedup: float
    core_no_regression: bool
    median_baseline_host_us: float | None
    median_candidate_host_us: float | None


@dataclass(frozen=True)
class MultiWorkloadGateResult:
    qualified: bool
    objective: Literal["latency", "throughput"]
    confirmation_sessions: int
    geometric_mean_speedup: float
    bootstrap_95_lower: float
    bootstrap_95_upper: float
    all_core_no_regression: bool
    workloads: tuple[WorkloadGate, ...]
    ranking_source: Literal["cuda_events"] = "cuda_events"
    host_time_diagnostic_only: bool = True
    baseline_provider: Literal["upstream_cuda"] = "upstream_cuda"


def _bootstrap(values: tuple[float, ...], *, samples: int = 4000) -> tuple[float, float]:
    generator = random.Random(20260726)
    medians = sorted(
        statistics.median(values[generator.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    return medians[int(samples * 0.025)], medians[min(samples - 1, int(samples * 0.975))]


def evaluate_multi_workload_gate(
    workloads: tuple[PairedWorkload, ...],
    *,
    objective: Literal["latency", "throughput"] = "latency",
    confirmation_sessions: int = 5,
    baseline_provider: Literal["upstream_cuda"] = "upstream_cuda",
) -> MultiWorkloadGateResult:
    if baseline_provider != "upstream_cuda":
        raise ValueError("formal ranking baseline must be upstream_cuda")
    if not workloads:
        raise ValueError("confirmation requires at least one workload")
    if confirmation_sessions != 5:
        raise ValueError("formal confirmation requires exactly five paired sessions")
    if any(item.weight <= 0 for item in workloads):
        raise ValueError("workload weights must be positive")
    if any(
        len(item.baseline_device_us) != confirmation_sessions
        or len(item.candidate_device_us) != confirmation_sessions
        for item in workloads
    ):
        raise ValueError("every workload requires five paired device sessions")
    total_weight = sum(item.weight for item in workloads)
    gates: list[WorkloadGate] = []
    per_session: list[float] = []
    for session in range(confirmation_sessions):
        per_session.append(
            math.exp(
                sum(
                    item.weight
                    * math.log(
                        item.baseline_device_us[session] / item.candidate_device_us[session]
                    )
                    for item in workloads
                )
                / total_weight
            )
        )
    for item in workloads:
        baseline = statistics.median(item.baseline_device_us)
        candidate = statistics.median(item.candidate_device_us)
        gates.append(
            WorkloadGate(
                workload_id=item.workload_id,
                median_baseline_device_us=baseline,
                median_candidate_device_us=candidate,
                median_speedup=baseline / candidate,
                core_no_regression=(not item.core or candidate <= baseline),
                median_baseline_host_us=(
                    statistics.median(item.baseline_host_us) if item.baseline_host_us else None
                ),
                median_candidate_host_us=(
                    statistics.median(item.candidate_host_us) if item.candidate_host_us else None
                ),
            )
        )
    aggregate = statistics.median(per_session)
    lower, upper = _bootstrap(tuple(per_session))
    no_regression = all(item.core_no_regression for item in gates)
    return MultiWorkloadGateResult(
        qualified=aggregate >= 1.05 and lower > 1.0 and no_regression,
        objective=objective,
        confirmation_sessions=confirmation_sessions,
        geometric_mean_speedup=aggregate,
        bootstrap_95_lower=lower,
        bootstrap_95_upper=upper,
        all_core_no_regression=no_regression,
        workloads=tuple(gates),
    )


@dataclass(frozen=True)
class HardwareRankingKey:
    hardware_fingerprint: str
    direction: Literal["forward", "backward"]
    numerics_class: Literal["exact", "approximate", "quantized"]
    determinism: Literal["bitwise", "bounded"]
    backend: Literal["cuda", "triton"]


@dataclass(frozen=True)
class HardwareWinner:
    key: HardwareRankingKey
    candidate_digest: str
    gate: MultiWorkloadGateResult
    primary_cuda_winner: bool


def select_hardware_winner(
    key: HardwareRankingKey,
    candidates: tuple[tuple[str, MultiWorkloadGateResult], ...],
) -> HardwareWinner | None:
    qualified = [(digest, gate) for digest, gate in candidates if gate.qualified]
    if not qualified:
        return None
    digest, gate = max(qualified, key=lambda item: item[1].geometric_mean_speedup)
    return HardwareWinner(
        key=key,
        candidate_digest=digest,
        gate=gate,
        primary_cuda_winner=key.backend == "cuda",
    )


__all__ = [
    "HardwareRankingKey",
    "HardwareWinner",
    "MultiWorkloadGateResult",
    "PairedWorkload",
    "WorkloadGate",
    "evaluate_multi_workload_gate",
    "select_hardware_winner",
]
