"""Unbounded project search with independent CUDA/Triton convergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Backend = Literal["cuda", "triton"]


@dataclass
class BackendConvergence:
    backend: Backend
    active: bool = True
    best_discovery_score: float | None = None
    stagnant_rankable: int = 0
    consecutive_unrankable: int = 0
    rankable_total: int = 0
    unrankable_total: int = 0
    blocked_total: int = 0
    reason: str | None = None

    def record_rankable(self, score: float) -> None:
        if not self.active:
            raise RuntimeError("cannot update a converged backend")
        self.rankable_total += 1
        self.consecutive_unrankable = 0
        if self.best_discovery_score is None or score >= self.best_discovery_score * 1.01:
            self.best_discovery_score = score
            self.stagnant_rankable = 0
        else:
            self.best_discovery_score = max(self.best_discovery_score, score)
            self.stagnant_rankable += 1
        if self.stagnant_rankable >= 24:
            self.active = False
            self.reason = "rankable_converged"

    def record_unrankable(self) -> None:
        if not self.active:
            raise RuntimeError("cannot update a converged backend")
        self.unrankable_total += 1
        self.consecutive_unrankable += 1
        if self.consecutive_unrankable >= 50:
            self.active = False
            self.reason = "unrankable_converged"

    def record_blocked(self) -> None:
        """External quota/service interruption is recoverable, never convergence."""
        self.blocked_total += 1


class BackendScheduler:
    """Deterministic 70/30 schedule while both candidate backends are active."""

    _cycle: tuple[Backend, ...] = (
        "cuda", "cuda", "triton", "cuda", "cuda", "triton", "cuda", "cuda", "triton", "cuda"
    )

    def __init__(self, *, triton_supported: bool = True) -> None:
        self.states = {
            "cuda": BackendConvergence("cuda"),
            "triton": BackendConvergence("triton", active=triton_supported),
        }
        if not triton_supported:
            self.states["triton"].reason = "backend_unsupported"
        self._cursor = 0
        self.events_available = True

    def next_backend(self) -> Backend | None:
        if not self.events_available:
            return None
        active = [name for name, state in self.states.items() if state.active]
        if not active:
            return None
        if len(active) == 1:
            return active[0]  # type: ignore[return-value]
        selected = self._cycle[self._cursor % len(self._cycle)]
        self._cursor += 1
        return selected

    def stop_for_events_unavailable(self) -> None:
        self.events_available = False
        for state in self.states.values():
            if state.active:
                state.active = False
                state.reason = "events_unavailable"

    @property
    def complete(self) -> bool:
        return not self.events_available or not any(state.active for state in self.states.values())


__all__ = ["Backend", "BackendConvergence", "BackendScheduler"]
