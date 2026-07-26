"""Independent baseline-provider runtime and ranking contracts."""

from .contracts import (
    BaselineCapabilities,
    BaselineProvider,
    BaselineProvenance,
    BaselineReasonCode,
    BaselineRequest,
    BaselineResult,
    BaselineStatus,
    BaselineWorkloadMeasurement,
)
from .coordinator import BaselineCoordinator, BaselineMatrix
from .ranking import (
    HardwareRankingKey,
    HardwareWinner,
    MultiWorkloadGateResult,
    PairedWorkload,
    evaluate_multi_workload_gate,
    select_hardware_winner,
)
from .search import BackendConvergence, BackendScheduler

__all__ = [
    "BaselineCapabilities",
    "BaselineCoordinator",
    "BaselineMatrix",
    "BaselineProvider",
    "BaselineProvenance",
    "BaselineReasonCode",
    "BaselineRequest",
    "BaselineResult",
    "BaselineStatus",
    "BaselineWorkloadMeasurement",
    "BackendConvergence",
    "BackendScheduler",
    "HardwareRankingKey",
    "HardwareWinner",
    "MultiWorkloadGateResult",
    "PairedWorkload",
    "evaluate_multi_workload_gate",
    "select_hardware_winner",
]
