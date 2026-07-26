from src.kernelblaster.baseline_jobs.ranking import (
    HardwareRankingKey,
    PairedWorkload,
    evaluate_multi_workload_gate,
    select_hardware_winner,
)


def _workload(name, baseline, candidate, *, weight=1, core=True, host=1000):
    return PairedWorkload(
        workload_id=name,
        weight=weight,
        core=core,
        baseline_device_us=(baseline,) * 5,
        candidate_device_us=(candidate,) * 5,
        baseline_host_us=(host,) * 5,
        candidate_host_us=(host * 10,) * 5,
    )


def test_strict_gate_uses_device_geomean_bootstrap_and_ignores_host_for_ranking():
    gate = evaluate_multi_workload_gate(
        (
            _workload("hot", 10, 9, weight=0.7),
            _workload("rotating", 20, 18, weight=0.3),
        )
    )
    assert gate.qualified
    assert gate.geometric_mean_speedup >= 1.05
    assert gate.bootstrap_95_lower > 1
    assert gate.host_time_diagnostic_only


def test_one_core_regression_rejects_a_fast_geometric_mean():
    gate = evaluate_multi_workload_gate(
        (
            _workload("dominant", 100, 50, weight=0.9),
            _workload("regressed", 10, 10.01, weight=0.1),
        )
    )
    assert gate.geometric_mean_speedup > 1.05
    assert not gate.all_core_no_regression
    assert not gate.qualified


def test_winners_are_separate_per_hardware_class_and_triton_is_not_primary():
    gate = evaluate_multi_workload_gate((_workload("hot", 10, 9),))
    cuda_key = HardwareRankingKey(
        hardware_fingerprint="gpu-a",
        direction="forward",
        numerics_class="exact",
        determinism="bitwise",
        backend="cuda",
    )
    triton_key = HardwareRankingKey(**{**cuda_key.__dict__, "backend": "triton"})
    assert select_hardware_winner(cuda_key, (("a" * 64, gate),)).primary_cuda_winner
    assert not select_hardware_winner(triton_key, (("b" * 64, gate),)).primary_cuda_winner
