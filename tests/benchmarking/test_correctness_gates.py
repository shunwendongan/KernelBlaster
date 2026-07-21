from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cuda_gates", ROOT / "scripts" / "benchmark_cuda.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _variant(max_error: float, p99_error: float):
    return {
        "correctness": [
            {
                "driver": "extra-0-edge",
                "mode": "normalized-extra-0-edge",
                "passed": True,
                "metrics": {
                    "max_abs_error": max_error,
                    "p99_abs_error": p99_error,
                    "finite": True,
                    "deterministic": True,
                },
            }
        ]
    }


def test_correctness_error_gate_accepts_ten_percent_plus_floor():
    gate = MODULE._validate_correctness_error_regression(
        {
            "baseline": _variant(0.01, 0.001),
            "candidate": _variant(0.01105, 0.00115),
        },
        candidate_label="candidate",
    )
    assert gate["status"] == "passed"


def test_correctness_error_gate_rejects_regression():
    with pytest.raises(RuntimeError, match="regressed"):
        MODULE._validate_correctness_error_regression(
            {
                "baseline": _variant(0.01, 0.001),
                "candidate": _variant(0.02, 0.01),
            },
            candidate_label="candidate",
        )


def _comparison(*, gate_passed: bool = True, formal_valid: bool = True):
    return {
        "formal_valid": formal_valid,
        "all_sessions_not_slower": True,
        "performance_gate": {"passed": gate_passed},
    }


def test_confirmation_requires_five_independent_sessions():
    with pytest.raises(ValueError, match="at least 5"):
        MODULE._validate_session_protocol("confirmation", 4)
    MODULE._validate_session_protocol("confirmation", 5)


def test_correctness_only_does_not_require_timing_sessions():
    MODULE._validate_session_protocol(
        "confirmation", 1, correctness_only=True
    )


def test_discovery_is_valid_but_never_a_formal_claim():
    result = MODULE._classify_benchmark_result(
        phase="discovery", stable=True, comparison=_comparison()
    )
    assert result == {
        "outcome": "completed",
        "execution_valid": True,
        "performance_claim_allowed": False,
    }


def test_stable_confirmation_without_gain_is_no_improvement():
    result = MODULE._classify_benchmark_result(
        phase="confirmation", stable=True, comparison=_comparison(gate_passed=False)
    )
    assert result["outcome"] == "no_improvement"
    assert result["execution_valid"] is True
    assert result["performance_claim_allowed"] is False


@pytest.mark.parametrize(
    ("stable", "formal_valid"),
    [(False, True), (True, False)],
)
def test_unstable_or_invalid_comparison_is_inconclusive(stable, formal_valid):
    result = MODULE._classify_benchmark_result(
        phase="confirmation",
        stable=stable,
        comparison=_comparison(formal_valid=formal_valid),
    )
    assert result["outcome"] == "inconclusive"
    assert result["execution_valid"] is False
    assert result["performance_claim_allowed"] is False
