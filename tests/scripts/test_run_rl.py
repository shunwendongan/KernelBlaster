from __future__ import annotations

from unittest.mock import patch

from scripts.run_RL import resolve_target_gpu
from src.kernelblaster.config import GPUType


def test_explicit_remote_gpu_does_not_probe_local_hardware():
    with patch.object(GPUType, "current", side_effect=AssertionError("unexpected probe")):
        assert resolve_target_gpu("l40s") is GPUType.L40S


def test_unspecified_remote_gpu_uses_local_hardware():
    with patch.object(GPUType, "current", return_value=GPUType.RTX3080):
        assert resolve_target_gpu(None) is GPUType.RTX3080
