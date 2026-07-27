from __future__ import annotations

from unittest.mock import patch

from src.kernelblaster.config import gpu_config
from src.kernelblaster.config.gpu_config import GPUType, RuntimeGPU, resolve_gpu


def test_current_gpu_uses_fixed_argv_without_a_shell(monkeypatch):
    monkeypatch.setattr(gpu_config, "_current_gpu", None)
    with patch.object(
        gpu_config.subprocess,
        "check_output",
        return_value=b"NVIDIA GeForce RTX 3080, 8.6\n",
    ) as check_output:
        assert GPUType.current() is GPUType.RTX3080

    check_output.assert_called_once_with(
        [
            "nvidia-smi",
            "--query-gpu=gpu_name,compute_cap",
            "--format=csv,noheader",
        ]
    )


def test_unknown_gpu_uses_runtime_capability_without_extending_enum(monkeypatch):
    monkeypatch.setattr(gpu_config, "_current_gpu", None)
    with patch.object(
        gpu_config.subprocess,
        "check_output",
        return_value=b"Unlisted Future GPU, 9.9\n",
    ):
        resolved = GPUType.current()
    assert isinstance(resolved, RuntimeGPU)
    assert resolved.sm == "sm_99"
    assert resolve_gpu("auto").sm == "sm_99"
