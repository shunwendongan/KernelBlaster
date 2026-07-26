from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from loguru import logger

from src.kernelblaster.config import GPUType
from src.kernelblaster.outcomes import RunOutcome, RunStatus


MODULE = importlib.import_module(
    "src.kernelblaster.graph.nodes.optimization_rl_ncu"
)


@pytest.mark.asyncio
async def test_sandbox_graph_never_reads_a_local_private_driver(monkeypatch, tmp_path):
    folder = tmp_path / "level1" / "036"
    folder.mkdir(parents=True)
    source = tmp_path / "init.cu"
    source.write_text("// candidate", encoding="utf-8")
    forbidden_driver = tmp_path / "private-driver.cpp"
    captured = {}

    class Runtime:
        execution_backend = SimpleNamespace(value="sandbox")

        def create_candidate_evaluator(self, **kwargs):
            captured["runtime"] = kwargs
            assert not kwargs["driver_path"].exists()
            return object()

    class Agent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs
            assert not kwargs["fb_config"].test_code_fp.exists()
            self.num_rl_iterations = 0

        async def initialize(self):
            return None

        async def run(self):
            return RunOutcome(status=RunStatus.FAILED, reason="unit")

    monkeypatch.setattr(MODULE, "RLNCUAgent", Agent)
    state = {
        "task_id": "036",
        "folder": folder,
        "cuda_fp": source,
        "test_code_fp": forbidden_driver,
        "logger": logger,
        "model": "unit-model",
        "gpu": GPUType.RTX3080,
        "retry_failed": False,
        "rl_rollout_steps": 1,
        "rl_buffer_size": 1,
        "rl_update_frequency": 1,
        "rl_iterations": 1,
    }
    result = await MODULE.optimization_rl_ncu(state, runtime=Runtime())
    assert result["run_outcome"]["status"] == "failed"
    assert captured["runtime"]["run_id"].startswith("agent-036-")
