# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import time
import asyncio
import loguru
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator
import shutil

from ..graph import build_graph
from ..config import config, WorkflowConfig
from ..graph.state import save_state_to_json
from ..outcomes import RunOutcome, RunStatus

__all__ = ["WorkflowResult", "run_workflow"]


@dataclass
class WorkflowResult:
    config: WorkflowConfig
    rl_cuda_perf_filepath: Path = None  # RL-optimized CUDA code
    outcome: RunOutcome = field(
        default_factory=lambda: RunOutcome(
            status=RunStatus.FAILED,
            reason="Failed code generation due to an error or reaching the maximum number of attempts.",
        )
    )

    @property
    def error(self) -> str:
        return self.outcome.reason or self.outcome.status.value

    @property
    def timeout(self) -> bool:
        return self.outcome.status is RunStatus.TIMEOUT

    def set_outcome(self, outcome: RunOutcome):
        self.outcome = outcome
        self.rl_cuda_perf_filepath = outcome.artifact_path if outcome.success else None

    @property
    def success(self) -> bool:
        return self.outcome.success

    def agents(self) -> Iterator[str]:
        """
        Returns the names of all available agents.
        Currently only supports RL optimization.
        """
        if hasattr(self, "rl_cuda_perf_filepath"):
            yield "rl_cuda_perf"

    def running_agents(self) -> Iterator[str]:
        """
        Returns the names of the agents that are supposed to be running.
        """
        # RL optimization always runs if enabled
        yield "rl_cuda_perf"

    @property
    def generated_codes(
        self,
    ) -> dict[str, str]:
        def stringify(filepath: Path | None) -> str | None:
            if filepath is None:
                return None
            return str(filepath)

        # Return dict with RL-optimized CUDA code filepath
        return {"rl_cuda_perf": stringify(self.rl_cuda_perf_filepath)}

    def write_failures(
        self,
        folder: str,
    ):
        if not self.success:
            (folder / "failed_rl_cuda_perf").write_text(self.error, encoding="utf-8")
            finished = folder / "rl_ncu" / ".finished"
            finished.parent.mkdir(parents=True, exist_ok=True)
            finished.write_text(self.outcome.status.value + "\n", encoding="utf-8")

    def remove_existing_files(self, folder: Path):
        failed_file = folder / "failed_rl_cuda_perf"
        if failed_file.exists() and self.config.retry_failed:
            # Remove the agent folder if the retry_failed flag is set and the agent failed.
            shutil.rmtree(folder / "rl_ncu", ignore_errors=True)
        # This file should be removed regardless of the retry_failed flag. It will be recreated by the agents themselves if their folder does not contain a successful file.
        failed_file.unlink(missing_ok=True)


async def run_workflow(
    task_id: str,
    user_message: str,
    reference_code: str,
    folder: Path,
    workflow_config: WorkflowConfig,
    job_logger: loguru.Logger,
    timeout_seconds: int,
    shared_database=None,
) -> WorkflowResult:

    folder.mkdir(exist_ok=True, parents=True)
    start = time.time()

    job_logger.info(f"Starting workflow for task {task_id}.")
    config.print_config(job_logger)

    result = WorkflowResult(config=workflow_config)

    # Prepare output directory for the run
    result.remove_existing_files(folder)

    workflow = build_graph()
    workflow_input = {
        "user_message": user_message,
        "reference_code": reference_code,
        "folder": folder,
        "logger": job_logger,
        "model": workflow_config.model,
        # Pass shared database directly from caller (runner)
        "shared_optimization_database": shared_database,
        **workflow_config.dict(),
    }

    try:
        final_state = await asyncio.wait_for(
            workflow.ainvoke(workflow_input),
            timeout=timeout_seconds,
        )
        save_state_to_json(final_state, folder / "state.json")
        outcome_payload = final_state.get("run_outcome")
        outcome = (
            RunOutcome.from_dict(outcome_payload)
            if outcome_payload
            else RunOutcome(
                status=RunStatus.FAILED,
                reason="Workflow completed without a terminal run outcome.",
            )
        )
        result = WorkflowResult(
            config=workflow_config,
            rl_cuda_perf_filepath=(outcome.artifact_path if outcome.success else None),
            outcome=outcome,
        )
    except asyncio.TimeoutError:
        result.set_outcome(
            RunOutcome(
                status=RunStatus.TIMEOUT,
                reason=f"Timeout after {timeout_seconds / 60} minutes",
            )
        )
    except Exception as error:
        job_logger.exception(f"Workflow failed for task {task_id}: {error}")
        result.set_outcome(
            RunOutcome(
                status=RunStatus.FAILED,
                reason=f"{type(error).__name__}: {error}",
            )
        )

    # Successes will be written by the agents themselves
    # We write the failures here instead of inside the agents incase of exceptions or timeouts.
    result.write_failures(folder)
    duration = time.time() - start
    job_logger.info(f"Workflow completed in {duration:0.2f} seconds")
    return result
