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

"""把工作流状态转换为 RL 优化任务，组织输入产物、Profiler 和最终结果。"""

from pathlib import Path
from ...agents import FeedbackConfig
from ...agents.opt_ncu_rl import RLNCUAgent
from ..state import GraphState, save_state_to_json
from ...outcomes import RunOutcome, RunStatus
from ...profiling import CudaEventsRunner, EventsProfilerBackend


async def optimization_rl_ncu(state: GraphState, *, runtime=None):
    """
    解析一个 KernelBench-CUDA 任务并运行 RLNCUAgent 优化循环。

    名称中的 ``ncu`` 来自早期实现；当前排名信号可以是 CUDA Events。
    NCU/NSYS 仅在可用时提供诊断，不能覆盖 correctness 或 timing 终态。
    节点优先读取 ``data/kernelbench-cuda`` 中的 ``init.cu`` 与
    ``driver.cpp``。``runtime is None`` 或显式 ``trusted_local`` 使用经典
    Driver 执行；安全 sandbox runtime 在 ``master`` 上会 fail closed，等待结构化
    CandidateEvaluator 接管生成候选，绝不会静默退回本地执行。

    参数:
        state: 工作流节点读取并按约定更新的共享状态。

    返回:
        仅更新 ``rl_ncu_cuda_fp`` 和序列化 ``run_outcome`` 的状态片段。
    """
    base_folder = Path(state["folder"])
    base_folder.mkdir(parents=True, exist_ok=True)
    
    # 从输出目录的 <level>/<problem_name> 结构反推原始任务。源码文件位于
    # <repo>/src/kernelblaster/graph/nodes/，因此 parents[4] 是仓库根目录。
    repo_root = Path(__file__).resolve().parents[4]
    curated_root = Path(state.get("kernelbench_cuda_root", repo_root / "data/kernelbench-cuda"))
    level = base_folder.parent.name
    problem_name = base_folder.name
    curated_dir = curated_root / level / problem_name

    curated_driver_cpp = curated_dir / "driver.cpp"
    curated_init_cu = curated_dir / "init.cu"

    # 后备：允许使用运行文件夹中已存在的文件。
    run_driver_cpp = base_folder / "driver.cpp"
    run_init_cu = base_folder / "init.cu"
    
    # 显式 state 路径优先；否则使用仓库内任务，最后才读取恢复目录。
    cuda_fp = state.get("cuda_fp")
    if cuda_fp is None:
        if curated_init_cu.exists():
            cuda_fp = curated_init_cu
            state["logger"].info(f"Using curated init.cu from {curated_dir} as cuda_fp: {cuda_fp}")
        elif run_init_cu.exists():
            cuda_fp = run_init_cu
            state["logger"].info(f"Using run-folder init.cu as cuda_fp: {cuda_fp}")
        else:
            state["logger"].error(
                f"No cuda_fp available. Required files not found:\n"
                f"  - Curated: {curated_init_cu}\n"
                f"  - Run folder: {run_init_cu}\n"
                f"Skipping problem {problem_name} - curated CUDA files are required."
            )
            outcome = RunOutcome(
                status=RunStatus.FAILED,
                reason=f"Missing CUDA source for {problem_name}",
            )
            return {"rl_ncu_cuda_fp": None, "run_outcome": outcome.to_dict()}

    cuda_fp = Path(cuda_fp)
    
    # Driver 同样按“显式输入 → 仓库任务 → 恢复目录”解析。
    test_code_fp = state.get("test_code_fp")
    if test_code_fp is None:
        if curated_driver_cpp.exists():
            test_code_fp = curated_driver_cpp
            state["logger"].info(f"Using curated driver.cpp from {curated_dir} as test_code_fp: {test_code_fp}")
        elif run_driver_cpp.exists():
            test_code_fp = run_driver_cpp
            state["logger"].info(f"Using run-folder driver.cpp as test_code_fp: {test_code_fp}")
        else:
            state["logger"].error(
                f"No test_code_fp available. Required files not found:\n"
                f"  - Curated: {curated_driver_cpp}\n"
                f"  - Run folder: {run_driver_cpp}\n"
                f"Skipping problem {problem_name} - curated driver.cpp is required."
            )
            outcome = RunOutcome(
                status=RunStatus.FAILED,
                reason=f"Missing correctness driver for {problem_name}",
            )
            return {"rl_ncu_cuda_fp": None, "run_outcome": outcome.to_dict()}
    
    test_code_fp = Path(test_code_fp)
    
    save_state_to_json(state, base_folder / "state.json")

    # FeedbackConfig 只描述本任务；实际执行后端在 profiler_backend 中注入。
    fb_config = FeedbackConfig(
        agent_name="rl_ncu",
        base_folder=base_folder,
        logger=state["logger"],
        init_user_prompt="",  # 这将在 RLNCUAgent.initialize() 中设置
        model=state["model"],
        gpu=state["gpu"],
        test_code_fp=test_code_fp,
        retry_failed=state["retry_failed"],
        num_pgen=4,  # 强化学习代理使用更少的并行编码器，因为它更具战略性
    )
    
    # rollout 参数来自 WorkflowConfig，并随 state.json 一起保留以便审计。
    database_path = base_folder / "optimization_database.md"
    max_rollout_steps = state.get("rl_rollout_steps", 5)
    replay_buffer_size = state.get("rl_buffer_size", 100)
    update_frequency = state.get("rl_update_frequency", 3)
    rl_iterations = state.get("rl_iterations", 10)
    
    if runtime is None:
        events_backend = EventsProfilerBackend(
            CudaEventsRunner(
                driver_path=test_code_fp,
                gpu=state["gpu"],
                logger=state["logger"],
                work_dir=base_folder / "events",
            )
        )
    else:
        events_backend = runtime.create_events_backend(
            driver_path=test_code_fp,
            gpu=state["gpu"],
            logger=state["logger"],
            work_dir=base_folder / "events",
        )

    agent_rl_ncu = RLNCUAgent(
        fb_config=fb_config,
        code_to_optimize_fp=cuda_fp,
        database_path=database_path,
        max_rollout_steps=max_rollout_steps,
        replay_buffer_size=replay_buffer_size,
        update_frequency=update_frequency,
        database=state.get("shared_optimization_database"),
        profiler_backend=events_backend,
    )
    
    # initialize 先建立 correctness-passing 基线；run 才能比较新候选。
    await agent_rl_ncu.initialize()
    
    # 通过多次迭代运行 RL 优化
    state["logger"].info(f"Starting RL optimization with {rl_iterations} iterations")
    
    # 设置代理中的迭代次数
    agent_rl_ncu.num_rl_iterations = rl_iterations
    
    # 运行 RL 代理（它将在内部处理多次迭代）
    outcome = await agent_rl_ncu.run()
    
    if outcome.success:
        state["logger"].info(
            f"RL optimization completed successfully: {outcome.artifact_path}"
        )
    else:
        state["logger"].warning(
            f"RL optimization ended with {outcome.status.value}: {outcome.reason}"
        )

    # 只有标准终态为 success 才复制最终 CUDA 文件。失败和无提升不能产生
    # 看似可用的 final_rl_cuda_perf.cu。
    final_file = None
    if outcome.success:
        final_file = base_folder / "final_rl_cuda_perf.cu"
        final_file.write_text(outcome.artifact_path.read_text(), encoding="utf-8")
        state["logger"].info(f"RL optimization completed. Best result saved to {final_file}")
        outcome = RunOutcome(
            status=outcome.status,
            artifact_path=final_file,
            reason=outcome.reason,
            profiling_mode=outcome.profiling_mode,
            measurement=outcome.measurement,
            execution_status=outcome.execution_status,
            correctness_status=outcome.correctness_status,
            timing_status=outcome.timing_status,
            diagnostic_status=outcome.diagnostic_status,
            reason_code=outcome.reason_code,
            metrics=outcome.metrics,
        )

    save_state_to_json(
        {
            **state,
            "rl_ncu_cuda_fp": final_file,
            "run_outcome": outcome.to_dict(),
        },
        base_folder / "state.json",
    )

    return {"rl_ncu_cuda_fp": final_file, "run_outcome": outcome.to_dict()}
