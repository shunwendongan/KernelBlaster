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

"""组装 LangGraph 状态图，并连接 KernelBlaster 的优化节点。"""

from functools import partial

from langgraph.graph import StateGraph, START, END

from .nodes import optimization_rl_ncu
from .state import GraphState


def build_graph(runtime=None):
    """
    构建当前单节点优化图。

    ``runtime`` 只负责选择候选的执行后端，不改变 Graph 拓扑。默认值
    ``None`` 使用经典本地 Events 路径；显式 runtime 由 ``run_RL.py``
    在完成 backend/capability 检查后注入。

    返回:
        已编译的 LangGraph，输入和输出均遵循 ``GraphState``。
    """
    graph_builder = StateGraph(GraphState)
    
    # 当前 Graph 只有一个写入终态的节点。增加并行节点前必须先定义
    # GraphState 字段的所有权和合并规则，避免候选路径互相覆盖。
    node = (
        optimization_rl_ncu
        if runtime is None
        else partial(optimization_rl_ncu, runtime=runtime)
    )
    graph_builder.add_node("Baseline RL Optimization", node)

    # 所有任务都进入同一条 correctness-first 优化路径。
    graph_builder.add_edge(START, "Baseline RL Optimization")

    # optimization_rl_ncu 必须返回 run_outcome；顶层 workflow 负责兜底。
    graph_builder.add_edge("Baseline RL Optimization", END)

    return graph_builder.compile()
