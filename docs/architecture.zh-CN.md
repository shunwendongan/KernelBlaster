# KernelBlaster 源码架构

[English](architecture.md) | **简体中文**

本文描述默认分支 `master` 的静态源码结构。它回答四个问题：入口在哪里、一次任务如何流动、各模块拥有什么状态，以及旧执行路径与安全执行面如何共存。

## 1. 项目不是一条单一流水线

当前仓库同时保留三类能力：

| 层次 | 主要用途 | 关键入口 | 当前边界 |
| --- | --- | --- | --- |
| Agent 研究链 | 用 LLM、Profiler、知识库和 rollout 搜索 CUDA 候选 | `scripts/run_RL.py`、`src/kernelblaster/agents/` | 经典 `trusted_local` 链仍依赖外部编译/GPU 服务 |
| 安全执行面 | 用 Control、CAS、GPU Supervisor、一次性沙箱和 Profiler Worker 承载不可信任务 | `compose.yaml`、`src/kernelblaster/servers/control.py` | `master` 已有 preflight 和执行基础设施，但生成候选漏斗在后续 stacked PR |
| Portfolio 证据链 | 对固定候选执行正确性优先的可复现实验并发布证据 | `scripts/benchmark_*.py`、`portfolio/`、`artifacts/` | 结果受 GPU、shape、dtype、layout 和协议约束 |

这三层共享测量、状态和证据理念，但不应被写成同一个“开箱即用命令”。

## 2. 顶层目录职责

| 路径 | 所有权 | 读者首先应找什么 |
| --- | --- | --- |
| `src/kernelblaster/` | 框架实现 | Workflow、Agent、测量、服务、存储和契约 |
| `scripts/` | 命令编排 | 任务入口、preflight、benchmark、分析和文档同步 |
| `data/kernelbench-cuda/` | 输入任务 | 每题的 `init.cu`、`driver.cpp` 和任务说明 |
| `portfolio/case_studies/` | 已审核候选 | 手工候选、边界 Driver、能力清单和案例说明 |
| `portfolio/suites/` | 实验集合 | Core 10、Pilot 和 RMSNorm 的任务选择与预算 |
| `artifacts/` | 不可变证据 | 环境、原始结果摘要、报告、图表和哈希清单 |
| `docs/` | 使用与审计文档 | 快速开始、架构、状态契约和 Portfolio 解释 |
| `tests/` | 行为契约 | CPU、GPU、沙箱、Profiler、安全和脚本预期行为 |
| `docker/`、`compose.yaml` | 部署边界 | Control、Supervisor、Profiler 和开发容器 |

## 3. Agent 研究链的调用顺序

在经典研究路径中，一次任务按以下顺序流动：

1. `scripts/run_RL.py` 解析数据集、GPU、模型、rollout 和后端参数。
2. `data/dataset.py` 与 `data/kernelbench_cuda.py` 解析任务目录。
3. `run_workflow()` 构造 `GraphState`，设置顶层超时并统一终态。
4. `build_graph()` 当前只连接一个主要节点 `optimization_rl_ncu`。
5. 节点解析 `init.cu` 与 `driver.cpp`，建立 `FeedbackConfig` 和 Profiler backend。
6. `RLNCUAgent.initialize()` 测量初始实现并建立性能状态。
7. `RLNCUAgent.run()` 组织 rollout、候选生成、正确性/性能反馈、Replay Buffer 与知识库更新。
8. `RunOutcome` 把 improved、no improvement、blocked、failed 或 timeout 返回顶层。
9. 成功产物写为 `final_rl_cuda_perf.cu`，过程状态和事件写入运行目录。

需要注意：类名仍包含 `NCU`，但排名来源不必是 NCU。`EventsProfilerBackend` 可以提供低开销 CUDA Events 测量；NCU/NSYS 应被当作诊断证据，不应覆盖正确性或排名终态。

## 4. 安全执行面的职责

安全执行面采用“控制面不执行候选、执行面不持有 LLM 密钥”的分工：

| 模块 | 职责 | 不应拥有的能力 |
| --- | --- | --- |
| `servers/control.py` | Run/Job 状态、lease、CAS API、Supervisor/Profiler 路由 | 不直接执行 CUDA 候选 |
| `storage/repository.py` | SQLite 中的 run/job/attempt 元数据 | 不保存大型源码和报告正文 |
| `storage/cas.py` | 按 SHA-256 保存不可变 Artifact | 不决定任务终态 |
| `gpu_jobs/supervisor.py` | 设备能力、单 GPU 队列、Job 生命周期 | 不接收 Provider Key |
| `gpu_jobs/sandbox.py` | 一次性容器、固定资源、无网络与输出白名单 | 不允许调用方提供任意命令 |
| `profiler_jobs/worker.py` | 固定 NSYS/NCU plan 和结构化摘要 | 不参与 CUDA Events 排名 |
| `preflight/runner.py` | 按顺序验证 Provider、存储、GPU、沙箱、Events 和诊断能力 | 不把部分成功伪装成完整可用 |

`preflight/backends.py` 在 `master` 上是有意 fail-closed 的：`sandbox` 后端不能退回本地 `driver.cpp` 执行。把 Agent 生成源码转成结构化候选并交给沙箱的 CandidateEvaluator 属于后续 stacked PR，因此主线当前的安全基础设施与 Agent 生成循环仍有集成边界。

## 5. 状态与 Artifact 的关系

不要把以下对象混为一谈：

| 对象 | 表达什么 | 典型位置 |
| --- | --- | --- |
| `GraphState` | 单任务节点间的瞬时数据 | `graph/state.py`、任务目录 `state.json` |
| `RunOutcome` | 一次优化任务的标准终态 | `outcomes.py` |
| `Measurement` | 有单位、来源、协议和硬件指纹的单次测量 | `measurements.py` |
| Job/Lease 状态 | 分布式执行的排队、租约和重试事实 | SQLite `JobRepository` |
| CAS Artifact | 源码、二进制、日志、报告等不可变内容 | `ContentAddressedStore` |
| RunRecorder 事件 | 配置、预算、Prompt 元数据与执行事件 | `observability/recorder.py` |
| Portfolio Artifact | 经审核后提交的公开实验结论 | `artifacts/portfolio-v*/` |

`state.json` 方便恢复与排查，但不是性能证据本身；一个 speedup 声明还需要正确性结果、测量协议、硬件、样本稳定性与源码哈希。

## 6. 性能与正确性模块

- `benchmarking.py`：识别 Driver 中的 launch，生成 CUDA Events/NCU 测量变体，并执行性能门控。
- `profiling.py`：统一 Events 与 NCU 结果，显式保留 measurement unit 和不可用原因。
- `result_analysis.py`：把执行、正确性、timing 和 diagnostic 字段归一到测量 schema。
- `servers/cuda_env/correctness_metrics.h`：数值误差、NaN/Inf 和正确性统计。
- `scripts/benchmark_cuda.py`：固定候选的正确性优先 CUDA 对比。
- `scripts/benchmark_candidates.py`：按能力清单批量复现 Core 10 候选。
- `scripts/benchmark_pytorch.py`：同卡 PyTorch 参考列。
- `scripts/analyze_core10_comparison.py`：在前述结果上计算严格对比结论。

## 7. LLM 与知识库

`llm/` 提供 OpenAI-compatible Provider 抽象，负责并发、重试、预算和用量；`agents/utils/query.py` 负责上层消息裁剪与代码提取。密钥只应存在于环境或外部 env 文件中。

`agents/database.py` 维护“性能状态 → 可选优化 → 历史反馈”的知识结构。它帮助搜索复用经验，但模型预测、数据库 confidence 和真实 speedup 是三种不同信号，文档和代码都不应互相替代。

## 8. 推荐阅读顺序

1. `measurements.py` 与 `outcomes.py`：先理解事实如何表达。
2. `config/config.py` 与 `graph/state.py`：理解输入和共享状态。
3. `workflow/workflow.py` 与 `graph/nodes/optimization_rl_ncu.py`：串起任务。
4. `agents/opt_ncu_rl.py`、`rl_agents.py` 与 `database.py`：理解搜索闭环。
5. `benchmarking.py` 与 `profiling.py`：理解正确性优先测量。
6. `preflight/`、`gpu_jobs/`、`profiler_jobs/` 与 `storage/`：理解安全执行面。
7. `portfolio/case_studies/rmsnorm/`：把框架概念映射到一个具体算子。

更细的函数级阅读顺序见 [核心源码阅读指南](source-guide.zh-CN.md)，快速执行入口见 [快速开始](quickstart.zh-CN.md)。
