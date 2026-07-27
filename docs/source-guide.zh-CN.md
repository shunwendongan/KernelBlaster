# KernelBlaster 核心源码阅读指南

[English](source-guide.md) | **简体中文**

本文只解决“按什么顺序读代码”。目录职责见 [源码架构](architecture.zh-CN.md)，CUDA 优化方法见 [高性能算子开发指南](operator-development.zh-CN.md)。

## 阅读前提

默认分支同时保留经典 Agent 研究链和安全执行基础设施。审计时 `master` 位于 PR‑07a；结构化 CandidateEvaluator、通用 Harness 和 Candidate Package 仍在后续 stacked 开发线。不要假设两条路径已经完全接通。

## 1. 先读事实类型

按以下顺序阅读：

1. `src/kernelblaster/measurements.py`
2. `src/kernelblaster/outcomes.py`
3. `docs/measurement-status-contract.zh-CN.md`

先弄清三个区别：

- `Measurement` 表示带单位、来源、协议和硬件指纹的测量；
- `RunOutcome` 表示 improved、no improvement、blocked、failed 或 timeout；
- diagnostic unavailable 不等于 correctness failed，也不能改写 CUDA Events 排名。

## 2. 再读配置和共享状态

阅读：

- `config/config.py`：Provider、服务 URL、token audience 和 Workflow 参数；
- `config/gpu_config.py`：用户 GPU 名称到 `GPUType` 的映射；
- `graph/state.py`：Graph 节点共享字段和 JSON 序列化。

跟踪 `model`、`gpu`、`folder`、`cuda_fp`、`test_code_fp`、rollout 参数、`shared_optimization_database` 和 `run_outcome`。配置描述意图，GraphState 传递过程，RunOutcome 保存最终事实。

## 3. 串起经典 Agent 调用链

核心顺序是：

1. `scripts/run_RL.py::async_main`
2. `scripts/run_RL.py::process_problem`
3. `workflow/workflow.py::run_workflow`
4. `graph/graph.py::build_graph`
5. `graph/nodes/optimization_rl_ncu.py::optimization_rl_ncu`
6. `agents/opt_ncu_rl.py::RLNCUAgent.initialize`
7. `agents/opt_ncu_rl.py::RLNCUAgent.run`

`run_RL.py` 负责参数、数据集、后端和运行记录；Workflow 负责顶层超时和终态；Graph 当前只有一个主优化节点；Agent 才负责候选搜索。

类名中的 `NCU` 是历史命名。当前代码允许 CUDA Events 作为排名信号，NCU/NSYS 只提供诊断。

## 4. 理解 rollout 和优化记忆

建议按调用而不是文件行号阅读：

- `agents/feedback.py`：一次候选尝试的生命周期；
- `agents/rl_agents.py`：`TrajectoryStep`、`Trajectory`、`ReplayBuffer` 和策略更新；
- `agents/database.py`：性能状态、候选优化、相似度、置信度和持久化；
- `agents/utils/query.py`：Prompt 裁剪、代码提取和 Provider 调用；
- `agents/reprofile.py`：经典 NCU 重新分析路径。

阅读时始终区分 predicted improvement、measured speedup、reward 和 database confidence。模型预测不能代替真实测量，失败轨迹也不能被全部丢弃。

## 5. 理解正确性优先测量

阅读：

- `benchmarking.py`：寻找 host launcher、拆分编译单元、Driver 插桩和会话统计；
- `profiling.py`：Events backend、Profiler result 和性能门控；
- `servers/cuda_env/correctness_metrics.h`：误差、NaN/Inf 和数值统计；
- `scripts/benchmark_cuda.py`：固定候选的端到端 benchmark 协议。

重点检查：

- baseline/candidate 是否使用相同输入、设备和计时范围；
- 微秒与 cycles 是否显式区分；
- correctness 是否先于 timing；
- Session 中位数、spread 和 Bootstrap 下界是否共同决定结论；
- Driver 文本扫描对模板、宏、多行调用和括号嵌套是否安全。

## 6. 区分两种执行路径

### 经典可信路径

阅读 `servers/compile.py`、`servers/gpu.py`、`servers/management.py`、`resources/client.py` 和 `agents/utils/commands.py`。这条路径使用外部 Compile/GPU 服务执行受信任的 `init.cu + driver.cpp` 工作流。顶层 Agent 运行器不再拥有子服务生命周期；调用方必须显式配置远端服务 URL，或通过独立 Worker 入口部署服务。

### 安全执行面

阅读顺序：

1. `servers/control.py`
2. `storage/repository.py` 与 `storage/cas.py`
3. `gpu_jobs/contracts.py`、`supervisor.py` 与 `sandbox.py`
4. `profiler_jobs/contracts.py` 与 `worker.py`
5. `preflight/contracts.py`、`runner.py` 与 `backends.py`

安全路径只接受 digest、固定 stage、资源上限和 deadline；生成 Job 不接收 LLM Key、Docker socket 或任意命令。`sandbox` 失败不能自动回退到 `trusted_local`。

## 7. 输出与故障定位

经典任务目录通常包含：

- `state.json`：Graph 状态快照；
- `rl_ncu/`：候选、日志和分析中间产物；
- `final_rl_cuda_perf.cu`：仅在 improved 终态产生；
- `failed_rl_cuda_perf` 与 `.finished`：失败原因和完成标记。

安全路径还包含 SQLite run/job/lease 元数据、CAS Artifact 和 `capability-report/v1`。

推荐排查顺序：

1. 查看 `RunOutcome.status`、`reason_code` 和四类子状态；
2. 查看 capability report 是否过期、硬件不匹配或某个 hard check unavailable；
3. 编译问题看拆分单元、命令和 stderr；
4. 正确性问题看 Driver、边界输入与特殊值；
5. 性能问题看 paired Session、单位、设备并发和系统抖动；
6. Provider 问题看预算、重试和脱敏后的事件元数据；
7. NSYS/NCU 不可用时保留 Events 事实，不伪造完整诊断。

## 8. 修改核心代码必须保持的不变量

1. 未通过正确性的候选不能进入性能排名。
2. 微秒、cycles 和秒不能隐式比较。
3. 成功、异常、取消和超时都必须让 Future 与 RunOutcome 收敛。
4. 候选代码始终按不可信输入处理。
5. sandbox 不得隐式回退 trusted-local。
6. 关键状态和 Artifact 必须可恢复、可追踪并绑定 digest。
7. 预测收益、实测收益和知识库置信度必须分开。
8. Prompt、Token、API Key 和认证头不得进入公开日志。

## 9. 阅读完成后的自测

- 顶层超时发生后，底层子进程在哪一层被取消？
- `NO_IMPROVEMENT` 与 `FAILED` 对统计汇总有什么不同？
- `runtime is None`、`trusted_local` 和 `sandbox` 分别走哪条执行路径？
- NCU permission denied 为什么不应覆盖 CUDA Events 结果？
- Replay Buffer 只保留成功轨迹会产生什么偏差？
- 一个 speedup 声明至少需要绑定哪些 Artifact 和环境事实？
