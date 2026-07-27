# KernelBlaster 快速开始

[English](quickstart.md) | **简体中文**

KernelBlaster 是面向研究和性能工程的原型，不是一条在任意机器都能直接运行的命令。最快的正确用法取决于你的目标。

## 1. 先选择使用路径

| 目标 | 是否需要 NVIDIA GPU | 是否需要 LLM Key | 入口 |
| --- | ---: | ---: | --- |
| 阅读源码、检查配置、生成干跑记录 | 否 | 否 | 文档、`scripts/run_portfolio.py --dry-run` |
| 学习并手工优化固定 CUDA 算子 | 是 | 否 | `portfolio/case_studies/`、`scripts/benchmark_cuda.py` |
| 复现 Core 10 候选与 PyTorch 对比 | 是 | 否 | `scripts/benchmark_candidates.py`、`scripts/benchmark_pytorch.py` |
| 检查安全执行基础设施 | 是 | 是（preflight 包含一次有界鉴权） | Compose、`scripts/run_preflight.py` |
| 运行完整 Agent 候选搜索 | 是 | 是 | 先阅读下方“主线集成状态” |

macOS 可以阅读、编辑和执行部分纯 CPU/干跑工具，但不能替代仓库声明的 Linux/WSL2 + NVIDIA CUDA 环境。

## 2. 获取代码并认识任务

```bash
git clone https://github.com/shunwendongan/KernelBlaster.git
cd KernelBlaster
```

一个 KernelBench-CUDA 任务通常位于：

```text
data/kernelbench-cuda/level1/036_RMSNorm/
├── init.cu       # 待优化的上游 CUDA 实现
└── driver.cpp    # 输入、参考结果、正确性检查和 launch 契约
```

不要先改 Kernel。先从 `driver.cpp` 记录以下契约：

- 输入、输出、dtype、shape 和 layout；
- 被调用的 host 入口及参数顺序；
- stream、同步和资源生命周期要求；
- 误差阈值、NaN/Inf 和边界 case；
- 正式计时是否只覆盖 Kernel，还是还包含初始化与同步。

## 3. 无 GPU 的最短入口

查看项目状态与文档：

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/run_portfolio.py --suite rmsnorm --dry-run \
  --output-dir out/portfolio/rmsnorm/dry-run
```

第一条只读取候选能力清单，不编译或调用 CUDA。第二条解析 Suite、预算与输出契约，并生成标记为 dry-run 的运行记录，不发起 API 或 CUDA 调用。

推荐继续阅读：

1. [源码架构](architecture.zh-CN.md)；
2. [RMSNorm 优化案例](portfolio/rmsnorm-case-study.zh-CN.md)；
3. [高性能算子开发指南](operator-development.zh-CN.md)。

## 4. 手工优化一个算子的最短路径

这条路径不依赖 LLM。以 RMSNorm 为例：

1. 保持 `data/kernelbench-cuda/level1/036_RMSNorm/init.cu` 不变，作为上游基线。
2. 阅读 `portfolio/case_studies/rmsnorm/README.md` 中的 V0→V3c 假设链。
3. 复制候选到新的实验文件，不直接覆盖已审核结果。
4. 一次只改变一个主要假设，例如线程映射、向量化、归约或 block size。
5. 先通过官方 Driver 和 edge Driver，再比较 CUDA Events。

在支持的 NVIDIA 环境中，固定候选的命令形状如下：

```bash
python scripts/benchmark_cuda.py \
  --task-dir data/kernelbench-cuda/level1/036_RMSNorm \
  --task-id 036 \
  --kernel RMSNorm \
  --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu \
  --candidate-name my-rmsnorm \
  --extra-correctness-driver portfolio/case_studies/rmsnorm/edge_driver.cpp \
  --warmup 20 \
  --repetitions 100 \
  --sessions 3 \
  --output-dir out/experiments/rmsnorm/<run-id>
```

探索阶段可以使用三次 Session；要发布正式确认，应遵循 Portfolio 文档中的五次 Session 与同卡对比协议。不要把探索数字直接写进 README。

## 5. 复现已审核候选

先读取机器可读的支持边界：

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/benchmark_candidates.py \
  --describe-capabilities --task-id 036
```

在匹配的 `sm_86` 环境中复现单题：

```bash
python scripts/benchmark_candidates.py \
  --task-id 036 \
  --phase confirmation \
  --warmup 20 \
  --repetitions 100 \
  --sessions 5 \
  --output-dir out/portfolio/candidates/<run-id>
```

能力清单中的 `hardened` 表示在声明的 shape/dtype/layout/stream 边界内有额外正确性与资源生命周期契约；它不表示通用生产可用。`legacy_research_only` 候选只适合作为研究记录。

## 6. 安全基础设施入口

安全部署以根目录 `compose.yaml` 为唯一规范。它将系统拆为：

- `control`：CPU 控制面、SQLite/CAS 与 LLM 配置；
- `gpu-supervisor`：受控 GPU Job 和一次性沙箱；
- `profiler-worker`：固定 NSYS/NCU 计划；
- `dev`：可信的交互式 CUDA 开发环境。

密钥文件必须放在仓库外，并为四个 token audience 生成不同值：

```bash
mkdir -p "$HOME/secrets" "$HOME/runs/KernelBlaster/state"
cp -n .env.example "$HOME/secrets/KernelBlaster.control.env"
export KERNELBLASTER_CONTROL_ENV_FILE="$HOME/secrets/KernelBlaster.control.env"
export KERNELBLASTER_STATE_HOST_DIR="$HOME/runs/KernelBlaster/state"
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" config
```

启用生成代码 Job 前，还需要不可变 GPU Job 镜像 digest、Docker socket group 和 Supervisor-only 私有评估 profile。详细边界见 [Portfolio 架构](portfolio/architecture.zh-CN.md)。

## 7. 主线 Agent 集成状态

默认分支 `master` 已实现：

- OpenAI-compatible Provider、预算与运行记录；
- RL/知识库研究链；
- Control、SQLite/CAS、GPU Job、一次性沙箱与 Profiler Worker；
- 有序 runtime preflight 和 capability report；
- 固定候选的正确性优先 benchmark 与 Portfolio 证据。

但 `master` 上的安全 `sandbox` backend 明确拒绝退回本地 Driver；把 Agent 生成源码转换为结构化候选并提交给沙箱的 CandidateEvaluator 位于 PR‑07b 之后的 stacked 开发线，因此已经删除旧版本地一键脚本。`scripts/run_trusted_pilot.py` 可以完成 preflight 编排，但在候选漏斗合入主线前不应宣称端到端 Agent 可用；学习算子优化时应使用固定候选与 `benchmark_cuda.py` 路径。

分支关系与后续能力见 [开发历史与分支状态](development-history.zh-CN.md)。

## 8. 输出应该怎么看

| 输出 | 含义 | 是否足以宣称加速 |
| --- | --- | ---: |
| `final_rl_cuda_perf.cu` | Agent 选择的终态候选 | 否，还要核对测量和正确性 |
| `state.json` | Graph 状态快照 | 否 |
| `run_manifest.json` | 配置、源码和环境上下文 | 否 |
| `events.jsonl` | 执行与决策事件 | 否 |
| `summary.json` | 终态、预算和用量摘要 | 仅摘要 |
| benchmark `summary.json` | 正确性、样本和会话统计 | 在协议与硬件边界内可作为证据输入 |
| `artifacts/portfolio-v*/` | 审核后提交的证据包 | 只支持包内明确声明的结论 |

完整的算子优化检查表见 [高性能算子开发指南](operator-development.zh-CN.md)。
