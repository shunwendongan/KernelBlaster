# KernelBlaster

[English](README.md) | **简体中文**

KernelBlaster 是一个**正确性优先、性能分析驱动、带长期优化记忆的 CUDA Kernel 优化研究框架**。它把 KernelBench-CUDA 任务、LLM 候选生成、rollout/经验回放、CUDA Events、NSYS/NCU 诊断、可复现实验与证据管理组织在同一仓库中。

> 项目定位是可信研究原型，不是通用生产算子库。任何性能结论都必须绑定 GPU、shape、dtype、layout、正确性协议和测量方法。

## 从这里开始

| 你的目标 | 最短入口 |
| --- | --- |
| 十分钟理解项目 | [文档导航](docs/README.zh-CN.md) → [快速开始](docs/quickstart.zh-CN.md) |
| 理清源码结构 | [源码架构](docs/architecture.zh-CN.md) → [核心源码阅读指南](docs/source-guide.zh-CN.md) |
| 学习写高性能算子 | [高性能算子开发指南](docs/operator-development.zh-CN.md) → [RMSNorm 案例](docs/portfolio/rmsnorm-case-study.zh-CN.md) |
| 复现固定候选 | `scripts/benchmark_candidates.py`、`scripts/benchmark_cuda.py` |
| 了解验证结果 | [Portfolio 状态](docs/portfolio/README.zh-CN.md) → `artifacts/portfolio-v*/` |
| 了解分支演进 | [开发历史与分支状态](docs/development-history.zh-CN.md) |

## Portfolio 证据快照

<!-- PORTFOLIO_STATUS:START -->
已提交的 Portfolio 证据记录了 **NVIDIA GeForce RTX 3080（sm_86）** 上的 Day 1–10 基础设施、RMSNorm 深度案例、Core 10 手工候选和同卡 PyTorch 对比。记录环境为 WSL2、CUDA 12.8.61、驱动 591.86。

| 验证项目 | 当前状态 |
| --- | --- |
| CPU 测试 | **177 项通过**（`portfolio/status.json` 最近记录） |
| CUDA 编译与官方正确性 | **历史 10/10；schema v2 完整验证 10/10 通过** |
| CUDA Events 与同卡 PyTorch | **schema v2 完整验证：4 项提升、1 项无提升、5 项无法定论；9/10 题有稳定 PyTorch 方法** |
| 外部 LLM 冒烟测试 | **失败：当前 HTTP 401（1 次请求、0 次重试、0 tokens；2026-07-22）** |
| Nsight Compute 硬件计数器 | **阻塞：ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)** |
| 跨 GPU 复测 | **阻塞：requires authorized A100/L40S rental** |

| 历史 v1 实测范围 | 相对仓库原版（诊断 / 旧严格口径） | 相对 PyTorch 最快方法（诊断 / 旧严格口径） |
| --- | ---: | ---: |
| 本轮新增九题 | 5.020× / 3.302× | 1.415× / 0.931× |
| 完整 Core 10（含 RMSNorm） | 6.351× / 4.356× | 1.447× / 0.992× |

上述严格值作为不可变的历史 v1 证据保留。独立的 schema v2 完整手工确认验证了 10/10 正确性，正式确认 004/007/036/040，将 088 标为无提升，并把 019/023/026/047/095 保持为无法定论。当前口径下，严格 Core 10 相对上游的几何平均为 4.381×；仅在 9/10 个存在正确且稳定 PyTorch 方法的可比任务上，严格结果相对最快稳定方法的几何平均为 1.053×。它仍不是 Agent 搜索结果。新口径还检查 p99/max 误差回归、NaN/Inf 和五次确定性。当前 Agent Pilot 与 Core 10 Agent 搜索均未运行。

[Schema v2 完整 Core 10 验证](artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.zh-CN.md) · [Schema v2 完整结果 JSON](artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json) · [Schema v2 定向验证](artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.zh-CN.md) · [Schema v2 结果 JSON](artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [中文完整报告](artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [英文摘要](artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [逐题 JSON](artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [对比图](artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [原始文件哈希](artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [候选清单](portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_STATUS:END -->

### 复现 RTX 3080 正式对比

以下命令应在固定的 NGC 25.01 容器和 `sm_86` GPU 中执行。原始输出保存在被忽略的 `out/portfolio/`，审核后的结果单独提交。

```bash
python scripts/benchmark_candidates.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>

python -m pytest -q
python scripts/sync_portfolio_docs.py --check
```

这里的 Agent 优化循环执行 rollout 搜索和经验库更新，不会微调或训练底层大语言模型权重。上表结果来自手工候选，不能写成 Agent 搜索结果。

### Portfolio v2.1 证据

v2.1 加固了 Issue #10 的五个 CUDA 候选，但没有扩大生产可用性声明。稳定能力契约仅接受已审核的 `sm_86`、FP16、连续 row-major、legacy default stream、单 stream、forward-only、非 graph capture 和 manifest 白名单场景；`production_ready` 仍为 `false`。

- [证据索引与 SHA-256 清单](artifacts/portfolio-v2.1/SHA256SUMS.json)
- [五任务正确性与资源生命周期汇总](artifacts/portfolio-v2.1/issue-10/rtx3080/correctness-summary.json)
- [Issue #7 API/Pilot 状态](artifacts/portfolio-v2.1/issue-7/rtx3080/trusted-pilot-summary.json)：HTTP 401，Pilot 未运行
- [Issue #8 Profiler 状态](artifacts/portfolio-v2.1/issue-8/rtx3080/ncu-preflight-summary.json)：Windows 原生 NCU/NSYS 证据已发布；WSL counters 和跨 GPU 复测仍未完成

## 项目解决什么问题

CUDA 优化不是简单的语法改写。一个实现是否更快取决于数据布局、线程映射、访存合并、归约方式、指令吞吐、occupancy、GPU 架构和测量噪声。传统固定启发式难以覆盖所有组合，朴素 LLM Agent 又容易遗忘早期探索并重复失败。

KernelBlaster 通过以下闭环缩小搜索空间：

1. 读取 `init.cu`、`driver.cpp` 和任务契约；
2. 建立正确性通过的初始性能基线；
3. 从 Profiler/测量结果提取当前性能状态；
4. 从持久化优化知识库检索可用策略；
5. 让 LLM 生成有明确假设的新候选；
6. 先编译和验证正确性，再测量 CUDA Events；
7. 将成功与失败轨迹写入 Replay Buffer 和优化数据库；
8. 用标准 `RunOutcome` 保存终态和最佳候选。

LLM 负责提出候选，不负责宣布性能结论。CUDA Events 是排名来源；NSYS/NCU 是诊断来源；任何正确性失败都必须阻断排名。

## `master` 已实现的功能

| 能力 | 关键模块 | 状态说明 |
| --- | --- | --- |
| MAIC-RL 风格优化循环 | `agents/opt_ncu_rl.py`、`rl_agents.py`、`database.py` | 保留经典研究链、Replay Buffer 和跨任务知识 |
| OpenAI-compatible LLM | `llm/` | 支持并发、重试、预算、usage 与脱敏记录 |
| 正确性优先测量 | `benchmarking.py`、`profiling.py`、`measurements.py` | 显式区分单位、测量来源和不可用原因 |
| 标准终态 | `outcomes.py`、`docs/measurement-status-contract.zh-CN.md` | 区分 improved、no improvement、blocked、failed 和 timeout |
| 安全 Control 面 | `servers/control.py`、`storage/` | SQLite 元数据与 SHA-256 CAS 分离 |
| GPU Job 与沙箱 | `gpu_jobs/` | digest-only manifest、硬件能力、一次性无网络容器 |
| Profiler Worker | `profiler_jobs/` | 固定 NSYS/NCU plan，诊断不参与排名 |
| Runtime preflight | `preflight/` | 有序检查 Provider、存储、GPU、沙箱、Events 与诊断能力 |
| Portfolio | `portfolio/`、`artifacts/`、`scripts/benchmark_*.py` | 固定候选、严格协议、报告与哈希证据 |

## 重要的主线边界

审计时默认分支 `master` 位于 PR‑07a。PR‑07b 到 PR‑09 是逐层合入后续开发分支的 stacked PR，尚未整体进入 `master`。这些开发分支包含 Agent candidate funnel、通用 correctness harness、独立 baseline provider、CUDA/Triton candidate package、AutoDL 可移植性和 release 编排。

因此当前主线有两个需要明确的事实：

- 安全 `sandbox` backend 会 fail closed，不允许退回本地 `driver.cpp` 执行；
- 将 Agent 生成源码转为结构化候选并提交沙箱的 CandidateEvaluator 尚在 stacked 开发线。

已经移除会误导使用者的旧版本地一键脚本；`master` 上应使用显式的固定候选工具。详细分支关系见 [开发历史与分支状态](docs/development-history.zh-CN.md)。

## 不调用 GPU 或 API 的快速检查

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/run_portfolio.py --suite rmsnorm --dry-run \
  --output-dir out/portfolio/rmsnorm/dry-run
```

这适合在 macOS 上审阅能力清单、Suite、预算和 Artifact 结构。CUDA 候选开发、安全部署与结果解释统一放在 [快速开始](docs/quickstart.zh-CN.md)，源码职责只在 [源码架构](docs/architecture.zh-CN.md) 中维护。

## 论文与引用

**arXiv：** [arXiv:2602.14293](https://arxiv.org/abs/2602.14293) · **仓库内 PDF：** [KernelBlaster.pdf](docs/figures/KernelBlaster.pdf)

上游作者报告在 KernelBench Level 1/2/3 上相对 PyTorch 的几何平均加速分别为 1.43×、2.50× 和 1.50×。这些论文全量结果只是背景，与本 Fork 的 RTX 3080 Core 10 手工候选结果严格分开。

```bibtex
@article{dong2026kernelblaster,
  title={KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning},
  author={Dong, Kris Shengjun and Modi, Sahil and Nikiforov, Dima and Damani, Sana and Lin, Edward and Hari, Siva Kumar Sastry and Kozyrakis, Christos},
  journal={arXiv preprint arXiv:2602.14293},
  year={2026}
}
```

## 贡献

提交修改前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。新增 benchmark、候选或 Artifact 时必须同步更新 README/docs 或 `portfolio/status.json`，并保持中英文文档成对。

上游贡献者包括 Kris Shengjun Dong、Sahil Modi、Dima Nikiforov、Sana Damani、Edward Lin、Siva Kumar Sastry Hari 和 Christos Kozyrakis。项目大部分上游工作由 Kris Shengjun Dong 在 2025 年 NVIDIA 暑期实习期间完成。
