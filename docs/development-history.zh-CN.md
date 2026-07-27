# KernelBlaster 开发历史与分支状态

[English](development-history.md) | **简体中文**

本文根据 2026-07-27 的 GitHub 提交与 PR 记录整理，用于解释源码中为什么同时出现旧 Agent 路径、安全基础设施和未完全接通的接口。它是静态审计快照，不替代 GitHub 的实时分支状态。

## 默认分支基线

审计时 `master` 指向 `60eb264f`（PR #21，runtime capability preflight）。因此 README 对“当前可用功能”的描述必须以该提交及其祖先为准。

## 已进入 `master` 的阶段

| 阶段 | 代表提交/PR | 主要结果 |
| --- | --- | --- |
| Provider 与可观测性 | PR #1–2 | OpenAI-compatible Provider、预算、双语文档和运行记录 |
| 可复现实验 | PR #3–6 | RMSNorm/Core 10、PyTorch 对比、Artifact 哈希和文档同步 |
| 测量与状态契约 | PR #14 | correctness、timing、diagnostic 和终态字段分离 |
| Compose 边界 | PR #15 | Control 与 GPU 服务隔离，token audience 分离 |
| 状态与 Artifact | PR #16 | SQLite 任务状态和 SHA-256 CAS |
| GPU Job | PR #17 | 硬件能力检测、digest-only manifest 和 Supervisor |
| 一次性沙箱 | PR #18 | 非 root、无网络、固定资源的生成 Job 容器 |
| Profiler Worker | PR #19–20 | 固定 NSYS/NCU plan、WSL preflight 与 CSV 解析 |
| Runtime preflight | PR #21 | Provider→存储→GPU→沙箱→Events→诊断的有序能力报告 |

## Stacked 开发线

PR #22–27 不是依次合入 `master`，而是每个 PR 合入下一个开发分支。审计时的关系是：

```text
master / PR-07a
  -> PR-07b Agent candidate funnel
  -> PR-07c generic correctness harness
  -> PR-07d independent baseline providers
  -> PR-07e CUDA/Triton candidate isolation
  -> PR-08 AutoDL portability
  -> PR-09 E2E release hardening
```

这些 PR 的 GitHub 状态显示为 merged，是因为它们合入了下一层分支；不能据此推断功能已经进入默认分支。

| PR | 开发能力 | 对主线文档的影响 |
| --- | --- | --- |
| #22 / PR-07b | 结构化 CandidateEvaluation、沙箱 CandidateEvaluator、发现/确认漏斗 | 解决 `master` 安全 backend 与 Agent 生成循环之间的缺口 |
| #23 / PR-07c | 通用 forward/backward correctness harness、TaskSpec 和 Adapter 契约 | 将私有 Driver/seed 与候选代码进一步分离 |
| #24 / PR-07d | 独立 Baseline Worker、多 Provider 参考列和多 workload 排名 | 丰富参考列，但 upstream CUDA 仍是正式基线 |
| #25 / PR-07e | `candidate-package/v2`、CUDA/Triton AOT 隔离与 replay capsule | 禁止执行候选 Python/host launcher，强化发布资格门控 |
| #26 / PR-08 | AutoDL 独立实例、run bundle 导入导出和跨机汇总 | 提供多实例可移植性工具 |
| #27 / PR-09 | release 编排、故障计划、证据与备份恢复 | 形成 0.3.0 release gate，但真实最终硬件验收仍延后 |

## 为什么主线文档要显式写这个边界

如果省略分支关系，会出现三种误导：

1. 把 `RuntimeBackendBundle` 的 fail-closed 行为误判为普通配置错误；
2. 把 stacked PR 的 harness、baseline、candidate package 和 AutoDL 命令写成 `master` 已有入口；
3. 把 PR 描述中的开发分支测试结果写成默认分支当前验证结果。

因此根 README 只总结主线能力，并将后续开发线作为状态说明；具体命令必须和所在分支的实际文件对应。

## 合并开发线后的文档维护清单

当 PR‑07b 到 PR‑09 真正合入默认分支后，应同时更新：

- [快速开始](quickstart.zh-CN.md) 的 Agent 主线状态与默认命令；
- [源码架构](architecture.zh-CN.md) 中 CandidateEvaluator、Harness、Baseline Worker、Candidate Package 和 Portability 模块；
- 根 README 的功能矩阵和目录树；
- `portfolio/status.json` 中可自动生成的验证状态；
- 英文配对文档和 `scripts/sync_portfolio_docs.py --check`。

在此之前，开发分支的验证记录可作为演进证据，但不能替代 `master` 的能力声明。
