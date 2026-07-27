# KernelBlaster 文档导航

[English](README.md) | **简体中文**

这组文档按“先找到入口，再理解系统，最后优化算子”的顺序组织。除 `docs/portfolio/` 中明确标记的历史 GPU 证据外，功能描述以默认分支 `master` 的源码为准。

## 我应该先读哪一篇？

| 目标 | 推荐文档 | 读完后能做什么 |
| --- | --- | --- |
| 十分钟看懂项目 | [快速开始](quickstart.zh-CN.md) | 区分无 GPU、手工优化和 Agent/基础设施入口 |
| 理清源码模块 | [源码架构](architecture.zh-CN.md) | 知道入口、状态、执行面、存储和证据层分别在哪里 |
| 学习写高性能算子 | [高性能算子开发指南](operator-development.zh-CN.md) | 从契约、基线、瓶颈假设到正确性与性能门控完成一次迭代 |
| 阅读核心调用链 | [核心源码阅读指南](source-guide.zh-CN.md) | 按调用顺序阅读 `run_RL.py`、Graph、Agent 和 Profiler |
| 理解历史演进 | [开发历史与分支状态](development-history.zh-CN.md) | 区分 `master` 已有能力和 stacked PR 开发能力 |
| 复现实验结果 | [Portfolio 导航](portfolio/README.zh-CN.md) | 找到 RTX 3080 结果、验证口径和 RMSNorm 案例 |
| 理解测量状态 | [测量与状态契约](measurement-status-contract.zh-CN.md) | 正确解释 correctness、timing、diagnostic 和终态 |

## 按内容分区

### 使用与学习

- [快速开始](quickstart.zh-CN.md)：最短入口、依赖边界和常见误区。
- [高性能算子开发指南](operator-development.zh-CN.md)：可迁移到其他 CUDA 算子的优化方法。
- [核心源码阅读指南](source-guide.zh-CN.md)：框架内部调用链和关键不变量。

### 架构与契约

- [源码架构](architecture.zh-CN.md)：源码目录、执行路径、数据和信任边界。
- [测量与状态契约](measurement-status-contract.zh-CN.md)：机器可读结果字段的语义。
- [Portfolio 架构](portfolio/architecture.zh-CN.md)：实验套件、Runner 和 Artifact 契约。

### 结果与证据

- [Portfolio 状态](portfolio/README.zh-CN.md)：当前验证进度。
- [验证协议](portfolio/validation.zh-CN.md)：结果如何产生、什么可以宣称。
- [RMSNorm 优化案例](portfolio/rmsnorm-case-study.zh-CN.md)：从访存映射到实测候选的完整案例。
- `artifacts/portfolio-v*/`：不可变的结果、报告、图表和 SHA-256 清单。

## 文档事实口径

阅读本仓库时请始终区分三类事实：

1. **源码已实现**：默认分支中存在代码、契约和测试定义；不等于已在当前机器运行。
2. **历史已验证**：提交的 Artifact 给出硬件、协议、结果和哈希；结论只在声明边界内成立。
3. **开发中或受阻**：代码位于非 `master` 分支，或者仍依赖 Provider 凭据、GPU、Profiler 权限与跨卡资源。

文档变更应同步维护英文文件 [README.md](README.md)。Portfolio 的自动生成状态区块由 `scripts/sync_portfolio_docs.py` 管理，不应手工修改区块内部文本。
