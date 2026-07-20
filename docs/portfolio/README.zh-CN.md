# Portfolio 验证与开发进度

**简体中文** | [English](README.md)

<!-- PORTFOLIO_PROGRESS:START -->
**更新日期：2026-07-20**

- Day 1–2：Provider、Recorder、Suite、dry-run 与 CPU 测试已完成。
- Day 3–7：WSL2/RTX 3080、容器、编译、正确性和 CUDA Events 基准设施已完成；LLM Agent Pilot/Core 10 因 API 401 未执行。
- Day 8–10：RMSNorm V0–V3c 已完成，独立深度结果 49.348×；统一 Core 10 复测 52.772×。
- 后续 Core 10：九个新增候选全部正确，严格通过 3/9；完整 Core 10 严格组合 4.356×。
- PyTorch 对照：完整十题严格组合为 0.992×，总体基本持平；不稳定任务在报告中保持显式标记。

[中文完整报告](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [英文摘要](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [逐题 JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [对比图](../../artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [原始文件哈希](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [候选清单](../../portfolio/case_studies/core10/candidates.json) · [Draft PR #5](https://github.com/shunwendongan/KernelBlaster/pull/5)
<!-- PORTFOLIO_PROGRESS:END -->

## 文档导航

- [架构、Provider、Recorder、Benchmark 与文档同步边界](architecture.md)
- [验证门槛、已完成工作、证据与剩余阻塞项](validation.md)
- [RMSNorm V0–V3c 实测案例](rmsnorm-case-study.md)
- [RTX 3080 已发布 artifact 包](../../artifacts/portfolio-v1.0/README.md)
- [Core 10 与 PyTorch 中文完整分析](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [英文独立摘要](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)

## 更新流程

`portfolio/status.json` 记录叙述状态并指向权威证据；性能数字直接从已提交结果 JSON 推导，不在状态清单中重复手填。任何 Benchmark、候选、分析或发布变更后执行：

```bash
python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

GitHub 文档同步 workflow 会重复检查；相关源码或结果变更没有同步 README/docs/status 时，PR 将失败。
