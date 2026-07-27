# 发布操作与证据

[English version](release-operations.md)

KernelBlaster 的发布以可验证证据为中心，不作 production deployment 声明。自动生成执行仍默认关闭，只有运维者显式配置既有 feature flag 和服务密钥后才会开启。

## 规划并验证 E2E

先使用公开 profile 生成可审计计划，再进行任何付费 Provider 请求或 GPU 工作负载：

```bash
python scripts/e2e_release.py --profile configs/release.example.toml
python scripts/fault_inject.py --output release-evidence/fault-plan.json
```

`--execute` 会执行可执行的 preflight、64-token Provider smoke 与受限 RMSNorm Agent 阶段，并将 preflight 的 capability digest 传给 Agent。它假定 Compose 服务和外部密钥已经由运维者启动。硬件、模型、token 预算、路径和超时均来自 profile，不会写死到源码；trusted smoke 仍使用固定的受信任 smoke model。

## 证据

从公开 JSON 摘要生成脱敏证据，然后校验索引：

```bash
python scripts/release_evidence.py local-<gpu-slug> \
  --profile configs/release.example.toml \
  --evidence-json out/release-summary.json
python scripts/verify_release.py release-evidence
```

写入器会记录当前 commit、规范化 profile hash 与 schema digest，然后移除凭据、token、私有值、主机/实例信息和状态路径，并将每个证据文件加入 `SHA256SUMS.json`。`--include` 只接受公开 JSON 摘要并再次脱敏。大型原始 NSYS/NCU 报告应作为 Release artifact 保存；仓库只提交摘要和 hash。

## 备份与回滚

在 migration 或发布前备份 SQLite control state：

```bash
python scripts/backup_state.py /safe/backups --state-dir "$KERNELBLASTER_STATE_DIR"
python scripts/restore_state_backup.py /safe/backups/kernelblaster-state-<timestamp> \
  --state-dir "$KERNELBLASTER_STATE_DIR" --yes
```

备份使用 SQLite online backup API，并记录 snapshot SHA-256；CAS 不可变且不会复制。恢复前必须停止 Control。回滚只会在显式确认后恢复 SQLite，不进行破坏性的 schema down-migration，也不会删除历史 CAS payload。

打 tag 前请查看 [发布检查清单](release-checklist.zh-CN.md)。
