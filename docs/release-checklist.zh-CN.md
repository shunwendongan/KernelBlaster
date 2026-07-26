# 发布检查清单

[English version](release-checklist.md)

- [ ] `uv lock --check`、完整 CPU 测试、静态检查和 Compose config 全部通过。
- [ ] 本地硬件证据记录实际 capability、环境、镜像 digest 和 measurement protocol。
- [ ] 至少一台全新 AutoDL 完成 bootstrap、preflight、smoke、受限 run、export 和本地 import。
- [ ] 故障计划记录重启、lease、存储、profiler 和 secret boundary 结果；危险故障必须显式启用。
- [ ] `scripts/verify_release.py release-evidence` 返回有效 hash。
- [ ] 已保留上一版 SQLite state 备份以及 Compose/镜像引用。
- [ ] README 与 runbook 不宣称 production-ready、通用 NCU 权限或跨 GPU 性能排名。
- [ ] 自动不可信执行继续默认关闭。
- [ ] tag 和 GitHub Release 仍需单独最终批准。
