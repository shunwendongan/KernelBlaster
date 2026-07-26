# AutoDL 独立实例

KernelBlaster 支持独立运行的 AutoDL 实例。每个实例独享 Control、SQLite、CAS、GPU
Supervisor、Profiler Worker 和 Baseline Worker；实例之间不共享数据库、队列、CAS 或服务发现。

## 启动

将 `deploy/autodl/profile.env.example` 复制到仓库外，设置状态目录、GPU 设备号和可选的不可变镜像引用：

```bash
bash scripts/autodl/bootstrap.sh --profile /secure/path/autodl.env --check-only
bash scripts/autodl/bootstrap.sh --profile /secure/path/autodl.env
```

脚本只探测平台，绝不会安装、升级或替换 NVIDIA display/kernel driver；驱动由 AutoDL 镜像或平台管理。
GPU 型号、CUDA、驱动和编译架构均由 capability probe 自动确定，不需要修改源码中的型号枚举。

使用 `python scripts/manage_instance.py show --state-dir <state>` 查看稳定 instance ID。若复制了数据盘，必须显式执行
`python scripts/manage_instance.py rotate --yes` 生成新身份。

## 独立执行与迁移

完成一个终态 run 后执行：

```bash
python scripts/export_run.py <run-id> /data/run.tar --state-dir <state>
python scripts/import_run.py /data/run.tar --state-dir <local-state>
python scripts/aggregate_runs.py /data/aggregate --state-dir <local-state>
```

bundle 只包含该 run 的公开 artifact 闭包，不包含凭据、私有 evaluation profile、driver 或 seed。导入器将归档视为不可信输入，拒绝链接、路径穿越、未知成员、超限文件和 hash 不匹配。相同内容可幂等重复导入；相同 run ID 但不同内容会被拒绝。

## 可选显式 remote target

`deploy/autodl/targets.toml.example` 用于用户显式选择 SSH target，仅保存 SSH alias、远端工作目录和 profile 引用。密钥仍由 SSH agent/config 管理。系统不会自动调度、切换机器或迁移正在运行的 run。
