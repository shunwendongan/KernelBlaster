# 通用多算子 Harness

English version: [generic-harness.md](generic-harness.md)

KernelBlaster 通过版本化合约评测算子，而不是建立单一 RMSNorm ABI。
`harness-task/v1` 声明前/反向、Tensor/标量 ABI、有界动态 shape、全部可微输入、
数值与确定性等级、workspace、一个调用方提供的 stream、可选 CUDA Graph 行为及
带权性能 workload。首个 catalog 覆盖 Core 10 全部 forward/backward；RMSNorm
只是 normalization 类的一个样例。

可信 runtime 生成 Dev、Feedback、Audit 输入，快照所有不可变输入，并独占生成
`correctness-result/v2`。它检查输出 poison、低层 Adapter 报告的 guard canary、
输入 mutation、shape/dtype、NaN/Inf、CUDA error、全部目标梯度及重复执行稳定性；
候选不能自行宣称通过。Feedback/Audit case 会完整披露，因此协议明确标为
`adaptive_disclosed`，不宣称 held-out。

常规算子使用声明式 TaskSpec；复杂 reference 和资源生命周期使用镜像内可信
Adapter。既有 KernelBench `driver.cpp` 由 `LegacyDriverAdapter` 包装而不批量重写；
Core 10 中“比较失败仍返回进程状态 0”的确认 bug 已作最小修复。Backward reference
使用 FP16 输入和 FP32 计算的 PyTorch autograd。PyTorch 只是 oracle，不是外部项目
必须满足的 Custom Op gate。十项确定、可审计的 CUDA backward baseline 位于
`portfolio/harness/core10/backward/`。

## 私有 case 与 CAS 注册

真实 case 文件应位于仓库外，例如 `$HOME/secrets/kernelblaster/cases`。仓库只保存
确定性的公开 fixture 与工具。Control 运行后，可选择用公开 fixture 初始化外部目录，
再把每个规范化 TaskSpec、case bundle 和可信源码 bundle 上传 CAS：

```bash
uv run python scripts/register_core10_harness.py \
  --case-root "$HOME/secrets/kernelblaster/cases" \
  --output "$HOME/secrets/kernelblaster/core10-harness-catalog.json" \
  --public-fixtures
```

真实评测部署应省略 `--public-fixtures`。脚本会在上传前验证已有 case 文件的 schema、
shape、TaskSpec digest 和 SHA-256 绑定。

## 签名 Adapter 插件

外部项目通常只需新增 TaskSpec 和 Ed25519 签名 Adapter bundle，无需改 KernelBlaster
核心源码。私钥必须在仓库外生成，然后构建确定性 bundle 并验证：

```bash
uv run python scripts/adapter_plugin.py keygen \
  --private-key "$HOME/secrets/kernelblaster/adapter.key" \
  --public-key "$HOME/secrets/kernelblaster/adapter.pub"
uv run python scripts/adapter_plugin.py build \
  --descriptor /path/to/plugin-descriptor.json \
  --payload-dir /path/to/plugin-payload \
  --private-key "$HOME/secrets/kernelblaster/adapter.key" \
  --output /path/to/adapter-plugin.tar
uv run python scripts/adapter_plugin.py verify \
  --bundle /path/to/adapter-plugin.tar --key-id project-owner \
  --public-key "$HOME/secrets/kernelblaster/adapter.pub"
```

镜像构建还会同时校验精确 bundle digest、发布者公钥、plugin 身份和 Adapter ID
allowlist。只有公钥和已签名 bundle 会进入临时 Docker 构建上下文，私钥不会进入：

```bash
uv run python scripts/build_adapter_job_image.py \
  --bundle /path/to/adapter-plugin.tar \
  --trusted-keys "$HOME/secrets/kernelblaster/trusted-adapter-keys.json" \
  --allowlist "$HOME/secrets/kernelblaster/adapter-plugin-allowlist.json" \
  --tag local/kernelblaster-gpu-job:adapter-v1
```

Supervisor 必须固定生成镜像的 `sha256:` ID，不能以 tag 作为信任锚。

## RTX 3080 smoke

固定 smoke 会验证 20 个 TaskSpec、披露的动态 case、在 Harness 所有非默认 stream
运行的十项朴素 CUDA backward baseline，以及 mutation/NaN/poison/canary 恶意 fixture：

```bash
python scripts/run_core10_harness_smoke.py --device cuda --backward-cuda \
  --output /tmp/core10-harness-smoke.json
```

CUDA Events 排名、独立 baseline provider、AOT 候选隔离和 profiler replay 会在后续
堆叠改动中建立于本 correctness 合约之上。NSYS/NCU 始终只用于诊断，不决定正确性。
