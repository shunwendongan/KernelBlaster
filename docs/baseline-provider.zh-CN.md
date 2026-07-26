# 独立 Baseline Provider 与排名

English version: [baseline-provider.md](baseline-provider.md)

可信 Baseline Worker 使用独立镜像、UID、Bearer token audience 和内部网络；它不与
generated Job 共用镜像，也不会接收生成候选源码。公开 Provider 列固定为
`upstream_cuda`、`pytorch_eager`、`pytorch_compile`、`triton`、`cublas`、
`cudnn` 和 `cutlass`。

每个 Provider 必须先通过相同的规范化 TaskSpec 和 case bundle，测量结果才能标记
`comparable`。工具缺失、算子不适用、执行失败或 correctness 失败只生成
`unavailable/reason_code`，不会阻断其他列。Worker 已包含真实 PyTorch eager runtime；
其他列在固定镜像包含相应实现前会明确报告 unavailable，绝不会在 Provider 之间回退。

Forward/backward 的正式比较始终以 `upstream_cuda` 为主基线。PyTorch、Triton、
cuBLAS、cuDNN 和 CUTLASS 只作为参考列。generated candidate 不得链接或调度这些
供应商库。缓存键绑定 TaskSpec、case bundle、evaluation bundle、Provider、Baseline
镜像、硬件指纹、target arch、协议和主目标。

## 计时与资格

每次 run 只有一个主目标（`latency` 或 `throughput`）。Worker 同时记录 CUDA Events
device operator time 和 host end-to-end time；只有 device time 能参与排名。输入创建、
编译和 workspace 分配位于计时外。TaskSpec workload 同时包含 hot-cache 和
buffer-rotation/L2 扰动；rotation 会分配足以超过两倍 L2 容量的独立输入 bank，同时
限制已大于 L2 的 Tensor 不被无界复制。

正式资格固定要求五组配对 confirmation session：

- 带权几何平均 speedup 至少 1.05；
- 配对 bootstrap 95% 下界大于 1.0；
- 任一核心 workload 均不得退化。

Winner 按硬件指纹、前/反向、数值等级、确定性等级和后端隔离。跨 GPU 只能汇总
correctness/portability，不能混合绝对 latency。Triton 可作为参考，但只有 CUDA
结果可成为 `primary_cuda_winner`。

## 搜索收敛

CUDA/Triton 同时活跃时采用确定性 70%/30% 调度。两者各自在连续 24 个 rankable
候选没有 1% discovery 改善，或连续 50 个候选因 compile/correctness/Events 不可排名
时收敛；另一后端继续独立运行。Provider quota 或传输中断属于可恢复 `blocked`，不得
伪装成收敛。CUDA Events 不可用才终止 GPU 搜索，因为 NSYS/NCU 只用于诊断。

## 构建与本地 smoke

构建独立镜像、读取不可变 ID，并把 ID 提供给 Compose：

```bash
docker compose --profile build build baseline-worker
export KERNELBLASTER_BASELINE_IMAGE_DIGEST="$(docker image inspect \
  --format '{{.Id}}' local/kernelblaster-baseline-worker:cuda12.8-dev)"
export KERNELBLASTER_BASELINE_TOKEN=<distinct-secret>
```

RTX 3080 smoke 会运行完整 TaskSpec correctness、hot/rotating workload 的 device/host
计时、缓存 provenance 及固定的严格 gate fixture：

```bash
python scripts/run_baseline_worker_smoke.py \
  --image-digest "$KERNELBLASTER_BASELINE_IMAGE_DIGEST" \
  --output /tmp/baseline-worker-smoke.json
```
