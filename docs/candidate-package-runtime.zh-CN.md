# CUDA/Triton AOT 候选运行时

英文版：[candidate-package-runtime.md](candidate-package-runtime.md)

`generated_v2` 只通过不可变、无 host 代码的链路评测设备代码：

```text
candidate-package/v2 -> 一次性 compile -> candidate-capsule/v1
  -> 一次性 correctness / Events -> correctness-gated profiler capsule
  -> NSYS / NCU / compute-sanitizer 下的固定 Harness replay
```

## 契约与边界

每个包只包含 canonical `candidate-manifest.json`、`launch-plan.json`，以及
`candidate.cu` 或 `candidate.py` 二者之一。source、Launch Plan、TaskSpec、
cubin 与 capsule 全部通过 SHA-256 绑定。受控计划支持有界 shape dispatch、
顺序多 kernel、`ceil_div/min/max`、动态 shared memory、单 Harness stream、
Harness workspace 与 TaskSpec CUDA Graph 策略。

CUDA 包只能定义设备 kernel。运行前会拒绝 `main`、host launcher、构造器、
host allocation/stream API、inline assembly、非白名单头文件、供应商库、
文件/网络/进程调用和未声明 kernel。Triton source 经过 AST 白名单验证，
只在 compile 容器运行，不会进入 correctness、Events、Profiler 或 sanitizer。
Triton AOT v1 支持单个整 warp kernel，首轮覆盖 007 Small-K MatMul、026 GELU、
036 RMSNorm 与 047 Sum；其余 Core 10 Triton 任务返回 `backend_unsupported`。

CUDA fixture 覆盖 Core 10 全部 forward/backward。只有 CUDA 能成为
`primary_cuda_winner`，Triton 只能作为辅助候选或参考。

## 可信 replay 与诊断

Compile Job 只导出 `module.cubin` 和有界日志。Supervisor 会重新验证源包、
构建 capsule，并在每个后续阶段重新只读挂载私有 TaskSpec/cases。输入、输出、
launch 参数、非默认 stream、correctness verdict 和 CUDA Events 计时都由
可信 Harness 控制。

只有结构化 `correctness-result/v2` 成功后才生成 `profiler_replay`。Control
会拒绝其他 generated-v2 artifact 进入 Profiler；Profiler 只执行固定可信
wrapper，不执行候选 host 代码。NSYS/NCU 始终只是诊断，Events 始终是排名
依据；CUDA winner 还必须通过 memcheck、initcheck、racecheck、synccheck。

## RTX 3080 / WSL smoke

```bash
docker build --target gpu-job -t local/kernelblaster-gpu-job:pr07e \
  -f docker/Dockerfile .
export KERNELBLASTER_GPU_JOB_IMAGE="$(docker image inspect \
  --format '{{.Id}}' local/kernelblaster-gpu-job:pr07e)"

KERNELBLASTER_RUN_GPU_TESTS=1 \
  python -m pytest -q -m gpu tests/candidate_packages/test_candidate_package_gpu.py

KERNELBLASTER_RUN_GPU_SANDBOX_TESTS=1 \
  uv run --with 'docker>=7,<8' --extra dev --extra server \
  python -m pytest -q -m gpu_sandbox tests/gpu_jobs/test_sandbox_integration.py

python scripts/run_profiler_capsule_smoke.py --target-arch sm_86
```

最后一条命令应在 `profiler-worker` 中，以 Compose 相同的非 root UID 10002
和仅 `SYS_ADMIN` capability 运行。NSYS、NCU、结构化指标或任一 sanitizer
不可用时都会失败。旧 artifact 仍可读取；旧 `.cu +
launch_gpu_implementation()` executable 不会自动转换成 v2。
