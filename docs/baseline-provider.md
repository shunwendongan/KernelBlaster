# Independent baseline providers and ranking

Chinese version: [baseline-provider.zh-CN.md](baseline-provider.zh-CN.md)

The trusted Baseline Worker is a separate image, UID, bearer-token audience,
and internal network. It does not share the generated Job image and never
receives generated candidate code. Its public provider columns are
`upstream_cuda`, `pytorch_eager`, `pytorch_compile`, `triton`, `cublas`,
`cudnn`, and `cutlass`.

Every provider must pass the same canonical TaskSpec and case bundle before its
measurements become `comparable`. A missing tool, unsupported operator, failed
execution, or failed correctness check produces `unavailable/reason_code` and
does not stop the other columns. The worker ships a real PyTorch eager runtime;
other columns remain explicit and unavailable until the pinned image contains
their fixed implementation. There is no provider-to-provider fallback.

Forward and backward formal comparisons always use `upstream_cuda`. PyTorch,
Triton, cuBLAS, cuDNN, and CUTLASS are reference columns only. Generated
candidates may not link or dispatch to those vendor libraries. A cache key binds
TaskSpec, case bundle, evaluation bundle, provider, Baseline image, hardware
fingerprint, target architecture, protocol, and objective.

## Timing and qualification

One run has one primary objective (`latency` or `throughput`). The worker records
CUDA Events device operator time and host end-to-end time; only device time can
rank. Input creation, compilation, and workspace allocation are outside timing.
TaskSpec workloads contain both hot-cache and rotating-buffer/L2-disruption
modes. Rotation allocates enough distinct input banks to exceed twice the
reported L2 capacity, bounded to avoid multiplying already-L2-sized tensors.

Formal qualification requires exactly five paired confirmation sessions:

- weighted geometric-mean speedup of at least 1.05;
- paired bootstrap 95% lower bound greater than 1.0;
- no regression on any core workload.

Winners are isolated by hardware fingerprint, direction, numeric class,
determinism class, and backend. Cross-GPU reports may summarize correctness and
portability but never merge absolute latency. Triton rows can be useful
references, but only a CUDA row can become `primary_cuda_winner`.

## Search convergence

CUDA and Triton use a deterministic 70/30 schedule while both are active. They
converge independently after 24 rankable candidates without a 1% discovery
improvement, or 50 consecutive unrankable compile/correctness/Events results.
The surviving backend continues alone. Provider quota or transport failures are
recoverable `blocked` events and never masquerade as convergence. An unavailable
CUDA Events path stops GPU search because NSYS/NCU are diagnostic-only.

## Build and local smoke

Build the independent image, inspect its immutable ID, and then provide the ID
to Compose:

```bash
docker compose --profile build build baseline-worker
export KERNELBLASTER_BASELINE_IMAGE_DIGEST="$(docker image inspect \
  --format '{{.Id}}' local/kernelblaster-baseline-worker:cuda12.8-dev)"
export KERNELBLASTER_BASELINE_TOKEN=<distinct-secret>
```

The RTX 3080 smoke exercises full TaskSpec correctness, hot/rotating workload
device and host timing, cache provenance, and a fixed strict-gate fixture:

```bash
python scripts/run_baseline_worker_smoke.py \
  --image-digest "$KERNELBLASTER_BASELINE_IMAGE_DIGEST" \
  --output /tmp/baseline-worker-smoke.json
```
