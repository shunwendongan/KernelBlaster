# CUDA/Triton AOT Candidate Runtime

Chinese version: [candidate-package-runtime.zh-CN.md](candidate-package-runtime.zh-CN.md)

`generated_v2` evaluates device code through an immutable, host-code-free chain:

```text
candidate-package/v2 -> one-shot compile -> candidate-capsule/v1
  -> one-shot correctness / Events -> correctness-gated profiler capsule
  -> fixed Harness replay under NSYS / NCU / compute-sanitizer
```

## Contracts and boundaries

A package contains canonical `candidate-manifest.json`, `launch-plan.json`, and
exactly one of `candidate.cu` or `candidate.py`. SHA-256 binds the source, launch
plan, TaskSpec, compiled cubin, and every capsule. The controlled plan supports
bounded shape dispatch, sequential launches, `ceil_div/min/max`, dynamic shared
memory, one Harness stream, Harness-owned workspace, and the TaskSpec CUDA Graph
policy.

CUDA packages may define device kernels only. Validation rejects `main`, host
launchers, constructors, host allocation/stream APIs, inline assembly,
non-allowlisted headers, vendor libraries, file/network/process calls, and
undeclared kernels. Triton source is AST allowlisted, runs only in the compile
container, and is absent from correctness, Events, Profiler, and sanitizer
Jobs. Triton AOT v1 supports one fixed whole-warp kernel for 007 Small-K MatMul,
026 GELU, 036 RMSNorm, and 047 Sum. Other Core 10 Triton tasks return
`backend_unsupported`.

CUDA fixtures cover all Core 10 forward and backward TaskSpecs. Only CUDA may
become `primary_cuda_winner`; Triton remains an auxiliary candidate/reference.

## Trusted replay and diagnostics

The compile Job exports only `module.cubin` and bounded logs. The Supervisor
revalidates the package, builds the candidate capsule, and remounts private
TaskSpec/cases read-only per stage. The Harness owns inputs, outputs, launch
arguments, the non-default stream, correctness verdict, and CUDA Events timing.

Only successful structured `correctness-result/v2` creates a `profiler_replay`
artifact. Control rejects other generated-v2 artifacts at the Profiler boundary.
The Profiler executes a fixed trusted wrapper, never candidate host code. NSYS
and NCU remain diagnostic; Events remains the ranking source. A CUDA winner must
also pass memcheck, initcheck, racecheck, and synccheck to become qualified.

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

Run the last command in `profiler-worker` with Compose's non-root UID 10002 and
`SYS_ADMIN`-only capability setup. It fails if NSYS, NCU, structured metrics,
or any sanitizer plan is unavailable. Legacy artifacts remain readable; raw
`.cu + launch_gpu_implementation()` executables are not converted into v2.
