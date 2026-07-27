# KernelBlaster quick start

**English** | [简体中文](quickstart.zh-CN.md)

KernelBlaster is a research and performance-engineering prototype, not one command that runs on every machine. The shortest correct path depends on your goal.

## 1. Choose a path

| Goal | NVIDIA GPU | LLM key | Entry point |
| --- | ---: | ---: | --- |
| Read source, inspect configuration, or create a dry-run record | No | No | Documentation, `scripts/run_portfolio.py --dry-run` |
| Learn and manually optimize a fixed CUDA operator | Yes | No | `portfolio/case_studies/`, `scripts/benchmark_cuda.py` |
| Reproduce Core 10 candidates and PyTorch comparisons | Yes | No | `scripts/benchmark_candidates.py`, `scripts/benchmark_pytorch.py` |
| Check secure execution infrastructure | Yes | Yes (preflight includes one bounded auth probe) | Compose, `scripts/run_preflight.py` |
| Run a complete Agent candidate search | Yes | Yes | Read “Agent integration status” below first |

macOS is suitable for reading, editing, and selected CPU/dry-run tools, but it does not replace the declared Linux/WSL2 NVIDIA CUDA environment.

## 2. Get the code and identify a task

```bash
git clone https://github.com/shunwendongan/KernelBlaster.git
cd KernelBlaster
```

A KernelBench-CUDA task typically contains:

```text
data/kernelbench-cuda/level1/036_RMSNorm/
├── init.cu       # upstream CUDA implementation to optimize
└── driver.cpp    # inputs, reference, correctness, and launch contract
```

Do not modify the kernel first. Extract this contract from `driver.cpp`:

- input, output, dtype, shape, and layout;
- host entry point and argument order;
- stream, synchronization, and resource-lifetime requirements;
- tolerances, NaN/Inf behavior, and edge cases;
- whether formal timing covers only the kernel or includes setup/synchronization.

## 3. Shortest no-GPU entry point

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/run_portfolio.py --suite rmsnorm --dry-run \
  --output-dir out/portfolio/rmsnorm/dry-run
```

The first command only reads the candidate capability manifest. The second resolves the suite, budget, and output contract and writes an explicitly labeled dry-run record without API or CUDA calls.

Continue with the [source architecture](architecture.md), [RMSNorm case study](portfolio/rmsnorm-case-study.md), and [operator development guide](operator-development.md).

## 4. Shortest manual operator loop

This path does not need an LLM. For RMSNorm:

1. Keep `data/kernelbench-cuda/level1/036_RMSNorm/init.cu` unchanged as the upstream baseline.
2. Read the V0-to-V3c hypothesis chain in `portfolio/case_studies/rmsnorm/README.md`.
3. Copy a candidate to a new experiment file; do not overwrite reviewed evidence.
4. Change one primary hypothesis at a time: mapping, vectorization, reduction, or block size.
5. Pass the official and edge drivers before comparing CUDA Events.

On a supported NVIDIA environment, the fixed-candidate command shape is:

```bash
python scripts/benchmark_cuda.py \
  --task-dir data/kernelbench-cuda/level1/036_RMSNorm \
  --task-id 036 \
  --kernel RMSNorm \
  --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu \
  --candidate-name my-rmsnorm \
  --extra-correctness-driver portfolio/case_studies/rmsnorm/edge_driver.cpp \
  --warmup 20 \
  --repetitions 100 \
  --sessions 3 \
  --output-dir out/experiments/rmsnorm/<run-id>
```

Three sessions are useful for discovery. A published confirmation should follow the five-session same-GPU Portfolio protocol. Do not promote discovery numbers into the README.

## 5. Reproduce reviewed candidates

Inspect the machine-readable boundary first:

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/benchmark_candidates.py \
  --describe-capabilities --task-id 036
```

Replay one task on a matching `sm_86` environment:

```bash
python scripts/benchmark_candidates.py \
  --task-id 036 \
  --phase confirmation \
  --warmup 20 \
  --repetitions 100 \
  --sessions 5 \
  --output-dir out/portfolio/candidates/<run-id>
```

`hardened` means extra correctness and resource-lifecycle contracts exist within declared shape/dtype/layout/stream boundaries. It does not mean general production readiness. `legacy_research_only` candidates are research records.

## 6. Secure infrastructure entry point

The root `compose.yaml` is the sole deployment specification. It separates:

- `control`: CPU control plane, SQLite/CAS, and LLM configuration;
- `gpu-supervisor`: controlled GPU Jobs and ephemeral sandboxes;
- `profiler-worker`: fixed NSYS/NCU plans;
- `dev`: trusted interactive CUDA development.

Keep secrets outside the repository and generate distinct values for all four token audiences:

```bash
mkdir -p "$HOME/secrets" "$HOME/runs/KernelBlaster/state"
cp -n .env.example "$HOME/secrets/KernelBlaster.control.env"
export KERNELBLASTER_CONTROL_ENV_FILE="$HOME/secrets/KernelBlaster.control.env"
export KERNELBLASTER_STATE_HOST_DIR="$HOME/runs/KernelBlaster/state"
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" config
```

Generated-code Jobs additionally require an immutable GPU Job image digest, Docker socket group, and a Supervisor-only private evaluation profile. See [Portfolio architecture](portfolio/architecture.md).

## 7. Agent integration status on `master`

The default branch implements:

- OpenAI-compatible Provider integration, budgets, and run recording;
- the RL/knowledge-base research path;
- Control, SQLite/CAS, GPU Jobs, ephemeral sandboxing, and a Profiler Worker;
- ordered runtime preflight and capability reports;
- correctness-first fixed-candidate benchmarks and Portfolio evidence.

However, the secure `sandbox` backend on `master` explicitly refuses local-driver fallback. The CandidateEvaluator that turns Agent source into structured candidates and submits them to the sandbox is in the stacked development line after PR-07b. The obsolete local one-command wrapper has therefore been removed. `scripts/run_trusted_pilot.py` orchestrates preflight, but end-to-end Agent availability must not be claimed before the candidate funnel reaches the mainline; use the fixed-candidate `benchmark_cuda.py` path for operator optimization on `master`.

See [development history and branch status](development-history.md).

## 8. Reading outputs

| Output | Meaning | Sufficient for a speedup claim? |
| --- | --- | ---: |
| `final_rl_cuda_perf.cu` | Agent-selected terminal candidate | No; inspect correctness and measurement |
| `state.json` | Graph state snapshot | No |
| `run_manifest.json` | Configuration, source, and environment context | No |
| `events.jsonl` | Execution and decision events | No |
| `summary.json` | Terminal state, budgets, and usage summary | Summary only |
| benchmark `summary.json` | Correctness, samples, and session statistics | Evidence input within protocol/hardware boundary |
| `artifacts/portfolio-v*/` | Reviewed checked-in evidence package | Only for claims explicitly made by the package |

The complete optimization checklist is in the [high-performance operator guide](operator-development.md).
