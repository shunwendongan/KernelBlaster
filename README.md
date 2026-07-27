# KernelBlaster

**English** | [简体中文](README.zh-CN.md)

KernelBlaster is a **correctness-first, profile-guided CUDA kernel optimization research framework with persistent optimization memory**. It brings KernelBench-CUDA tasks, LLM candidate generation, rollout/replay, CUDA Events, NSYS/NCU diagnostics, reproducible experiments, and evidence management into one repository.

> This is a trusted research prototype, not a general production operator library. Every performance statement must remain bound to a GPU, shape, dtype, layout, correctness protocol, and measurement method.

## Start here

| Goal | Shortest path |
| --- | --- |
| Understand the project in ten minutes | [Documentation index](docs/README.md) → [Quick start](docs/quickstart.md) |
| Understand the source tree | [Source architecture](docs/architecture.md) → [Core source guide](docs/source-guide.md) |
| Learn to write fast operators | [Operator development guide](docs/operator-development.md) → [RMSNorm case study](docs/portfolio/rmsnorm-case-study.md) |
| Reproduce fixed candidates | `scripts/benchmark_candidates.py`, `scripts/benchmark_cuda.py` |
| Inspect validation evidence | [Portfolio status](docs/portfolio/README.md) → `artifacts/portfolio-v*/` |
| Understand branch evolution | [Development history and branch status](docs/development-history.md) |

## Checked-in Portfolio evidence

<!-- PORTFOLIO_STATUS:START -->
The checked-in Portfolio evidence records the Day 1–10 infrastructure, RMSNorm deep case, manual Core 10 candidates, and a same-GPU PyTorch comparison on **NVIDIA GeForce RTX 3080 (sm_86)**. The recorded environment is WSL2, CUDA 12.8.61, and driver 591.86.

| Validation item | Current status |
| --- | --- |
| CPU tests | **177 passed** (latest record in `portfolio/status.json`) |
| CUDA build and official correctness | **historical 10/10; schema-v2 full 10/10 passed** |
| CUDA Events and same-GPU PyTorch | **schema v2 full: 4 improved, 1 no improvement, 5 inconclusive; 9/10 tasks have a stable PyTorch method** |
| External LLM smoke | **failed: current HTTP 401 (1 request, 0 retries, 0 tokens; 2026-07-22)** |
| Nsight Compute counters | **blocked: ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)** |
| Cross-GPU rerun | **blocked: requires authorized A100/L40S rental** |

| Historical v1 scope | Versus upstream (diagnostic / old strict gate) | Versus fastest PyTorch method (diagnostic / old strict gate) |
| --- | ---: | ---: |
| Nine new candidates | 5.020× / 3.302× | 1.415× / 0.931× |
| Full Core 10, including RMSNorm | 6.351× / 4.356× | 1.447× / 0.992× |

These immutable strict values remain historical v1 evidence. A separate full manual schema-v2 confirmation passed 10/10 correctness, formally confirmed 004/007/036/040, classified 088 as no improvement, and left 019/023/026/047/095 inconclusive. Under the current gate, the strict Core 10 geometric mean versus upstream is 4.381×; across the 9/10 tasks with a correct and stable PyTorch method, the strict ratio versus the fastest stable method is 1.053×. This is still not an Agent-search result. The new gate also checks p99/max error regression, NaN/Inf, and five-run determinism. Neither the Agent Pilot nor Core 10 Agent search has run.

[Schema-v2 full Core 10 validation](artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.en.md) · [Schema-v2 full result JSON](artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json) · [Schema-v2 targeted validation](artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.en.md) · [Schema-v2 result JSON](artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [Full Chinese report](artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [English summary](artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [Per-task JSON](artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [Comparison figure](artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [Raw-file hashes](artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [Candidate manifest](portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_STATUS:END -->

### Reproduce the validated RTX 3080 comparison

Run these commands inside the pinned NGC 25.01 container on an `sm_86` GPU. Raw outputs remain below ignored `out/portfolio/` paths; reviewed artifacts are checked in separately.

```bash
python scripts/benchmark_candidates.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>

python -m pytest -q
python scripts/sync_portfolio_docs.py --check
```

The Agent optimization loop performs rollout search and memory updates; it does not fine-tune the underlying language-model weights. The results above come from manual candidates and must not be described as Agent-search results.

### Portfolio v2.1 evidence

The v2.1 publication hardens the five Issue #10 CUDA candidates without expanding their production claim. The stable contract accepts only reviewed `sm_86`, FP16, contiguous row-major, legacy-default-stream, single-stream, forward-only, non-graph-capture, manifest-approved cases; `production_ready` remains `false`.

- [Evidence index and SHA-256 manifest](artifacts/portfolio-v2.1/SHA256SUMS.json)
- [Five-task correctness and lifecycle summary](artifacts/portfolio-v2.1/issue-10/rtx3080/correctness-summary.json)
- [Issue #7 API/Pilot status](artifacts/portfolio-v2.1/issue-7/rtx3080/trusted-pilot-summary.json) — HTTP 401; Pilot not run
- [Issue #8 profiler status](artifacts/portfolio-v2.1/issue-8/rtx3080/ncu-preflight-summary.json) — Windows-native NCU/NSYS evidence published; WSL counters and cross-GPU reruns remain open

## What problem does it solve?

CUDA optimization is not a syntax rewrite. Performance depends on layout, thread mapping, coalescing, reductions, instruction throughput, occupancy, GPU architecture, and measurement noise. Fixed compiler heuristics cannot cover every combination, while a naive LLM Agent tends to forget earlier exploration and repeat failures.

KernelBlaster narrows the search through this loop:

1. read `init.cu`, `driver.cpp`, and the task contract;
2. establish a correctness-passing initial performance baseline;
3. derive a performance state from profiler/measurement results;
4. retrieve candidate strategies from persistent optimization memory;
5. ask the LLM for a candidate with an explicit hypothesis;
6. compile and verify correctness before CUDA Events timing;
7. write successes and failures into replay and the optimization database;
8. persist terminal state and the best candidate through `RunOutcome`.

The LLM proposes candidates; it does not declare performance. CUDA Events rank candidates, NSYS/NCU diagnose them, and any correctness failure blocks ranking.

## Implemented on `master`

| Capability | Key modules | Status |
| --- | --- | --- |
| MAIC-RL-style optimization loop | `agents/opt_ncu_rl.py`, `rl_agents.py`, `database.py` | Classic research path, replay, and cross-task memory |
| OpenAI-compatible LLM | `llm/` | Concurrency, retries, budgets, usage, and redacted records |
| Correctness-first measurement | `benchmarking.py`, `profiling.py`, `measurements.py` | Explicit units, sources, protocols, and unavailable reasons |
| Standard terminal state | `outcomes.py`, `docs/measurement-status-contract.md` | Improved, no improvement, blocked, failed, and timeout |
| Secure Control plane | `servers/control.py`, `storage/` | SQLite metadata separated from SHA-256 CAS |
| GPU Jobs and sandbox | `gpu_jobs/` | Digest-only manifests, hardware capabilities, ephemeral no-network containers |
| Profiler Worker | `profiler_jobs/` | Fixed NSYS/NCU plans; diagnostics do not rank |
| Runtime preflight | `preflight/` | Ordered Provider, storage, GPU, sandbox, Events, and diagnostic checks |
| Portfolio | `portfolio/`, `artifacts/`, `scripts/benchmark_*.py` | Fixed candidates, strict protocols, reports, and hash evidence |

## Important mainline boundary

At audit time the default `master` branch ends at PR-07a. PR-07b through PR-09 form a stacked line merged into successive development branches, not yet as a whole into `master`. That line contains the Agent candidate funnel, generic correctness harness, independent baseline providers, CUDA/Triton candidate packages, AutoDL portability, and release orchestration.

Two facts follow on the current mainline:

- the secure `sandbox` backend fails closed instead of falling back to local `driver.cpp` execution;
- the CandidateEvaluator that turns Agent source into structured sandbox submissions remains in the stacked development line.

The obsolete local one-command wrapper has been removed rather than kept as a misleading entry point. Use explicit fixed-candidate tools on `master`; see [development history and branch status](docs/development-history.md).

## Quick check without GPU or API calls

```bash
python scripts/benchmark_candidates.py --describe-capabilities
python scripts/run_portfolio.py --suite rmsnorm --dry-run \
  --output-dir out/portfolio/rmsnorm/dry-run
```

This is suitable on macOS for inspecting manifests, suites, budgets, and artifact structure. For CUDA candidate development, secure deployment, and result interpretation, use the [quick start](docs/quickstart.md); source ownership is documented once in [source architecture](docs/architecture.md).

## Paper and citation

**arXiv:** [arXiv:2602.14293](https://arxiv.org/abs/2602.14293) · **Repository PDF:** [KernelBlaster.pdf](docs/figures/KernelBlaster.pdf)

The upstream authors report geometric-mean speedups over PyTorch of 1.43×, 2.50×, and 1.50× on KernelBench Levels 1, 2, and 3. Those paper-wide results are background and remain separate from this fork's manual RTX 3080 Core 10 evidence.

```bibtex
@article{dong2026kernelblaster,
  title={KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning},
  author={Dong, Kris Shengjun and Modi, Sahil and Nikiforov, Dima and Damani, Sana and Lin, Edward and Hari, Siva Kumar Sastry and Kozyrakis, Christos},
  journal={arXiv preprint arXiv:2602.14293},
  year={2026}
}
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. New benchmarks, candidates, or artifacts must update README/docs or `portfolio/status.json`, and English/Chinese documentation pairs must remain synchronized.

Upstream contributors include Kris Shengjun Dong, Sahil Modi, Dima Nikiforov, Sana Damani, Edward Lin, Siva Kumar Sastry Hari, and Christos Kozyrakis. Most upstream project work was completed by Kris Shengjun Dong during a 2025 NVIDIA internship.
