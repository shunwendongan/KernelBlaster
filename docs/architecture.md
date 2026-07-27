# KernelBlaster source architecture

**English** | [简体中文](architecture.zh-CN.md)

This document describes the static source structure on the default `master` branch. It answers four questions: where execution starts, how a task flows, which module owns each state, and how the classic execution path coexists with the secure execution plane.

## 1. The repository contains more than one pipeline

The current repository preserves three related capabilities:

| Layer | Primary purpose | Key entry points | Current boundary |
| --- | --- | --- | --- |
| Agent research path | Search CUDA candidates with an LLM, profiler feedback, memory, and rollouts | `scripts/run_RL.py`, `src/kernelblaster/agents/` | The classic `trusted_local` path still needs external compile/GPU services |
| Secure execution plane | Run untrusted work through Control, CAS, GPU Supervisor, ephemeral sandboxes, and a Profiler Worker | `compose.yaml`, `src/kernelblaster/servers/control.py` | `master` has preflight and execution infrastructure; the generated-candidate funnel is in later stacked PRs |
| Portfolio evidence path | Run correctness-first reproducible experiments for fixed candidates and publish evidence | `scripts/benchmark_*.py`, `portfolio/`, `artifacts/` | Results are constrained by GPU, shape, dtype, layout, and protocol |

These layers share measurement, status, and evidence concepts, but they must not be presented as one interchangeable “run everything” command.

## 2. Top-level ownership

| Path | Ownership | What to look for first |
| --- | --- | --- |
| `src/kernelblaster/` | Framework implementation | Workflow, Agent, measurement, services, storage, and contracts |
| `scripts/` | Command orchestration | Task entry points, preflight, benchmarks, analysis, and docs sync |
| `data/kernelbench-cuda/` | Input tasks | Each task's `init.cu`, `driver.cpp`, and description |
| `portfolio/case_studies/` | Reviewed candidates | Manual candidates, edge drivers, capability manifests, and case studies |
| `portfolio/suites/` | Experiment selection | Core 10, Pilot, and RMSNorm task and budget definitions |
| `artifacts/` | Immutable evidence | Environments, result summaries, reports, figures, and hash manifests |
| `docs/` | Usage and audit documentation | Quick start, architecture, status contract, and Portfolio interpretation |
| `tests/` | Behavioral contracts | CPU, GPU, sandbox, profiler, security, and script expectations |
| `docker/`, `compose.yaml` | Deployment boundary | Control, Supervisor, Profiler, and development containers |

## 3. Agent research call order

On the classic research path, one task flows as follows:

1. `scripts/run_RL.py` parses dataset, GPU, model, rollout, and backend settings.
2. `data/dataset.py` and `data/kernelbench_cuda.py` resolve the task directory.
3. `run_workflow()` constructs `GraphState`, applies a top-level timeout, and normalizes terminal state.
4. `build_graph()` currently connects one main node, `optimization_rl_ncu`.
5. The node resolves `init.cu` and `driver.cpp`, then creates `FeedbackConfig` and a profiler backend.
6. `RLNCUAgent.initialize()` measures the initial implementation and establishes a performance state.
7. `RLNCUAgent.run()` coordinates rollouts, candidate generation, correctness/performance feedback, the replay buffer, and knowledge updates.
8. `RunOutcome` returns improved, no improvement, blocked, failed, or timeout to the caller.
9. A successful result is written as `final_rl_cuda_perf.cu`; state and events stay in the run directory.

The class names still contain `NCU`, but ranking does not have to come from NCU. `EventsProfilerBackend` provides low-overhead CUDA Events timing; NCU/NSYS should remain diagnostic evidence and must not overwrite correctness or ranking state.

## 4. Secure execution-plane ownership

The secure plane follows “the control plane does not execute candidates; execution workers do not receive LLM secrets”:

| Module | Responsibility | Must not own |
| --- | --- | --- |
| `servers/control.py` | Run/Job state, leases, CAS API, and Supervisor/Profiler routing | Direct CUDA candidate execution |
| `storage/repository.py` | Run/job/attempt metadata in SQLite | Large source and report bodies |
| `storage/cas.py` | Immutable artifacts addressed by SHA-256 | Terminal-state decisions |
| `gpu_jobs/supervisor.py` | Device capabilities, single-GPU queue, and Job lifecycle | Provider keys |
| `gpu_jobs/sandbox.py` | Ephemeral containers, fixed resources, no network, output allowlists | Caller-controlled arbitrary commands |
| `profiler_jobs/worker.py` | Fixed NSYS/NCU plans and structured summaries | CUDA Events ranking |
| `preflight/runner.py` | Ordered checks for Provider, storage, GPU, sandbox, Events, and diagnostics | Pretending partial availability is complete availability |

`preflight/backends.py` is intentionally fail-closed on `master`: the `sandbox` backend cannot fall back to executing a local `driver.cpp`. The CandidateEvaluator that turns Agent output into structured candidates and submits them to the sandbox belongs to later stacked PRs. The secure infrastructure and Agent generation loop therefore still have an integration boundary on the default branch.

## 5. State and artifact relationships

Do not conflate these objects:

| Object | Meaning | Typical location |
| --- | --- | --- |
| `GraphState` | Transient data shared by task nodes | `graph/state.py`, task `state.json` |
| `RunOutcome` | Standard terminal state for an optimization task | `outcomes.py` |
| `Measurement` | A measurement with unit, source, protocol, and hardware fingerprint | `measurements.py` |
| Job/Lease state | Queueing, lease, and retry facts for distributed execution | SQLite `JobRepository` |
| CAS artifact | Immutable source, binary, log, or report content | `ContentAddressedStore` |
| RunRecorder event | Configuration, budget, prompt metadata, and execution events | `observability/recorder.py` |
| Portfolio artifact | Reviewed and checked-in experiment claim | `artifacts/portfolio-v*/` |

`state.json` aids recovery and debugging, but is not performance evidence by itself. A speedup claim also needs correctness, protocol, hardware, sample stability, and source hashes.

## 6. Performance and correctness modules

- `benchmarking.py`: locates launches in drivers, creates CUDA Events/NCU variants, and evaluates performance gates.
- `profiling.py`: normalizes Events and NCU results while preserving units and unavailable reasons.
- `result_analysis.py`: normalizes execution, correctness, timing, and diagnostic fields into the measurement schema.
- `servers/cuda_env/correctness_metrics.h`: numerical error, NaN/Inf, and correctness statistics.
- `scripts/benchmark_cuda.py`: correctness-first comparisons for fixed CUDA candidates.
- `scripts/benchmark_candidates.py`: capability-manifest-driven Core 10 replay.
- `scripts/benchmark_pytorch.py`: same-GPU PyTorch reference columns.
- `scripts/analyze_core10_comparison.py`: strict comparison conclusions over the preceding outputs.

## 7. LLM and optimization memory

`llm/` provides an OpenAI-compatible Provider abstraction for concurrency, retries, budgets, and usage; `agents/utils/query.py` handles higher-level message trimming and code extraction. Secrets belong only in the environment or an external env file.

`agents/database.py` stores “performance state → candidate optimization → historical feedback.” It helps the search reuse experience, but model predictions, database confidence, and measured speedup are three distinct signals.

## 8. Recommended reading order

1. `measurements.py` and `outcomes.py`: learn how facts are represented.
2. `config/config.py` and `graph/state.py`: understand inputs and shared state.
3. `workflow/workflow.py` and `graph/nodes/optimization_rl_ncu.py`: connect the task path.
4. `agents/opt_ncu_rl.py`, `rl_agents.py`, and `database.py`: understand the search loop.
5. `benchmarking.py` and `profiling.py`: understand correctness-first measurement.
6. `preflight/`, `gpu_jobs/`, `profiler_jobs/`, and `storage/`: understand the secure execution plane.
7. `portfolio/case_studies/rmsnorm/`: map framework concepts onto a concrete operator.

See the [core source guide](source-guide.md) for function-level reading order and the [quick start](quickstart.md) for practical entry points.
