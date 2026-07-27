# KernelBlaster core source guide

**English** | [简体中文](source-guide.zh-CN.md)

This guide answers only “in what order should I read the code?” See [source architecture](architecture.md) for directory ownership and the [operator development guide](operator-development.md) for CUDA optimization practice.

## Before reading

The default branch preserves both a classic Agent research path and secure execution infrastructure. At audit time `master` ends at PR-07a; the structured CandidateEvaluator, generic Harness, and Candidate Package remain in the later stacked line. Do not assume the paths are fully connected.

## 1. Start with fact types

Read in this order:

1. `src/kernelblaster/measurements.py`
2. `src/kernelblaster/outcomes.py`
3. `docs/measurement-status-contract.md`

Establish three distinctions:

- `Measurement` carries unit, source, protocol, and hardware fingerprint;
- `RunOutcome` represents improved, no improvement, blocked, failed, or timeout;
- diagnostic unavailable is not correctness failed and cannot overwrite CUDA Events ranking.

## 2. Read configuration and shared state

- `config/config.py`: Provider, service URLs, token audiences, and Workflow parameters;
- `config/gpu_config.py`: user-facing GPU names to `GPUType`;
- `graph/state.py`: fields shared by Graph nodes and JSON persistence.

Track `model`, `gpu`, `folder`, `cuda_fp`, `test_code_fp`, rollout parameters, `shared_optimization_database`, and `run_outcome`. Configuration describes intent, GraphState carries process state, and RunOutcome records terminal fact.

## 3. Connect the classic Agent call chain

Follow:

1. `scripts/run_RL.py::async_main`
2. `scripts/run_RL.py::process_problem`
3. `workflow/workflow.py::run_workflow`
4. `graph/graph.py::build_graph`
5. `graph/nodes/optimization_rl_ncu.py::optimization_rl_ncu`
6. `agents/opt_ncu_rl.py::RLNCUAgent.initialize`
7. `agents/opt_ncu_rl.py::RLNCUAgent.run`

`run_RL.py` owns arguments, dataset, backend, and run recording. Workflow owns top-level timeout and terminal convergence. The Graph currently has one main optimization node. The Agent owns candidate search.

`NCU` in class names is historical. CUDA Events can rank candidates; NCU/NSYS remain diagnostic.

## 4. Understand rollouts and optimization memory

Read by calls rather than linearly:

- `agents/feedback.py`: lifecycle of one candidate attempt;
- `agents/rl_agents.py`: `TrajectoryStep`, `Trajectory`, `ReplayBuffer`, and policy updates;
- `agents/database.py`: performance states, candidate optimizations, similarity, confidence, persistence;
- `agents/utils/query.py`: prompt trimming, code extraction, and Provider calls;
- `agents/reprofile.py`: classic NCU re-analysis path.

Keep predicted improvement, measured speedup, reward, and database confidence separate. Model predictions do not replace measurement, and failed trajectories remain useful evidence.

## 5. Understand correctness-first measurement

Read:

- `benchmarking.py`: host-launcher discovery, compilation-unit splitting, driver instrumentation, session statistics;
- `profiling.py`: Events backends, profiler results, and performance gates;
- `servers/cuda_env/correctness_metrics.h`: errors, NaN/Inf, and numerical statistics;
- `scripts/benchmark_cuda.py`: end-to-end fixed-candidate protocol.

Check that baseline and candidate use identical inputs, device, and timing scope; units remain explicit; correctness precedes timing; session median, spread, and bootstrap lower bound jointly decide outcomes; and controlled text scanning handles templates, macros, multiline calls, and nested delimiters.

## 6. Separate the execution paths

### Classic trusted path

Read `servers/compile.py`, `servers/gpu.py`, `servers/management.py`, `resources/client.py`, and `agents/utils/commands.py`. This path uses external Compile/GPU services for trusted `init.cu + driver.cpp` workflows. The top-level Agent runner no longer owns child-service lifecycles; callers must configure remote service URLs explicitly or deploy services through standalone worker entry points.

### Secure execution plane

Read in order:

1. `servers/control.py`
2. `storage/repository.py` and `storage/cas.py`
3. `gpu_jobs/contracts.py`, `supervisor.py`, and `sandbox.py`
4. `profiler_jobs/contracts.py` and `worker.py`
5. `preflight/contracts.py`, `runner.py`, and `backends.py`

The secure path accepts digests, fixed stages, resource limits, and deadlines. Generated Jobs receive no LLM key, Docker socket, or arbitrary command. Sandbox failure cannot fall back to trusted-local.

## 7. Outputs and troubleshooting

Classic task directories usually contain:

- `state.json`: Graph snapshot;
- `rl_ncu/`: candidates, logs, and intermediate analysis;
- `final_rl_cuda_perf.cu`: emitted only for improved;
- `failed_rl_cuda_perf` and `.finished`: failure and completion markers.

The secure path also has SQLite run/job/lease metadata, CAS artifacts, and `capability-report/v1`.

Troubleshoot in this order:

1. inspect `RunOutcome.status`, `reason_code`, and four sub-statuses;
2. inspect capability-report expiry, hardware mismatch, and hard checks;
3. for compilation, inspect split units, command, and stderr;
4. for correctness, inspect the driver, edge inputs, and special values;
5. for performance, inspect paired sessions, units, device concurrency, and noise;
6. for Provider failures, inspect budgets, retries, and redacted events;
7. when NSYS/NCU is unavailable, preserve Events facts instead of claiming complete diagnostics.

## 8. Core invariants

1. Candidates that fail correctness never enter ranking.
2. Microseconds, cycles, and seconds are never compared implicitly.
3. Success, exception, cancellation, and timeout all converge Futures and RunOutcome.
4. Candidate code is always treated as untrusted input.
5. Sandbox never falls back implicitly to trusted-local.
6. Critical state and artifacts are recoverable, traceable, and digest-bound.
7. Predicted gain, measured gain, and knowledge confidence stay separate.
8. Prompts, tokens, API keys, and authorization headers never enter public logs.

## 9. Self-check questions

- Which layer cancels child processes after a top-level timeout?
- How do `NO_IMPROVEMENT` and `FAILED` differ in aggregate statistics?
- Which path is used by `runtime is None`, `trusted_local`, and `sandbox`?
- Why must NCU permission denial not overwrite CUDA Events results?
- What bias appears if the replay buffer keeps only successful trajectories?
- Which artifacts and environment facts are required for a speedup claim?
