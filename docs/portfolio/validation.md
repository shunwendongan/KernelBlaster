# Deferred Validation and Benchmark Protocol

This document is a protocol, not a result report. No Python entrypoint, LLM API, CUDA compiler, GPU server, Docker container, Nsight Compute command, or benchmark has been executed for the current Mac-only implementation commit.

## Current status

| Gate | Status | Evidence required to close it |
| --- | --- | --- |
| Provider mock tests | NOT RUN | fan-out, retries, budget, missing-usage, redaction tests |
| Real gateway smoke test | NOT RUN | one successful request using the configured model alias |
| RTX 3080 compilation | NOT RUN | compile log and environment manifest |
| CUDA correctness | NOT RUN | driver output for every accepted candidate |
| NCU profiling | NOT RUN | Elapsed Cycles plus selected bottleneck metrics |
| CUDA Events timing | NOT RUN | warmup and repeated independent timing samples |
| Cross-GPU comparison | NOT RUN | matching 3080 and L40S/A100 configurations |
| Performance results | pending | reviewed tables generated from recorded artifacts |

## Validation order

1. Run CPU-only provider tests with a fake Chat Completions client. Check client-side fan-out, concurrency, retry classification, request budget, token accounting, and secret redaction.
2. Execute `run_portfolio.py --dry-run` and validate all three artifact schemas without an API or GPU.
3. Send one real gateway request using the exact externally supplied model ID. Do not begin CUDA rollouts until this smoke test passes.
4. On WSL2 with RTX 3080, record driver, CUDA toolkit, container, Python dependency, and Git commit information. Verify compilation, correctness, GPU server startup, and NCU permission.
5. Run the Core 10 suite with `3 rollouts × 3 steps`. Serialize GPU compilation and evaluation; cap LLM concurrency at four.
6. Run the RMSNorm case study with `8 rollouts × 5 steps`. If there is no correct improvement of at least 5% by day 9, switch the deep case study to task 047 Sum Reduction.
7. Use NCU Elapsed Cycles only as the search signal. Re-measure final candidates independently with CUDA Events after warmup and repeated runs.
8. Re-run RMSNorm, Softmax, and Small-K MatMul on RTX 3080 and L40S. Use A100 only when L40S is unavailable, and report the substitution.

## Result acceptance rules

- A candidate is rejected if any correctness input fails, even if its timing is faster.
- Report hardware, precision, shapes, warmup count, repetition count, central statistic, dispersion, CUDA version, driver, and commit with every performance table.
- Compare against the unmodified upstream CUDA kernel and a relevant PyTorch baseline; label each baseline separately.
- Do not compare NCU Elapsed Cycles directly across GPU architectures.
- Do not publish a speedup until the final CUDA Events measurement is reproducible and traceable to a committed kernel.
- Upstream KernelBlaster paper results are background context, not evidence for this fork.

The language model remains an external inference service throughout this work. Rollouts update search trajectories and the optimization database; they do not train or fine-tune model weights.
