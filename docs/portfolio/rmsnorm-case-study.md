# RMSNorm Optimization Case Study

Current status: **design only; no CUDA implementation or measurement has been validated**.

## Question

Can profile-guided agent search produce a correct RMSNorm kernel with at least 5% lower independently measured latency than the unmodified upstream CUDA implementation, and can the performance explanation be defended using memory traffic, reduction structure, occupancy, and instruction-level evidence?

## Planned versions

| Version | Change | Hypothesis | Required evidence |
| --- | --- | --- | --- |
| V0 | Unmodified upstream kernel | Establish correctness, cycles, latency, and bottleneck baseline | driver output, NCU metrics, CUDA Events samples |
| V1 | Thread remapping and reduction redesign | Improve coalescing and remove unnecessary cross-thread work | memory metrics, reduction diagram, occupancy |
| V2 | `half2` vectorization plus tail path | Reduce memory instructions while preserving odd-length correctness | alignment analysis, instruction counts, tail tests |
| V3 | Block size, work-per-thread, and `rsqrtf` tuning | Balance occupancy, registers, and instruction throughput | parameter sweep, register count, final timing |

No optimized `.cu` file is checked in at this stage because an uncompiled kernel would be misleading evidence.

## Experiment record template

For each version, record:

- Git commit and generated candidate hash.
- GPU, SM architecture, driver, CUDA toolkit, clock/power policy, and precision.
- Tensor shapes, dtype, alignment assumptions, and tail cases.
- Correctness tolerances and all driver outcomes.
- NCU Elapsed Cycles, DRAM/L2 throughput, occupancy, registers, warp stalls, and launch configuration.
- CUDA Events warmup, repetitions, median, p10/p90, and comparison baseline.
- Agent prompt hash, selected optimization strategy, token usage, request latency, and retry count.
- Explanation of why the change helped or failed on that GPU.

## Decision gate

Continue with RMSNorm only when a candidate is correct and the final CUDA Events result improves by at least 5%. If that gate has not been met by day 9, move the same evidence process to task 047 Sum Reduction. A failed RMSNorm investigation remains useful and should be documented as a bottleneck analysis, but it must not be presented as a speedup result.

## Resume evidence gate

This case study becomes resume-ready only after the repository contains a reproducible command, committed accepted kernel, correctness evidence, raw structured artifacts, reviewed result table, and an explanation that distinguishes measured facts from hypotheses. Until then it remains a clearly marked work in progress.
