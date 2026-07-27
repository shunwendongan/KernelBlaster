# Developing high-performance CUDA operators with KernelBlaster

**English** | [简体中文](operator-development.zh-CN.md)

This guide presents a repeatable optimization method. The goal is not to ask an LLM for many code variants, but to turn operator optimization into an engineering process with contracts, hypotheses, counter-evidence, and reproducible results.

## 1. What counts as an optimization?

A candidate qualifies only when all of these hold:

1. **Semantic correctness**: outputs, dtype, shape, special values, and edge inputs satisfy the task contract.
2. **Execution correctness**: launch, stream, synchronization, resource ownership, and error handling follow the driver contract.
3. **Comparable measurement**: baseline and candidate use the same device, inputs, timing scope, warmup, and session protocol.
4. **Statistical credibility**: the gain survives repeated samples and sessions instead of appearing only in one minimum.
5. **Declared applicability**: GPU architecture, numerics, layout, shape, direction, and fallback are explicit.

“Compiles” and “faster for one canonical shape” are not complete definitions of success.

## 2. Write the operator contract first

Extract these facts from `driver.cpp`, a TaskSpec, or the capability manifest:

| Dimension | Required facts | Common mistake |
| --- | --- | --- |
| Mathematical semantics | Formula, reduction dimensions, broadcasting, output shape | Treating an approximation as an identity |
| Numerical semantics | Input/accumulation/output dtype, atol/rtol, NaN/Inf | Accumulating entirely in FP16 |
| Memory semantics | Layout, contiguity, alignment, aliasing, input mutation | Using `half2` without alignment or odd-tail checks |
| Execution semantics | Host entry, argument ABI, stream, synchronization | Correct kernel with a launcher that breaks the contract |
| Resource semantics | Workspace, handles, initialization, release | Including first-use initialization or allocation in timing |
| Support boundary | GPU SM, shape, forward/backward, graph capture | Generalizing a one-GPU, one-shape result |

Write this contract into the experiment notes or candidate manifest before changing source.

## 3. Establish an immovable baseline

The baseline is a coordinate system, not merely old slow code:

- preserve upstream `init.cu` instead of editing it in place;
- record source digest, target architecture, compiler, and important flags;
- run correctness-only before timing;
- record warmup, repetitions, inner loops, session count, and input seed;
- use CUDA Events for ranking and host wall time only for end-to-end observation;
- use NCU/NSYS to explain results, not to replace same-source timing.

Discovery may be fast, but confirmation must restart under a defined protocol. Do not merge discovery and confirmation samples into one claim.

## 4. Form a performance hypothesis

Start with code- and shape-derived estimates, then decide which profiler evidence is needed.

### 4.1 Memory-bound signals

- little computation per output but substantial element traffic;
- adjacent threads access a large stride;
- repeated reads of the same input;
- scalar accesses can safely become `half2`, `float2`, or wider vectors;
- intermediates go to global memory despite register/shared-memory reuse.

Prioritize mapping, coalescing, vectorization, fusion, and fewer global-memory round trips.

### 4.2 Compute-bound signals

- high arithmetic intensity;
- many transcendental operations, matrix operations, or repeated index calculations;
- instruction throughput, dependency chains, or a specific pipeline may dominate.

Consider Tensor Cores/vendor libraries, equivalent math, unrolling, instruction-level parallelism, and eliminating repeated work.

### 4.3 Parallelism and scheduling signals

- far more blocks than independent work units, making launch/scheduling overhead prominent;
- too few blocks to fill the GPU;
- too much work per thread and a long dependency chain;
- block size, registers, or shared memory limit occupancy;
- reduction synchronization is disproportionate to data size.

Consider work-per-thread, grid/block shape, warp reductions, shared memory, and register pressure.

## 5. Test one primary hypothesis per candidate

Give each candidate an explanatory name and one primary change:

```text
V0 upstream baseline
V1 remap one thread to one spatial position
V2 add half2 for aligned even-stride inputs
V3a change block size from 256 to 128
V3b process two vector pairs per thread
V3c replace sqrt/divide with rsqrt/multiply
```

KernelBlaster's RMSNorm case follows this pattern. V1's mapping provides the main gain; later variants isolate vectorization, work-per-thread, and math-instruction effects. Preserve negative results because they eliminate weak directions.

## 6. Correctness gates

Run gates from cheaper to more expensive:

1. compilation and ABI checks;
2. canonical reference comparison;
3. boundary, odd, and neighboring shapes;
4. multiple seeds and repeated-run determinism;
5. NaN/Inf, maximum error, p99 error, and relative error;
6. input mutation, guards/canaries, and out-of-bounds checks;
7. resource reuse, multiple host threads, stream binding, and release;
8. sanitizer/memcheck-style checks before publication.

Any correctness failure blocks performance ranking. Do not “fix” one candidate by globally relaxing tolerance; explain the numerical boundary or fix the implementation.

## 7. Performance gates

### Discovery

- warm up before measuring;
- use a sample median, not one minimum;
- use identical inputs and timing scope for baseline and candidate;
- use three independent sessions to reject obviously unstable directions;
- retain only results with a stated explanation.

### Confirmation

- use five independent sessions;
- pair or alternate AB/BA order to reduce thermal and ordering bias;
- inspect session spread and bootstrap confidence bounds;
- represent no improvement and inconclusive outcomes instead of reporting only wins;
- bind source, environment, raw summary, and analysis to digests.

NCU/NSYS failure makes diagnostics unavailable. It must not change correctness state or replace CUDA Events ranking.

## 8. Choosing CUDA techniques

| Technique | Supporting evidence | Main risk | Minimum extra test |
| --- | --- | --- | --- |
| Thread/data remapping | Large stride, uncoalesced access, too many blocks | Indexing and coverage errors | Odd and neighboring shapes |
| Vectorized access | Contiguous and aligned data | Misalignment, tails, aliasing | Odd lengths and alignment boundaries |
| Warp reduction | Reduction width near a warp | Masks, active lanes, numerical order | Sizes below and above one warp |
| Shared-memory tiling | Cross-thread reuse | Bank conflicts, capacity, synchronization | Multiple block sizes and sanitizer |
| Kernel fusion | Significant intermediate global traffic | Register growth, lower parallelism | Every fused branch and end-to-end error |
| Work-per-thread | Launch/scheduling overhead | Dependency chains and register pressure | Multiple shapes and occupancy explanation |
| Tensor Core/vendor library | Matrix-like work and supported dtype/layout | Initialization, workspace, portability | Resource lifecycle and fallback |
| Fast math | Expensive instructions and allowed error | Changed numerical semantics | p99/max error, special values, multiple seeds |

Do not mechanically apply a checklist of tricks. Every choice should correspond to an observable bottleneck.

## 9. Code-review checklist

### Device code

- grid and block cover all work; 64-bit indices are not truncated by intermediate 32-bit expressions;
- `__restrict__`, const, and vector types match real aliasing/alignment conditions;
- every branch handles tails and tiny sizes;
- warp intrinsics use correct masks;
- shared memory is in bounds and required synchronization is present;
- register, spill, stack, and shared-memory use is explainable;
- the timed region excludes unnecessary allocation, initialization, and device-wide synchronization.

### Host launcher

- ABI exactly matches the driver;
- uses the required stream and does not silently switch device or synchronize globally;
- handle/workspace ownership, warmup, and release have explicit lifetimes;
- launch errors reach the caller;
- shape-specialized code is not silently used outside its boundary.

### Results and documentation

- speedup denominator, unit, and comparison scope are explicit;
- failures and inconclusive candidates are retained;
- manual candidates are not described as Agent-search results;
- historical hardware results are not described as current-machine validation;
- capability `production_ready` agrees with prose.

## 10. Let the Agent assist without replacing judgment

Useful Agent context includes:

- complete operator contract and forbidden changes;
- current baseline source and measurement summary;
- one explicit bottleneck hypothesis;
- prior candidate successes and failures;
- permitted CUDA/C++ dependencies and target SM;
- output format, budget, and stopping conditions.

Let the Agent propose candidates, not performance conclusions. Conclusions come from correctness gates, CUDA Events, and reviewed statistics.

## 11. Exercises

1. Explain why the RMSNorm V0 mapping causes strided access and sketch one warp's addresses.
2. Design `half2` plus scalar-tail handling for odd `spatial_size` without misaligned access.
3. Estimate FLOPs, global-memory bytes, and arithmetic intensity for an operator, then choose the first hypothesis.
4. Classify “candidate median is 2% faster, session spread is 6%” as improved, no improvement, or inconclusive.
5. Review a cuBLAS-handle launcher and list resource operations that must happen outside the timed region.
6. Compare CUDA Events with NCU elapsed cycles: when can each rank candidates, and when is it diagnostic only?

Afterward, choose a small task under `data/kernelbench-cuda/level1/` and write a hypothesis plus counter-evidence for V0, V1, and one failed candidate. This builds more transferable performance-engineering skill than generating many variants at once.
