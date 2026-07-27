# 用 KernelBlaster 开发高性能 CUDA 算子

[English](operator-development.md) | **简体中文**

本文给出一条可以重复使用的优化路线。目标不是“让 LLM 多生成几版代码”，而是把算子优化变成有契约、有假设、有反证据和可复现结果的工程过程。

## 1. 优化成功的定义

一个候选只有同时满足以下条件，才有资格被称为优化结果：

1. **语义正确**：输出、dtype、shape、特殊值和边界输入满足任务契约。
2. **执行正确**：launch、stream、同步、资源所有权和错误处理符合 Driver 约定。
3. **测量可比**：基线与候选使用同一设备、输入、计时范围、warmup 和 Session 协议。
4. **统计可信**：提升不是单次最快值，而能跨样本和 Session 稳定复现。
5. **适用边界明确**：GPU 架构、numerics、layout、shape、方向和 fallback 被写清楚。

“编译通过”和“单个 canonical shape 更快”都不满足完整定义。

## 2. 第一步：写出算子契约

从 `driver.cpp`、TaskSpec 或能力清单中提取以下内容：

| 维度 | 必须记录的事实 | 常见错误 |
| --- | --- | --- |
| 数学语义 | 公式、归约维度、广播与输出形状 | 把近似公式当作等价公式 |
| 数值语义 | 输入/累加/输出 dtype、atol/rtol、NaN/Inf | FP16 全程累加导致误差漂移 |
| 内存语义 | layout、连续性、对齐、alias 与输入可变性 | `half2` 路径未检查对齐或奇数尾部 |
| 执行语义 | host 入口、参数 ABI、stream、同步 | 在 Kernel 内正确但 host launcher 破坏协议 |
| 资源语义 | workspace、handle、初始化与释放 | 把首次初始化或分配混入计时区间 |
| 支持范围 | GPU SM、shape、forward/backward、graph capture | 把单卡单 shape 结果写成通用能力 |

建议把契约写进实验笔记或候选 manifest，再开始修改源码。

## 3. 第二步：建立不可移动的基线

基线不是“最慢的旧代码”，而是比较坐标系：

- 保留上游 `init.cu` 原文件，不在其上覆盖式迭代；
- 记录源码 digest、编译架构、编译器和关键 flags；
- 先运行 correctness-only，再进入计时；
- 记录 warmup、repetitions、inner loops、Session 数和输入 seed；
- CUDA Events 用于排名，host wall time 只用于观察端到端成本；
- NCU/NSYS 用于解释，不替代同源计时。

发现阶段可以快速，但确认阶段必须重新从干净进程和明确协议运行。不要把两阶段的数据混为同一个结果。

## 4. 第三步：建立性能假设

先用代码和问题规模估算，再决定需要哪些 Profiler 证据。

### 4.1 访存受限信号

- 每个输出只做少量 FLOP，却读写大量元素；
- 相邻线程访问跨大 stride；
- 同一输入被重复读取；
- 标量 load/store 可以安全变为 `half2`、`float2` 或更宽向量；
- 中间结果落到 global memory，而本可在寄存器或 shared memory 中复用。

优先尝试线程映射、连续访问、向量化、fusion 和减少往返内存。

### 4.2 计算受限信号

- arithmetic intensity 高；
- 大量 transcendental、矩阵乘加或重复索引计算；
- 指令吞吐、依赖链或特定 pipeline 可能成为瓶颈。

优先检查 Tensor Core/vendor library、数学等价变换、展开、指令级并行与减少重复计算。

### 4.3 并行度与调度信号

- block 数远大于实际独立工作单元，launch/调度开销过高；
- block 太少，GPU 无法填满；
- 单线程工作过多，长依赖链降低延迟隐藏；
- block size、寄存器和 shared memory 限制 occupancy；
- 归约同步次数与数据规模不匹配。

优先调整 work-per-thread、block/grid、warp 归约、shared memory 和寄存器压力。

## 5. 第四步：一次只验证一个主假设

给每个候选一个可解释名称，并记录唯一主变化：

```text
V0 upstream baseline
V1 remap one thread to one spatial position
V2 add half2 for aligned even-stride inputs
V3a change block size from 256 to 128
V3b process two vector pairs per thread
V3c replace sqrt/divide with rsqrt/multiply
```

KernelBlaster 的 RMSNorm 案例就是这个模式。V1 的线程映射是主要收益，后续版本用于验证向量化、work-per-thread 与数学指令的增量影响。负结果也应保留，因为它能排除错误方向。

## 6. 第五步：正确性门控

建议按由便宜到昂贵的顺序执行：

1. 编译与 ABI 检查；
2. canonical shape 的参考对比；
3. boundary、odd、neighbor shape；
4. 多 seed 与重复执行确定性；
5. NaN/Inf、最大误差、p99 误差与相对误差；
6. 输入未被意外修改、guard/canary 和越界检查；
7. 资源复用、多 host thread、stream 绑定与释放；
8. 需要发布时再做 sanitizer/memcheck 类检查。

任何正确性失败都必须阻断性能排名。不要通过放宽全局阈值来“修复”仅某个候选的错误；应解释数值边界或修正实现。

## 7. 第六步：性能门控

### 探索阶段

- 多次 warmup 后测量；
- 使用多样本中位数，不用单次最小值；
- 基线与候选使用相同输入和计时范围；
- 三个独立 Session 足以筛掉明显不稳定方向；
- 每次只保留具有解释的结果。

### 确认阶段

- 使用五个独立 Session；
- 配对或交替 AB/BA 顺序降低温度与运行顺序偏差；
- 检查 Session spread 和 Bootstrap 置信区间；
- 明确 no improvement 与 inconclusive，而不是只报告正结果；
- 将源码、环境、原始 summary 和分析结果绑定到 digest。

NCU/NSYS 的失败只能让诊断不可用，不能把正确性通过改成失败，也不能把 CUDA Events 的排名改成 profiler 排名。

## 8. 常见 CUDA 优化手段如何选择

| 手段 | 适合的证据 | 主要风险 | 最少应补的测试 |
| --- | --- | --- | --- |
| 线程/数据重映射 | stride 大、访存不合并、block 过多 | 索引错误、覆盖不全 | odd/neighbor shape |
| 向量化访存 | 连续且满足对齐 | 非对齐、尾部、alias | 奇数长度与非对齐边界 |
| Warp 归约 | 归约宽度接近 warp | mask、活跃 lane 与数值顺序 | 小于/大于 warp 的尺寸 |
| Shared-memory tiling | 数据跨线程复用 | bank conflict、容量、同步 | 多 block size 与 sanitizer |
| Kernel fusion | 中间 global traffic 明显 | 寄存器膨胀、并行度下降 | 每个融合分支与端到端误差 |
| Work-per-thread | launch/调度开销高 | 长依赖链、寄存器压力 | 多 shape 与 occupancy 解释 |
| Tensor Core/vendor library | 矩阵型计算、支持 dtype/layout | 初始化、workspace、边界与可移植性 | 资源生命周期和 fallback |
| 快速数学 | 指令成本高且误差允许 | 数值语义改变 | p99/max、特殊值和多 seed |

不要根据“常见技巧列表”机械套用。每次选择都要能对应一个可观察瓶颈。

## 9. 代码审查清单

### Kernel 本体

- grid 和 block 覆盖完整，64 位索引不会被中间 32 位表达式截断；
- `__restrict__`、const 和向量类型与真实 alias/对齐条件一致；
- 每个条件分支都覆盖 tail 和零/小尺寸；
- warp intrinsic 使用正确 mask；
- shared memory 无越界，必要同步没有遗漏；
- 寄存器、spill、stack 和 shared memory 使用量可解释；
- 不在 timed region 内进行不必要分配、初始化或设备同步。

### Host launcher

- ABI 与 Driver 完全一致；
- 使用约定的 stream，不偷偷切换设备或全局同步；
- handle/workspace 的所有权、预热和释放有清晰生命周期；
- launch error 能被上层捕获；
- 不把仅支持特定 shape 的实现静默用于其他输入。

### 结果与文档

- speedup 的分母、单位和比较范围明确；
- 失败和无法定论的候选也被保留；
- 手工候选不被写成 Agent 搜索结果；
- 历史硬件结果不被写成当前机器验证；
- 能力清单的 `production_ready` 与文字描述一致。

## 10. 如何让 Agent 帮你而不替代工程判断

给 Agent 的上下文应包含：

- 完整算子契约与禁止修改项；
- 当前基线源码和测量摘要；
- 一个明确瓶颈假设；
- 过去候选的成功与失败原因；
- 允许的 CUDA/C++ 依赖与目标 SM；
- 输出格式、预算和终止条件。

让 Agent 生成候选，不要让它生成“性能结论”。结论必须来自 correctness gate、CUDA Events 和审核后的统计分析。

## 11. 建议练习

1. 解释 RMSNorm V0 的线程映射为什么造成 stride 访存，并画出一个 warp 的地址关系。
2. 给 `spatial_size` 为奇数的输入设计 `half2` + scalar tail，而不产生未对齐访问。
3. 估算某算子的 FLOP、global-memory bytes 和 arithmetic intensity，判断首要假设。
4. 给一个“候选中位数快 2%，Session spread 6%”的结果决定 improved、no improvement 或 inconclusive，并说明理由。
5. 检查一个使用 cuBLAS handle 的 launcher，列出 timed region 之外必须完成的资源操作。
6. 比较 CUDA Events 与 NCU elapsed cycles：它们何时可用于排名，何时只能用于诊断？

完成练习后，选择 `data/kernelbench-cuda/level1/` 中一个较小任务，为 V0、V1 和一个失败候选各写一段假设与反证据。这个过程比一次生成大量候选更能建立可迁移的性能工程能力。
