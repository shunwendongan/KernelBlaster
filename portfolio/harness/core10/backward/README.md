# Core 10 backward baselines

These CUDA files are deliberately simple, deterministic starting points for
the versioned backward TaskSpecs. They use FP16 inputs/outputs and FP32
accumulation. PyTorch autograd remains the correctness oracle; these kernels
are the mandatory upstream-CUDA performance baselines, not vendor-library
implementations.

Every exported entry is device-only so the AOT candidate runtime can load it
without executing candidate-controlled host code. Allocation, stream choice,
workspace ownership, correctness comparison, and timing belong to the trusted
Harness.
