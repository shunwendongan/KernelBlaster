"""Correctness-first CUDA AOT fixtures covering Core 10 forward and backward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..harness.contracts import Direction, TaskSpec
from .contracts import (
    CandidateBackend,
    CandidateLaunchPlan,
    CandidateProvenance,
    Dimensions,
    DispatchRule,
    KernelDeclaration,
    KernelLaunch,
)
from .package import BackendUnsupportedError, build_candidate_package


_FORWARD = {
    "004": b'''#include <cuda_fp16.h>\n#include <stdint.h>\nextern "C" __global__ void kb004_matvec(half* output, const half* A, const half* B, int64_t M, int64_t K) {\n  extern __shared__ float scratch[];\n  int64_t row = blockIdx.x; float sum = 0.0f;\n  for (int64_t k = threadIdx.x; k < K; k += blockDim.x) sum += __half2float(A[row * K + k]) * __half2float(B[k]);\n  scratch[threadIdx.x] = sum; __syncthreads();\n  for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s]; __syncthreads(); }\n  if (threadIdx.x == 0 && row < M) output[row] = __float2half_rn(scratch[0]);\n}\n''',
    "007": b'''#include <cuda_fp16.h>\n#include <stdint.h>\nextern "C" __global__ void kb007_matmul(half* output, const half* A, const half* B, int64_t M, int64_t N, int64_t K) {\n  int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (index >= M * N) return;\n  int64_t row = index / N, column = index % N; float sum = 0.0f;\n  for (int64_t k = 0; k < K; ++k) sum += __half2float(A[row * K + k]) * __half2float(B[k * N + column]);\n  output[index] = __float2half_rn(sum);\n}\n''',
    "019": b'''#include <cuda_fp16.h>\n#include <stdint.h>\nextern "C" __global__ void kb019_relu(half* output, const half* input, int64_t B, int64_t D) {\n  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < B * D) { float x = __half2float(input[i]); output[i] = __float2half_rn(x > 0.0f ? x : 0.0f); }\n}\n''',
    "023": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb023_softmax(half* output, const half* input, int64_t B, int64_t D) {\n  extern __shared__ float scratch[]; int64_t row = blockIdx.x; float local = -INFINITY;\n  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) local = fmaxf(local, __half2float(input[row * D + d]));\n  scratch[threadIdx.x] = local; __syncthreads();\n  for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]); __syncthreads(); }\n  float maximum = scratch[0], sum = 0.0f;\n  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) sum += expf(__half2float(input[row * D + d]) - maximum);\n  scratch[threadIdx.x] = sum; __syncthreads();\n  for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s]; __syncthreads(); }\n  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) output[row * D + d] = __float2half_rn(expf(__half2float(input[row * D + d]) - maximum) / scratch[0]);\n}\n''',
    "026": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb026_gelu(half* output, const half* input, int64_t B, int64_t D) {\n  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < B * D) { float x = __half2float(input[i]); output[i] = __float2half_rn(0.5f * x * (1.0f + erff(x * 0.7071067811865476f))); }\n}\n''',
    "036": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb036_rmsnorm(half* output, const half* input, float eps, int64_t B, int64_t C, int64_t D1, int64_t D2) {\n  extern __shared__ float scratch[]; int64_t p = blockIdx.x, spatial = D1 * D2, b = p / spatial, s = p % spatial; float sum = 0.0f;\n  for (int64_t c = threadIdx.x; c < C; c += blockDim.x) { float x = __half2float(input[(b * C + c) * spatial + s]); sum += x * x; }\n  scratch[threadIdx.x] = sum; __syncthreads();\n  for (int stride = blockDim.x / 2; stride; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride]; __syncthreads(); }\n  float inv = rsqrtf(scratch[0] / C + eps); for (int64_t c = threadIdx.x; c < C; c += blockDim.x) { int64_t i = (b * C + c) * spatial + s; output[i] = __float2half_rn(__half2float(input[i]) * inv); }\n}\n''',
    "040": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb040_layernorm(half* output, const half* input, const half* weight, const half* bias, int64_t B, int64_t F, int64_t D1, int64_t D2) {\n  extern __shared__ float scratch[]; int64_t b = blockIdx.x, n = F * D1 * D2; float sum = 0.0f, square = 0.0f;\n  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float x = __half2float(input[b * n + i]); sum += x; square += x * x; }\n  scratch[threadIdx.x] = sum; scratch[blockDim.x + threadIdx.x] = square; __syncthreads();\n  for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) { scratch[threadIdx.x] += scratch[threadIdx.x + s]; scratch[blockDim.x + threadIdx.x] += scratch[blockDim.x + threadIdx.x + s]; } __syncthreads(); }\n  float mean = scratch[0] / n, inv = rsqrtf(scratch[blockDim.x] / n - mean * mean + 1e-5f);\n  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) output[b * n + i] = __float2half_rn((__half2float(input[b * n + i]) - mean) * inv * __half2float(weight[i]) + __half2float(bias[i]));\n}\n''',
    "047": b'''#include <cuda_fp16.h>\n#include <stdint.h>\nextern "C" __global__ void kb047_sum(half* output, const half* input, int64_t reduce_dim, int64_t B, int64_t D1, int64_t D2) {\n  int64_t out = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (out >= B * D2 || reduce_dim != 1) return; int64_t b = out / D2, c = out % D2; float sum = 0.0f;\n  for (int64_t r = 0; r < D1; ++r) sum += __half2float(input[(b * D1 + r) * D2 + c]); output[out] = __float2half_rn(sum);\n}\n''',
    "088": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb088_gelu(half* output, const half* input, int64_t B, int64_t D) {\n  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < B * D) { float x = __half2float(input[i]); float a = 0.7978845608028654f * (x + 0.044715f * x * x * x); output[i] = __float2half_rn(0.5f * x * (1.0f + tanhf(a))); }\n}\n''',
    "095": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb095_cross_entropy(half* loss, const half* predictions, const int64_t* targets, int64_t B, int64_t C) {\n  extern __shared__ float scratch[]; float local = 0.0f;\n  for (int64_t b = threadIdx.x; b < B; b += blockDim.x) { float maximum = -INFINITY; for (int64_t c = 0; c < C; ++c) maximum = fmaxf(maximum, __half2float(predictions[b * C + c])); float sum = 0.0f; for (int64_t c = 0; c < C; ++c) sum += expf(__half2float(predictions[b * C + c]) - maximum); local += logf(sum) + maximum - __half2float(predictions[b * C + targets[b]]); }\n  scratch[threadIdx.x] = local; __syncthreads(); for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s]; __syncthreads(); } if (threadIdx.x == 0) loss[0] = __float2half_rn(scratch[0] / B);\n}\n''',
}


_LAYER_NORM_BACKWARD = b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb040_layernorm_stats(unsigned char* workspace, const half* input, int64_t B, int64_t F, int64_t D1, int64_t D2) {\n+  float* mean = (float*)workspace; float* inv = mean + B; extern __shared__ float scratch[]; int64_t b = blockIdx.x, n = F * D1 * D2; float sum = 0.0f, square = 0.0f;\n+  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float x = __half2float(input[b*n+i]); sum += x; square += x*x; } scratch[threadIdx.x]=sum; scratch[blockDim.x+threadIdx.x]=square; __syncthreads();\n+  for (int s=blockDim.x/2;s;s>>=1) { if(threadIdx.x<s){scratch[threadIdx.x]+=scratch[threadIdx.x+s];scratch[blockDim.x+threadIdx.x]+=scratch[blockDim.x+threadIdx.x+s];} __syncthreads(); } if(threadIdx.x==0){mean[b]=scratch[0]/n;inv[b]=rsqrtf(scratch[blockDim.x]/n-mean[b]*mean[b]+1e-5f);}\n+}\n+extern "C" __global__ void kb040_layernorm_grad_input(half* grad_input, const half* input, const half* weight, const half* grad_output, unsigned char* workspace, int64_t B, int64_t F, int64_t D1, int64_t D2) {\n+  float* mean=(float*)workspace;float* inv=mean+B;extern __shared__ float scratch[];int64_t b=blockIdx.x,n=F*D1*D2;float a=0.0f,c=0.0f;for(int64_t i=threadIdx.x;i<n;i+=blockDim.x){float x=(__half2float(input[b*n+i])-mean[b])*inv[b];float g=__half2float(grad_output[b*n+i])*__half2float(weight[i]);a+=g;c+=g*x;}scratch[threadIdx.x]=a;scratch[blockDim.x+threadIdx.x]=c;__syncthreads();for(int s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s){scratch[threadIdx.x]+=scratch[threadIdx.x+s];scratch[blockDim.x+threadIdx.x]+=scratch[blockDim.x+threadIdx.x+s];}__syncthreads();}for(int64_t i=threadIdx.x;i<n;i+=blockDim.x){float x=(__half2float(input[b*n+i])-mean[b])*inv[b];float g=__half2float(grad_output[b*n+i])*__half2float(weight[i]);grad_input[b*n+i]=__float2half_rn((g-scratch[0]/n-x*scratch[blockDim.x]/n)*inv[b]);}\n+}\n+extern "C" __global__ void kb040_layernorm_grad_affine(half* grad_weight, half* grad_bias, const half* input, const half* grad_output, unsigned char* workspace, int64_t B, int64_t F, int64_t D1, int64_t D2) {\n+  float* mean=(float*)workspace;float* inv=mean+B;int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x,n=F*D1*D2;if(i>=n)return;float dw=0.0f,db=0.0f;for(int64_t b=0;b<B;++b){float g=__half2float(grad_output[b*n+i]);db+=g;dw+=g*(__half2float(input[b*n+i])-mean[b])*inv[b];}grad_weight[i]=__float2half_rn(dw);grad_bias[i]=__float2half_rn(db);\n+}\n+'''

# Strip patch-readable continuation markers from the embedded source above.
_LAYER_NORM_BACKWARD = _LAYER_NORM_BACKWARD.replace(b"\n+", b"\n")

_ELEMENTWISE_BACKWARD = {
    "019": b'''#include <cuda_fp16.h>\n#include <stdint.h>\nextern "C" __global__ void kb019_relu_backward(half* grad_input, const half* input, const half* grad_output, int64_t B, int64_t D) { int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<B*D) grad_input[i]=__float2half_rn(__half2float(input[i])>0.0f?__half2float(grad_output[i]):0.0f); }\n''',
    "026": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb026_gelu_backward(half* grad_input, const half* input, const half* grad_output, int64_t B, int64_t D) { int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<B*D){float x=__half2float(input[i]);float cdf=0.5f*(1.0f+erff(x*0.7071067811865476f));float pdf=0.3989422804014327f*expf(-0.5f*x*x);grad_input[i]=__float2half_rn(__half2float(grad_output[i])*(cdf+x*pdf));} }\n''',
    "088": b'''#include <cuda_fp16.h>\n#include <math.h>\n#include <stdint.h>\nextern "C" __global__ void kb088_mingpt_gelu_backward(half* grad_input, const half* input, const half* grad_output, int64_t B, int64_t D) { int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<B*D){float x=__half2float(input[i]);float u=0.7978845608028654f*(x+0.044715f*x*x*x);float t=tanhf(u);float derivative=0.5f*(1.0f+t)+0.5f*x*(1.0f-t*t)*0.7978845608028654f*(1.0f+3.0f*0.044715f*x*x);grad_input[i]=__float2half_rn(__half2float(grad_output[i])*derivative);} }\n''',
}


@dataclass(frozen=True)
class _Launch:
    name: str
    arguments: tuple[str, ...]
    grid: str
    block: str = "256"
    shared: str = "0"


_FORWARD_LAUNCHES = {
    "004": (_Launch("kb004_matvec", ("output", "A", "B", "M", "K"), "M", shared="1024"),),
    "007": (_Launch("kb007_matmul", ("output", "A", "B", "M", "N", "K"), "ceil_div(M * N, 256)"),),
    "019": (_Launch("kb019_relu", ("output", "input", "B", "D"), "ceil_div(B * D, 256)"),),
    "023": (_Launch("kb023_softmax", ("output", "input", "B", "D"), "B", shared="1024"),),
    "026": (_Launch("kb026_gelu", ("output", "input", "B", "D"), "ceil_div(B * D, 256)"),),
    "036": (_Launch("kb036_rmsnorm", ("output", "input", "eps", "B", "C", "D1", "D2"), "B * D1 * D2", shared="1024"),),
    "040": (_Launch("kb040_layernorm", ("output", "input", "weight", "bias", "B", "F", "D1", "D2"), "B", shared="2048"),),
    "047": (_Launch("kb047_sum", ("output", "input", "reduce_dim", "B", "D1", "D2"), "ceil_div(B * D2, 256)"),),
    "088": (_Launch("kb088_gelu", ("output", "input", "B", "D"), "ceil_div(B * D, 256)"),),
    "095": (_Launch("kb095_cross_entropy", ("loss", "predictions", "targets", "B", "C"), "1", shared="1024"),),
}

_BACKWARD_LAUNCHES = {
    "004": (_Launch("kb004_grad_a", ("grad_A", "grad_output", "B", "M", "K"), "ceil_div(M * K, 256)"), _Launch("kb004_grad_b", ("grad_B", "A", "grad_output", "M", "K"), "ceil_div(K, 256)")),
    "007": (_Launch("kb007_grad_a", ("grad_A", "grad_output", "B", "M", "N", "K"), "ceil_div(M * K, 256)"), _Launch("kb007_grad_b", ("grad_B", "A", "grad_output", "M", "N", "K"), "ceil_div(K * N, 256)")),
    "019": (_Launch("kb019_relu_backward", ("grad_input", "input", "grad_output", "B", "D"), "ceil_div(B * D, 256)"),),
    "023": (_Launch("kb023_softmax_backward", ("grad_input", "input", "grad_output", "B", "D"), "B", shared="1024"),),
    "026": (_Launch("kb026_gelu_backward", ("grad_input", "input", "grad_output", "B", "D"), "ceil_div(B * D, 256)"),),
    "036": (_Launch("kb036_rmsnorm_backward", ("grad_input", "input", "grad_output", "B", "C", "D1", "D2", "eps"), "B * D1 * D2", shared="2048"),),
    "040": (_Launch("kb040_layernorm_stats", ("workspace", "input", "B", "F", "D1", "D2"), "B", shared="2048"), _Launch("kb040_layernorm_grad_input", ("grad_input", "input", "weight", "grad_output", "workspace", "B", "F", "D1", "D2"), "B", shared="2048"), _Launch("kb040_layernorm_grad_affine", ("grad_weight", "grad_bias", "input", "grad_output", "workspace", "B", "F", "D1", "D2"), "ceil_div(F * D1 * D2, 256)")),
    "047": (_Launch("kb047_sum_backward", ("grad_input", "grad_output", "B", "D1", "D2"), "ceil_div(B * D1 * D2, 256)"),),
    "088": (_Launch("kb088_mingpt_gelu_backward", ("grad_input", "input", "grad_output", "B", "D"), "ceil_div(B * D, 256)"),),
    "095": (_Launch("kb095_cross_entropy_backward", ("grad_predictions", "predictions", "targets", "grad_loss", "B", "C"), "ceil_div(B, 256)"),),
}


def _number(task: TaskSpec) -> str:
    parts = task.id.split(".")
    if len(parts) < 4 or parts[:2] != ["kernelbench", "level1"]:
        raise BackendUnsupportedError("backend_unsupported")
    return parts[2]


def _backward_source(number: str) -> bytes:
    if number == "040":
        return _LAYER_NORM_BACKWARD
    if number in _ELEMENTWISE_BACKWARD:
        return _ELEMENTWISE_BACKWARD[number]
    root = Path(__file__).resolve().parents[3] / "portfolio" / "harness" / "core10" / "backward"
    matches = sorted(root.glob(f"{number}_*.cu"))
    if len(matches) != 1:
        raise FileNotFoundError(f"missing unique backward source for {number}")
    return matches[0].read_bytes()


def build_fixed_cuda_candidate(task: TaskSpec) -> bytes:
    """Build a deterministic device-only package for every Core 10 direction."""
    number = _number(task)
    if number not in _FORWARD or "cuda" not in task.candidate_backends:
        raise BackendUnsupportedError("backend_unsupported")
    definitions = (
        _FORWARD_LAUNCHES[number]
        if task.direction is Direction.FORWARD
        else _BACKWARD_LAUNCHES[number]
    )
    plan = CandidateLaunchPlan(
        task_spec_digest=task.canonical_sha256(),
        backend=CandidateBackend.CUDA,
        shape_symbols=tuple(item.name for item in task.dimensions),
        tensor_bindings=tuple(item.name for item in task.tensors),
        scalar_bindings=tuple(item.name for item in task.scalars),
        workspace_bytes=(128 if number == "040" and task.direction is Direction.BACKWARD else 0),
        kernels=tuple(
            KernelDeclaration(name=item.name, parameters=item.arguments)
            for item in definitions
        ),
        dispatch=(
            DispatchRule(
                launches=tuple(
                    KernelLaunch(
                        kernel=item.name,
                        grid=Dimensions(x=item.grid),
                        block=Dimensions(x=item.block),
                        dynamic_shared_bytes=item.shared,
                        arguments=item.arguments,
                    )
                    for item in definitions
                )
            ),
        ),
    )
    source = _FORWARD[number] if task.direction is Direction.FORWARD else _backward_source(number)
    return build_candidate_package(
        source,
        plan,
        task=task,
        provenance=CandidateProvenance(generator="kernelblaster-fixed-cuda-v1"),
    )


__all__ = ["build_fixed_cuda_candidate"]
