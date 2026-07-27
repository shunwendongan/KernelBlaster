#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

// RMSNorm over the channel dimension of a contiguous [B, C, D1, D2] tensor.
//
// The important mapping is "one thread -> one spatial position". For a fixed
// channel, adjacent lanes therefore load adjacent spatial values. The upstream
// implementation mapped a block to one spatial position, so adjacent lanes
// walked channels with a D1*D2 stride and then paid for a block reduction.
//
// This file is an sm_86/FP16 research candidate, not a shape-generic library
// kernel. See candidates.json for its exact layout, stream, and numerics contract.

// Even spatial sizes are represented as half2 pairs. One thread owns both values
// in a pair and accumulates in FP32, avoiding both a cross-thread reduction and
// FP16 accumulation error.
__global__ void rmsnorm_half2_rsqrt(
    half2* __restrict__ output,
    const half2* __restrict__ input,
    int64_t batch_size,
    int64_t channels,
    int64_t spatial_pairs,
    float eps
) {
    // Use int64_t before multiplication so large logical tensors do not truncate
    // an intermediate grid index to 32 bits.
    const int64_t linear_pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_pairs = batch_size * spatial_pairs;
    if (linear_pair >= total_pairs) {
        return;
    }
    const int64_t batch = linear_pair / spatial_pairs;
    const int64_t pair = linear_pair - batch * spatial_pairs;
    const int64_t base = batch * channels * spatial_pairs + pair;

    // Each loop iteration is contiguous across a warp because every lane keeps
    // its spatial pair fixed while all lanes advance through the same channel.
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float2 value = __half22float2(input[base + channel * spatial_pairs]);
        sum0 = fmaf(value.x, value.x, sum0);
        sum1 = fmaf(value.y, value.y, sum1);
    }

    // rsqrt + multiply avoids a separate sqrt and divide. Correctness still
    // follows the manifest's FP16-input/FP32-reference tolerances.
    const float inv0 = rsqrtf(sum0 / static_cast<float>(channels) + eps);
    const float inv1 = rsqrtf(sum1 / static_cast<float>(channels) + eps);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t index = base + channel * spatial_pairs;
        const float2 value = __half22float2(input[index]);
        output[index] = __floats2half2_rn(value.x * inv0, value.y * inv1);
    }
}

// When D1*D2 is odd, alternating channel bases are not guaranteed to remain
// four-byte aligned. The scalar path covers the entire tensor rather than using
// unsafe half2 loads plus a partial tail fix.
__global__ void rmsnorm_scalar_odd_rsqrt(
    half* __restrict__ output,
    const half* __restrict__ input,
    int64_t total,
    int64_t channels,
    int64_t spatial_size,
    float eps
) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t batch = linear / spatial_size;
    const int64_t spatial = linear - batch * spatial_size;
    const int64_t base = batch * channels * spatial_size + spatial;

    // Keep the same thread/data mapping as the vector path so the fallback
    // changes alignment behavior, not the mathematical decomposition.
    float sum = 0.0f;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float value = __half2float(input[base + channel * spatial_size]);
        sum = fmaf(value, value, sum);
    }
    const float inverse = rsqrtf(sum / static_cast<float>(channels) + eps);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t index = base + channel * spatial_size;
        output[index] = __float2half_rn(__half2float(input[index]) * inverse);
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2,
    float eps
) {
    constexpr int threads = 256;
    const int64_t spatial_size = dim1 * dim2;

    // Select half2 only when every channel row has an even number of scalar
    // elements. This makes all channel starts and pair indices naturally aligned.
    if ((spatial_size & 1) == 0) {
        const int64_t spatial_pairs = spatial_size / 2;
        const int64_t total_pairs = batch_size * spatial_pairs;
        rmsnorm_half2_rsqrt<<<
            static_cast<int>((total_pairs + threads - 1) / threads), threads
        >>>(
            static_cast<half2*>(output),
            static_cast<const half2*>(input),
            batch_size,
            num_features,
            spatial_pairs,
            eps
        );
    } else {
        const int64_t total = batch_size * spatial_size;
        rmsnorm_scalar_odd_rsqrt<<<
            static_cast<int>((total + threads - 1) / threads), threads
        >>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            total,
            num_features,
            spatial_size,
            eps
        );
    }

    // The historical KernelBench driver expects a synchronous host entry. A
    // reusable library operator would normally launch on a caller-provided stream
    // and avoid device-wide synchronization; do not remove this without updating
    // the driver and timing contract together.
    cudaDeviceSynchronize();
}
