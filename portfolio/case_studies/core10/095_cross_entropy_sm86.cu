#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cfloat>
#include <cstdint>

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void cross_entropy_partials_kernel(
    float* __restrict__ partials,
    const half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    int batch_size,
    int classes
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sample = blockIdx.x * 8 + warp;
    float loss = 0.0f;
    if (sample < batch_size) {
        float value = lane < classes
            ? __half2float(logits[static_cast<int64_t>(sample) * classes + lane])
            : -FLT_MAX;
        float maximum = warp_max(value);
        maximum = __shfl_sync(0xffffffff, maximum, 0);
        float exponential = lane < classes ? __expf(value - maximum) : 0.0f;
        float sum = warp_sum(exponential);
        sum = __shfl_sync(0xffffffff, sum, 0);
        const int target = static_cast<int>(targets[sample]);
        const float target_logit = __shfl_sync(0xffffffff, value, target);
        if (lane == 0) {
            loss = maximum + logf(sum) - target_logit;
        }
    }
    __shared__ float warp_losses[8];
    if (lane == 0) {
        warp_losses[warp] = loss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            total += warp_losses[index];
        }
        partials[blockIdx.x] = total;
    }
}

__global__ void cross_entropy_finalize_kernel(
    half* output,
    const float* partials,
    int partial_count,
    int batch_size
) {
    float sum = 0.0f;
    for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
        sum += partials[index];
    }
    sum = warp_sum(sum);
    __shared__ float warp_totals[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            output[0] = __float2half_rn(total / static_cast<float>(batch_size));
        }
    }
}

void launch_gpu_implementation(
    void* output,
    void* predictions,
    void* targets,
    int64_t batch_size,
    int64_t num_classes
) {
    assert(num_classes > 0 && num_classes <= 32);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((batch_size + 7) / 8);
    static float* partials = nullptr;
    static int capacity = 0;
    if (blocks > capacity) {
        if (partials != nullptr) {
            cudaFree(partials);
        }
        cudaMalloc(&partials, blocks * sizeof(float));
        capacity = blocks;
    }
    cross_entropy_partials_kernel<<<blocks, threads>>>(
        partials,
        static_cast<const half*>(predictions),
        static_cast<const int64_t*>(targets),
        static_cast<int>(batch_size),
        static_cast<int>(num_classes)
    );
    cross_entropy_finalize_kernel<<<1, threads>>>(
        static_cast<half*>(output),
        partials,
        blocks,
        static_cast<int>(batch_size)
    );
    cudaDeviceSynchronize();
}
