#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

struct Statistics {
    float mean;
    float inverse_std;
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : value;
}

__global__ void layernorm_partials_kernel(
    float2* __restrict__ partials,
    const half* __restrict__ input,
    int64_t norm_size,
    int tiles_per_batch
) {
    const int batch = blockIdx.x / tiles_per_batch;
    const int tile = blockIdx.x - batch * tiles_per_batch;
    const int64_t base = static_cast<int64_t>(batch) * norm_size;
    const int64_t stride = static_cast<int64_t>(tiles_per_batch) * blockDim.x;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int64_t index = static_cast<int64_t>(tile) * blockDim.x + threadIdx.x;
         index < norm_size;
         index += stride) {
        const float value = __half2float(input[base + index]);
        sum += value;
        square_sum = fmaf(value, value, square_sum);
    }
    __shared__ float shared_sum[8];
    __shared__ float shared_square[8];
    const float total = block_sum(sum, shared_sum);
    __syncthreads();
    const float total_square = block_sum(square_sum, shared_square);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = make_float2(total, total_square);
    }
}

__global__ void layernorm_statistics_kernel(
    Statistics* __restrict__ statistics,
    const float2* __restrict__ partials,
    int64_t norm_size,
    int tiles_per_batch
) {
    const int batch = blockIdx.x;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int tile = threadIdx.x; tile < tiles_per_batch; tile += blockDim.x) {
        const float2 value = partials[batch * tiles_per_batch + tile];
        sum += value.x;
        square_sum += value.y;
    }
    __shared__ float shared_sum[8];
    __shared__ float shared_square[8];
    const float total = block_sum(sum, shared_sum);
    __syncthreads();
    const float total_square = block_sum(square_sum, shared_square);
    if (threadIdx.x == 0) {
        const float mean = total / static_cast<float>(norm_size);
        const float variance = fmaxf(
            0.0f, total_square / static_cast<float>(norm_size) - mean * mean
        );
        statistics[batch] = {mean, rsqrtf(variance + 1e-5f)};
    }
}

__global__ void layernorm_apply_kernel(
    half2* __restrict__ output,
    const half2* __restrict__ input,
    const half2* __restrict__ weight,
    const half2* __restrict__ bias,
    const Statistics* __restrict__ statistics,
    int64_t total_pairs,
    int64_t norm_pairs
) {
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < total_pairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int batch = static_cast<int>(pair / norm_pairs);
        const int64_t feature_pair = pair - static_cast<int64_t>(batch) * norm_pairs;
        const float2 value = __half22float2(input[pair]);
        const float2 scale = __half22float2(weight[feature_pair]);
        const float2 offset = __half22float2(bias[feature_pair]);
        const Statistics stats = statistics[batch];
        output[pair] = __floats2half2_rn(
            (value.x - stats.mean) * stats.inverse_std * scale.x + offset.x,
            (value.y - stats.mean) * stats.inverse_std * scale.y + offset.y
        );
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int64_t batch_size,
    int64_t features,
    int64_t dim1,
    int64_t dim2
) {
    constexpr int threads = 256;
    constexpr int tiles_per_batch = 256;
    static float2* partials = nullptr;
    static Statistics* statistics = nullptr;
    static int64_t partial_capacity = 0;
    if (batch_size * tiles_per_batch > partial_capacity) {
        if (partials != nullptr) {
            cudaFree(partials);
            cudaFree(statistics);
        }
        cudaMalloc(&partials, batch_size * tiles_per_batch * sizeof(float2));
        cudaMalloc(&statistics, batch_size * sizeof(Statistics));
        partial_capacity = batch_size * tiles_per_batch;
    }

    const int64_t norm_size = features * dim1 * dim2;
    layernorm_partials_kernel<<<static_cast<int>(batch_size) * tiles_per_batch, threads>>>(
        partials,
        static_cast<const half*>(input),
        norm_size,
        tiles_per_batch
    );
    layernorm_statistics_kernel<<<static_cast<int>(batch_size), threads>>>(
        statistics,
        partials,
        norm_size,
        tiles_per_batch
    );
    const int64_t norm_pairs = norm_size / 2;
    const int64_t total_pairs = batch_size * norm_pairs;
    const int blocks = static_cast<int>(
        std::min<int64_t>(4096, (total_pairs + threads - 1) / threads)
    );
    layernorm_apply_kernel<<<blocks, threads>>>(
        static_cast<half2*>(output),
        static_cast<const half2*>(input),
        static_cast<const half2*>(weight),
        static_cast<const half2*>(bias),
        statistics,
        total_pairs,
        norm_pairs
    );
    cudaDeviceSynchronize();
}
