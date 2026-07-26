#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb040_layernorm_stats(
    float* means, float* inverse_std, const half* input,
    int64_t batch, int64_t normalized, float eps
) {
    extern __shared__ float scratch[];
    const int64_t row = blockIdx.x;
    if (row >= batch) return;
    float sum = 0.0f;
    float square = 0.0f;
    for (int64_t i = threadIdx.x; i < normalized; i += blockDim.x) {
        const float x = __half2float(input[row * normalized + i]);
        sum += x;
        square += x * x;
    }
    scratch[threadIdx.x] = sum;
    scratch[blockDim.x + threadIdx.x] = square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            scratch[blockDim.x + threadIdx.x] += scratch[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = scratch[0] / normalized;
    means[row] = mean;
    inverse_std[row] = rsqrtf(scratch[blockDim.x] / normalized - mean * mean + eps);
}

extern "C" __global__ void kb040_layernorm_grad_input(
    half* grad_input, const half* input, const half* weight, const half* grad_output,
    const float* means, const float* inverse_std,
    int64_t batch, int64_t normalized
) {
    extern __shared__ float scratch[];
    const int64_t row = blockIdx.x;
    if (row >= batch) return;
    const float mean = means[row];
    const float inv = inverse_std[row];
    float dyw_sum = 0.0f;
    float dyw_xhat_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < normalized; i += blockDim.x) {
        const float xhat = (__half2float(input[row * normalized + i]) - mean) * inv;
        const float dyw = __half2float(grad_output[row * normalized + i]) * __half2float(weight[i]);
        dyw_sum += dyw;
        dyw_xhat_sum += dyw * xhat;
    }
    scratch[threadIdx.x] = dyw_sum;
    scratch[blockDim.x + threadIdx.x] = dyw_xhat_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            scratch[blockDim.x + threadIdx.x] += scratch[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean_dyw = scratch[0] / normalized;
    const float mean_dyw_xhat = scratch[blockDim.x] / normalized;
    for (int64_t i = threadIdx.x; i < normalized; i += blockDim.x) {
        const float xhat = (__half2float(input[row * normalized + i]) - mean) * inv;
        const float dyw = __half2float(grad_output[row * normalized + i]) * __half2float(weight[i]);
        grad_input[row * normalized + i] = __float2half_rn(
            inv * (dyw - mean_dyw - xhat * mean_dyw_xhat)
        );
    }
}

extern "C" __global__ void kb040_layernorm_grad_affine(
    half* grad_weight, half* grad_bias, const half* input, const half* grad_output,
    const float* means, const float* inverse_std,
    int64_t batch, int64_t normalized
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= normalized) return;
    float dw = 0.0f;
    float db = 0.0f;
    for (int64_t row = 0; row < batch; ++row) {
        const float mean = means[row];
        const float inv = inverse_std[row];
        const float dy = __half2float(grad_output[row * normalized + i]);
        dw += dy * (__half2float(input[row * normalized + i]) - mean) * inv;
        db += dy;
    }
    grad_weight[i] = __float2half_rn(dw);
    grad_bias[i] = __float2half_rn(db);
}
