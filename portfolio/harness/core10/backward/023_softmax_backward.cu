#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void kb023_softmax_backward(
    half* grad_input, const half* input, const half* grad_output,
    int64_t batch, int64_t dim
) {
    extern __shared__ float scratch[];
    const int64_t row = blockIdx.x;
    if (row >= batch) return;
    float local_max = -FLT_MAX;
    for (int64_t column = threadIdx.x; column < dim; column += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(input[row * dim + column]));
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0f;
    for (int64_t column = threadIdx.x; column < dim; column += blockDim.x) {
        local_sum += expf(__half2float(input[row * dim + column]) - maximum);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = scratch[0];
    float local_dot = 0.0f;
    for (int64_t column = threadIdx.x; column < dim; column += blockDim.x) {
        const float y = expf(__half2float(input[row * dim + column]) - maximum) / denominator;
        local_dot += y * __half2float(grad_output[row * dim + column]);
    }
    scratch[threadIdx.x] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float dot = scratch[0];
    for (int64_t column = threadIdx.x; column < dim; column += blockDim.x) {
        const int64_t index = row * dim + column;
        const float y = expf(__half2float(input[index]) - maximum) / denominator;
        grad_input[index] = __float2half_rn(y * (__half2float(grad_output[index]) - dot));
    }
}
