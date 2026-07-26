#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb036_rmsnorm_backward(
    half* grad_input, const half* input, const half* grad_output,
    int64_t batch, int64_t channels, int64_t dim1, int64_t dim2, float eps
) {
    extern __shared__ float scratch[];
    const int64_t position = blockIdx.x;
    const int64_t spatial = dim1 * dim2;
    if (position >= batch * spatial) return;
    const int64_t b = position / spatial;
    const int64_t s = position % spatial;
    float square_sum = 0.0f;
    float dot_sum = 0.0f;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const int64_t index = (b * channels + c) * spatial + s;
        const float x = __half2float(input[index]);
        square_sum += x * x;
        dot_sum += x * __half2float(grad_output[index]);
    }
    scratch[threadIdx.x] = square_sum;
    scratch[blockDim.x + threadIdx.x] = dot_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            scratch[blockDim.x + threadIdx.x] += scratch[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / channels + eps);
    const float correction = scratch[blockDim.x] * inv * inv * inv / channels;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const int64_t index = (b * channels + c) * spatial + s;
        const float dx = __half2float(grad_output[index]) * inv
            - __half2float(input[index]) * correction;
        grad_input[index] = __float2half_rn(dx);
    }
}
