#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb004_grad_a(
    half* grad_a, const half* grad_output, const half* b, int64_t m, int64_t k
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = m * k;
    if (index < count) {
        grad_a[index] = __float2half_rn(
            __half2float(grad_output[index / k]) * __half2float(b[index % k])
        );
    }
}

extern "C" __global__ void kb004_grad_b(
    half* grad_b, const half* a, const half* grad_output, int64_t m, int64_t k
) {
    const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column < k) {
        float sum = 0.0f;
        for (int64_t row = 0; row < m; ++row) {
            sum += __half2float(a[row * k + column]) * __half2float(grad_output[row]);
        }
        grad_b[column] = __float2half_rn(sum);
    }
}
