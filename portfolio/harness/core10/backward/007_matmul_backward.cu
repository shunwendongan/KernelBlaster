#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb007_grad_a(
    half* grad_a, const half* grad_c, const half* b,
    int64_t m, int64_t n, int64_t k
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < m * k) {
        const int64_t row = index / k;
        const int64_t inner = index % k;
        float sum = 0.0f;
        for (int64_t column = 0; column < n; ++column) {
            sum += __half2float(grad_c[row * n + column])
                * __half2float(b[inner * n + column]);
        }
        grad_a[index] = __float2half_rn(sum);
    }
}

extern "C" __global__ void kb007_grad_b(
    half* grad_b, const half* a, const half* grad_c,
    int64_t m, int64_t n, int64_t k
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < k * n) {
        const int64_t inner = index / n;
        const int64_t column = index % n;
        float sum = 0.0f;
        for (int64_t row = 0; row < m; ++row) {
            sum += __half2float(a[row * k + inner])
                * __half2float(grad_c[row * n + column]);
        }
        grad_b[index] = __float2half_rn(sum);
    }
}
