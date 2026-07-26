#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb047_sum_backward(
    half* grad_input, const half* grad_output,
    int64_t batch, int64_t dim1, int64_t dim2
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = batch * dim1 * dim2;
    if (index < count) {
        const int64_t b = index / (dim1 * dim2);
        const int64_t column = index % dim2;
        grad_input[index] = grad_output[b * dim2 + column];
    }
}
