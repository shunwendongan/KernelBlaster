#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void kb019_relu_backward(
    half* grad_input, const half* input, const half* grad_output, int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        grad_input[index] = __hgt(input[index], __float2half(0.0f))
            ? grad_output[index] : __float2half(0.0f);
    }
}
