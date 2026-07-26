#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__ void kb026_gelu_backward(
    half* grad_input, const half* input, const half* grad_output, int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(input[index]);
        const float derivative = 0.5f * (1.0f + erff(x * CUDART_SQRT_HALF_F))
            + x * expf(-0.5f * x * x) * 0.3989422804014327f;
        grad_input[index] = __float2half_rn(__half2float(grad_output[index]) * derivative);
    }
}
