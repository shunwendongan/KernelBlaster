#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__ void kb088_mingpt_gelu_backward(
    half* grad_input, const half* input, const half* grad_output, int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(input[index]);
        const float a = 0.7978845608028654f;
        const float inner = a * (x + 0.044715f * x * x * x);
        const float t = tanhf(inner);
        const float derivative = 0.5f * (1.0f + t)
            + 0.5f * x * (1.0f - t * t) * a * (1.0f + 0.134145f * x * x);
        grad_input[index] = __float2half_rn(__half2float(grad_output[index]) * derivative);
    }
}
