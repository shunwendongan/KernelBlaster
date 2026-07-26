#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void kb095_cross_entropy_backward(
    half* grad_predictions, const half* predictions, const int64_t* targets,
    const half* grad_loss, int64_t batch, int64_t classes
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    float maximum = -FLT_MAX;
    for (int64_t column = 0; column < classes; ++column) {
        maximum = fmaxf(maximum, __half2float(predictions[row * classes + column]));
    }
    float denominator = 0.0f;
    for (int64_t column = 0; column < classes; ++column) {
        denominator += expf(__half2float(predictions[row * classes + column]) - maximum);
    }
    const float scale = __half2float(grad_loss[0]) / static_cast<float>(batch);
    for (int64_t column = 0; column < classes; ++column) {
        float value = expf(__half2float(predictions[row * classes + column]) - maximum)
            / denominator;
        if (column == targets[row]) value -= 1.0f;
        grad_predictions[row * classes + column] = __float2half_rn(value * scale);
    }
}
