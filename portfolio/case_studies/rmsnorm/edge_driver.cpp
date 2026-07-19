/* Additional correctness coverage for the Day 8-10 RMSNorm case study. */
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "cuda_model.cuh"

void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2,
    float eps
);

int main() {
    torch::manual_seed(20260719);
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat16)
                             .device(torch::kCUDA);
    const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> shapes = {
        {1, 4, 1, 3},
        {1, 63, 1, 7},
        {2, 64, 3, 5},
        {1, 65, 1, 17},
        {2, 65, 17, 19},
    };

    bool passed = true;
    for (const auto& [batch, channels, dim1, dim2] : shapes) {
        constexpr float eps = 1e-5f;
        torch::Tensor input = torch::randn(
            {batch, channels, dim1, dim2}, options
        );
        torch::Tensor reference = input / (input.pow(2).mean(1, true) + eps).sqrt();
        torch::Tensor output = torch::empty_like(input);

        launch_gpu_implementation(
            output.data_ptr(),
            input.data_ptr(),
            batch,
            channels,
            dim1,
            dim2,
            eps
        );
        cudaError_t error = cudaDeviceSynchronize();
        const bool current = error == cudaSuccess &&
            torch::allclose(output, reference, 1e-1, 1e-1);
        if (!current) {
            std::cerr << "failed shape B=" << batch << " C=" << channels
                      << " D1=" << dim1 << " D2=" << dim2
                      << " CUDA=" << cudaGetErrorString(error) << std::endl;
        }
        passed = passed && current;
    }

    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
