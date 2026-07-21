// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0
/* Additional correctness coverage for the Day 8-10 RMSNorm case study. */
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
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
    bool finite = true;
    bool deterministic = true;
    float max_error = 0.0f;
    float p99_error = 0.0f;
    for (const auto& [batch, channels, dim1, dim2] : shapes) {
        constexpr float eps = 1e-5f;
        torch::Tensor input = torch::randn(
            {batch, channels, dim1, dim2}, options
        );
        torch::Tensor reference = input / (input.pow(2).mean(1, true) + eps).sqrt();
        torch::Tensor output = torch::empty_like(input);

        torch::Tensor first;
        cudaError_t error = cudaSuccess;
        for (int repeat = 0; repeat < 5; ++repeat) {
            launch_gpu_implementation(
                output.data_ptr(),
                input.data_ptr(),
                batch,
                channels,
                dim1,
                dim2,
                eps
            );
            error = cudaDeviceSynchronize();
            if (repeat == 0) {
                first = output.clone();
            } else {
                deterministic = deterministic && torch::equal(first, output);
            }
        }
        finite = finite && torch::isfinite(output).all().item<bool>();
        auto absolute_error = (
            output.to(torch::kFloat32) - reference.to(torch::kFloat32)
        ).abs().flatten().cpu();
        auto sorted = std::get<0>(absolute_error.sort());
        const int64_t p99_index = static_cast<int64_t>(
            std::ceil(0.99 * (sorted.numel() - 1))
        );
        max_error = std::max(max_error, absolute_error.max().item<float>());
        p99_error = std::max(p99_error, sorted[p99_index].item<float>());
        const bool current = error == cudaSuccess && finite && deterministic &&
            torch::allclose(output, reference, 1e-1, 1e-1);
        if (!current) {
            std::cerr << "failed shape B=" << batch << " C=" << channels
                      << " D1=" << dim1 << " D2=" << dim2
                      << " CUDA=" << cudaGetErrorString(error) << std::endl;
        }
        passed = passed && current;
    }

    std::cout << "KERNELBLASTER_CORRECTNESS_JSON {\"max_abs_error\":" << max_error
              << ",\"p99_abs_error\":" << p99_error
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"deterministic\":" << (deterministic ? "true" : "false")
              << "}" << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
