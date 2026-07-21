// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, void*, int64_t, int64_t, int64_t, int64_t);

static bool run_case(int64_t batch, int64_t features, int64_t dim1, int64_t dim2,
                     float& max_error, float& p99_error,
                     bool& all_finite, bool& all_deterministic) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    std::vector<int64_t> normalized{features, dim1, dim2};
    auto input = torch::randn({batch, features, dim1, dim2}, options);
    auto weight = torch::randn(normalized, options);
    auto bias = torch::randn(normalized, options);
    auto reference = torch::layer_norm(input, normalized, weight, bias, 1e-5, true);
    auto output = torch::empty_like(reference);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(output.data_ptr(), input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                                  batch, features, dim1, dim2);
        if (repeat == 0) first = output.clone();
        else deterministic = deterministic && torch::equal(first, output);
    }
    auto error = (output.to(torch::kFloat32) - reference.to(torch::kFloat32)).abs().flatten().cpu();
    auto sorted = std::get<0>(error.sort());
    const int64_t index = static_cast<int64_t>(std::ceil(0.99 * (sorted.numel() - 1)));
    max_error = std::max(max_error, error.max().item<float>());
    p99_error = std::max(p99_error, sorted[index].item<float>());
    const bool finite = torch::isfinite(output).all().item<bool>();
    all_finite = all_finite && finite;
    all_deterministic = all_deterministic && deterministic;
    return deterministic && finite
        && torch::allclose(output, reference, 1e-1, 1e-1);
}

int main() {
    torch::manual_seed(20260721);
    float max_error = 0.0f, p99_error = 0.0f;
    bool finite = true, deterministic = true;
    bool passed = true;
    for (auto shape : {std::tuple<int64_t, int64_t, int64_t, int64_t>{1, 3, 5, 7}, {2, 7, 9, 11}, {3, 16, 17, 18}})
        passed = std::apply([&](auto... dims) { return run_case(dims..., max_error, p99_error, finite, deterministic); }, shape) && passed;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON {\"max_abs_error\":" << max_error
              << ",\"p99_abs_error\":" << p99_error
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"deterministic\":" << (deterministic ? "true" : "false") << "}" << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
