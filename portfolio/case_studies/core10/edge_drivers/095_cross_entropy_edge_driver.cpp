// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t);

static bool run_case(int64_t batch, int64_t classes, float& max_error, float& p99_error,
                     bool& all_finite, bool& all_deterministic) {
    auto half_options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto targets_options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    auto predictions = torch::randn({batch, classes}, half_options);
    auto targets = torch::randint(0, classes, {batch}, targets_options);
    auto reference = torch::nn::functional::cross_entropy(predictions, targets);
    auto output = torch::empty({1}, half_options);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(output.data_ptr(), predictions.data_ptr(), targets.data_ptr(), batch, classes);
        if (repeat == 0) first = output.clone();
        else deterministic = deterministic && torch::equal(first, output);
    }
    auto error = (output.to(torch::kFloat32) - reference.to(torch::kFloat32)).abs().cpu();
    const float value = error.max().item<float>();
    max_error = std::max(max_error, value);
    p99_error = std::max(p99_error, value);
    const bool finite = torch::isfinite(output).all().item<bool>();
    all_finite = all_finite && finite;
    all_deterministic = all_deterministic && deterministic;
    return deterministic && finite
        && torch::allclose(output.to(torch::kFloat32), reference.to(torch::kFloat32), 1e-1, 1e-1);
}

int main() {
    torch::manual_seed(20260721);
    float max_error = 0.0f, p99_error = 0.0f;
    bool finite = true, deterministic = true;
    bool passed = true;
    // The upstream implementation is only defined for classes <= 10. Keep the
    // formal head-to-head edge set inside that common domain.
    for (auto [batch, classes] : {std::pair<int64_t, int64_t>{1, 2}, {17, 7}, {65, 9}, {257, 10}})
        passed = run_case(batch, classes, max_error, p99_error, finite, deterministic) && passed;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON {\"max_abs_error\":" << max_error
              << ",\"p99_abs_error\":" << p99_error
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"deterministic\":" << (deterministic ? "true" : "false") << "}" << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
