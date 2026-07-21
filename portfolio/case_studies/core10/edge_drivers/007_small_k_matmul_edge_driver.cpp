// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t, int64_t);

static bool run_case(int64_t m, int64_t n, int64_t k, float& max_error, float& p99_error,
                     bool& all_finite, bool& all_deterministic) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto a = torch::randn({m, k}, options);
    auto b = torch::randn({k, n}, options);
    auto reference = torch::matmul(a, b);
    auto output = torch::empty({m, n}, options);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(output.data_ptr(), a.data_ptr(), b.data_ptr(), m, n, k);
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
    for (auto shape : {std::tuple<int64_t, int64_t, int64_t>{1, 1, 1}, {17, 19, 7}, {65, 33, 31}})
        passed = std::apply([&](auto... dims) { return run_case(dims..., max_error, p99_error, finite, deterministic); }, shape) && passed;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON {\"max_abs_error\":" << max_error
              << ",\"p99_abs_error\":" << p99_error
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"deterministic\":" << (deterministic ? "true" : "false") << "}" << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
