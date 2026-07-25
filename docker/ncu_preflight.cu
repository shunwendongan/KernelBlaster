// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

__global__ void kernelblaster_ncu_preflight(int* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = 1;
    }
}

int main() {
    int* output = nullptr;
    if (cudaMalloc(&output, sizeof(int)) != cudaSuccess) {
        return 1;
    }
    kernelblaster_ncu_preflight<<<1, 32>>>(output);
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    const cudaError_t free_status = cudaFree(output);
    return launch_status == cudaSuccess && sync_status == cudaSuccess &&
                   free_status == cudaSuccess
               ? 0
               : 1;
}
