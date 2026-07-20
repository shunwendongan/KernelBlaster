#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cassert>
#include <cstdint>

void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int64_t M,
    int64_t N,
    int64_t K
) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        const cublasStatus_t created = cublasCreate(&handle);
        assert(created == CUBLAS_STATUS_SUCCESS);
        const cublasStatus_t math_mode = cublasSetMathMode(
            handle, CUBLAS_TENSOR_OP_MATH
        );
        assert(math_mode == CUBLAS_STATUS_SUCCESS);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        input_B,
        CUDA_R_16F,
        static_cast<int>(N),
        input_A,
        CUDA_R_16F,
        static_cast<int>(K),
        &beta,
        output,
        CUDA_R_16F,
        static_cast<int>(N),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    assert(status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();
}
