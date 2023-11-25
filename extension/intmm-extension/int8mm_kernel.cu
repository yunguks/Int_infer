#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
// #include <helper_cuda.h>
#include <ATen/cuda/CUDAContext.h>

// tensor_core_int8_mm 으로 작성하면 tensor_core가 무조껀 사용됨
// int8_mm tensor_core 가 사용할 수 있는 환경이라면 사용됨
torch::Tensor tensor_core_int8_mm(torch::Tensor lhs,torch::Tensor rhs)
{
    int32_t alpha = 1;
    int32_t beta = 0;
    /* only support m,n,k multiply of 4 */
    int m = lhs.size(0);
    int k = lhs.size(1);
    int n = rhs.size(1);

    int lda = k;
    int ldb = n;
    int ldc = m;
    // create the result tensor in a transposed way
    auto results=torch::empty({n,m},torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Pytorch is row major, cublas is column major
    // need to use TT version gemm 
    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,
            m,n,k,&alpha,
            lhs.data<int8_t>(),CUDA_R_8I,lda,
            rhs.data<int8_t>(),CUDA_R_8I,ldb,
            &beta,results.data<int32_t>(),
            CUDA_R_32I,ldc,CUDA_R_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // need to tranpose it for pytorch usage
    return results.transpose(0,1);
}


torch::Tensor tensor_core_float_mm(torch::Tensor lhs,torch::Tensor rhs)
{
    float alpha = 1.;
    float beta = 0.;
    /* only support m,n,k multiply of 4 */
    int m = lhs.size(0);
    int k = lhs.size(1);
    int n = rhs.size(1);

    int lda = k;
    int ldb = n;
    int ldc = m;
    // create the result tensor in a transposed way
    auto results=torch::empty({n,m},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Pytorch is row major, cublas is column major
    // need to use TT version gemm 
    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,
            m,n,k,&alpha,
            lhs.data<float>(),CUDA_R_32F,lda,
            rhs.data<float>(),CUDA_R_32F,ldb,
            &beta,results.data<float>(),
            CUDA_R_32F,ldc,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // need to tranpose it for pytorch usage
    return results.transpose(0,1);
}

// torch::Tensor tensor_core_int8_mm_no_trans(torch::Tensor lhs,torch::Tensor rhs)
// {
//     int32_t alpha = 1;
//     int32_t beta = 0;
//     /* only support n,k multiply of 4 */
//     int m = lhs.size(0);
//     int k = lhs.size(1);
//     int n = rhs.size(1);

//     int lda = k;
//     int ldb = n;

//     auto results=torch::empty({m,n},torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
//     int ldc = n;
    
//     cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
//     // Pytorch is row major, cublas is column major
//     // need to use Bt * At = Ct gemm 
//     cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
//             n,m,k,&alpha,
//             rhs.data<int8_t>(),CUDA_R_8I,lda,
//             lhs.data<int8_t>(),CUDA_R_8I,ldb,
//             &beta,results.data<int32_t>(),
//             CUDA_R_32I,ldc,CUDA_R_32I, 
//             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

//     return results;
// }