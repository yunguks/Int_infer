#include <torch/extension.h>

/* Actual Tensor Core Function */
torch::Tensor tensor_core_int8_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding,
        int32_t mode);

torch::Tensor tensor_core_float_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding,
        int32_t mode);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Extension Interface */
torch::Tensor int8_pool(
        torch::Tensor input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding,
        int32_t mode){
    CHECK_INPUT(input);
    return tensor_core_int8_pool(input, kernel_size, stride, padding, mode);
}


torch::Tensor float_pool(
        torch::Tensor input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding,
        int32_t mode){
    CHECK_INPUT(input);
    return tensor_core_float_pool(input, kernel_size, stride, padding, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_pool", &int8_pool, "int8 max pooling forward Nvidia GPU tensor core");
  m.def("float_pool", &float_pool, "int8 max pooling forward Nvidia GPU tensor core");
}