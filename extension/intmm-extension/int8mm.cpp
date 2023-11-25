#include <torch/extension.h>

torch::Tensor tensor_core_int8_mm(torch::Tensor lhs,torch::Tensor rhs);
torch::Tensor tensor_core_float_mm(torch::Tensor lhs,torch::Tensor rhs);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor int8_mm(torch::Tensor lhs,torch::Tensor rhs) 
{
    CHECK_INPUT(lhs);
    CHECK_INPUT(rhs);
    return tensor_core_int8_mm(lhs, rhs);
}

torch::Tensor float_mm(torch::Tensor lhs,torch::Tensor rhs) 
{
    CHECK_INPUT(lhs);
    CHECK_INPUT(rhs);
    return tensor_core_float_mm(lhs, rhs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_mm", &int8_mm, "int8 matrix multiply using Nvidia GPU");
  m.def("float_mm", &float_mm, "float matrix multiply using Nvidia GPU");
}