#include <torch/extension.h>

/* Actual Tensor Core Function */
//torch::Tensor tensor_core_dgrad(
        //torch::Tensor& err_in, 
        //torch::Tensor& weight);

torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight);


#define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

/* Extension Interface */
// stride 1, padding 1, dilation 1, kernel 3x3
torch::Tensor int8_conv(torch::Tensor input, 
        torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv(input, weight);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_conv", &int8_conv, "int8 convolution forward Nvidia GPU tensor core");
}
