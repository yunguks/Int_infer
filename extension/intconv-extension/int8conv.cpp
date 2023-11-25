#include <torch/extension.h>

/* Actual Tensor Core Function */
torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation);

torch::Tensor tensor_core_float_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

/* Extension Interface */

torch::Tensor int8_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv(input, weight, stride, padding, dilation);
}


torch::Tensor float_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_float_conv(input, weight, stride, padding, dilation);
}


void find_algo(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag);

void conv_find_algo(
        torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return find_algo(input, weight, stride, padding, dilation, float_flag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_conv", &int8_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("float_conv", &float_conv, "float convolution forward Nvidia GPU tensor core");
  m.def("conv_find_algo", &conv_find_algo, "find the convolution forward algorithm");
}