import int8mm_cuda
import int8pool_cuda
import int8conv_cuda
import torch.nn as nn
import torch
from torch.ao.quantization.observer import HistogramObserver

class IntLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(IntLinear,self).__init__()
        self.weight = torch.randint(-128,127,(in_channels, out_channels), dtype=torch.int8)
        self.b = bias
        self.bias = None
        if self.b:
            self.bias = torch.zeros((out_channels),dtype=torch.int8)

    def forward(self,x):
        # weight [OUT, IN} - > [IN, OUT]
        # input [BATCH, IN]
        x = x.contiguous()
        y = int8mm_cuda.int8_mm(x,self.weight)
        if self.b:
            y = y+self.bias
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()
        if self.b:
            self.bias = self.bias.cuda()


class IntPool(nn.Module):
    # mode 0 -> max pooling , mode 1 -> average pooling
    def __init__(self,kernel_size = 2, stride = 2, padding=0, mode=0):
        super(IntPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
    
    def forward(self,x):
        y = int8pool_cuda.int8_pool(x,self.kernel_size, self.stride, self.padding, self.mode)
        return y
    

class IntConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =1, padding =1, bias=True):
        super(IntConv2d,self).__init__()
        self.weight = torch.randint(-128,127,(out_channels, kernel_size, kernel_size, in_channels), dtype=torch.int8)
        self.stride = stride
        self.padding = padding
        self.b = bias
        self.bias = None
        if self.b:
            self.bias = torch.zeros((out_channels),dtype=torch.int8)

    def forward(self,x):
        y = int8conv_cuda.int8_conv(x,self.weight,self.stride, self.padding,1)
        if self.b:
            y = y+self.bias
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()
        if self.b:
            self.bias = self.bias.cuda()

class FLOATLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(FLOATLinear,self).__init__()
        self.weight = torch.rand((in_channels, out_channels))
        self.b = bias
        self.bias = None
        if self.b:
            self.bias = torch.zeros((out_channels),dtype=torch.int8)
        

    def forward(self,x):
        # weight [OUT, IN} - > [IN, OUT]
        # input [BATCH, IN]
        x = x.contiguous()
        y = int8mm_cuda.float_mm(x,self.weight)
        if self.b:
            y = y+self.bias
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()
        if self.b:
            self.bias = self.bias.cuda()


class FLOATPool(nn.Module):
    # mode 0 -> max pooling , mode 1 -> average pooling
    def __init__(self,kernel_size = 2, stride = 2, padding=0, mode=0):
        super(FLOATPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
    
    def forward(self,x):
        y = int8pool_cuda.float_pool(x,self.kernel_size, self.stride, self.padding, self.mode)
        return y
    

class FLOATConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =1, padding =1, bias=True):
        super(FLOATConv2d,self).__init__()
        self.weight = torch.rand((out_channels, kernel_size, kernel_size, in_channels))
        self.stride = stride
        self.padding = padding
        self.b = bias
        self.bias = None
        if self.b:
            self.bias = torch.zeros((out_channels),dtype=torch.float)

    def forward(self,x):
        y = int8conv_cuda.float_conv(x,self.weight,self.stride, self.padding,1)
        if self.b:
            y = y+self.bias
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()
        if self.b:
            self.bias = self.bias.cuda()

class QuantReLU(nn.Module):
    def __init__(self,int_dim=1):
        super(QuantReLU,self).__init__()
        self.int_dim = int_dim

    # def forward(self,x):
    #     # x = x + 2**15

    #     x = torch.sqrt(torch.clamp(x, min=0, max=2**16-1))
    #     # x = x-128
    #     return x.type(torch.int8)
    def forward(self,x):
        activation_observer = HistogramObserver(quant_max=127, quant_min=-128, reduce_range=True).to(x.device)
        quant_data = activation_observer(x)
        scale, zero_tensor = activation_observer.calculate_qparams()
        scale = scale.to(x.device)
        zero_tensor = zero_tensor.to(x.device)
        x = torch.tensor(x/scale + zero_tensor, dtype=torch.int8)
        return x
    # def forward(self,x):
    #     x = torch.sqrt(torch.clamp(x+2**15, min=0, max=2**16-1))-128
    #     return x.type(torch.int8)


    