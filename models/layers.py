import int8mm_cuda
import int8pool_cuda
import int8conv_cuda
from torch.nn.modules import Module
import torch

class IntLinear(Module):
    def __init__(self, in_channels, out_channels):
        super(IntLinear,self).__init__()
        self.weight = torch.randint(-128,127,(out_channels, in_channels), dtype=torch.int8)

    def forward(self,x):
        # weight [OUT, IN} - > [IN, OUT]
        # input [BATCH, IN]
        y = int8mm_cuda.int8_mm(x,self.weight.transpose(1,0).contiguous())
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()


class IntPool(Module):
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
    

class IntConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =1, padding =1):
        super(IntConv2d,self).__init__()
        self.weight = torch.randint(-128,127,(out_channels, kernel_size, kernel_size, in_channels), dtype=torch.int8)
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        y = int8conv_cuda.cu_int8_conv(x,self.weight,self.stride, self.padding,1)
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()


class QuantReLU(Module):
    def __init__(self):
        super(QuantReLU,self).__init__()
    
    def forward(self,x):
        x = x + 2**15
        x = torch.clamp(x, min=0, max=2**16-1)
        x = torch.sqrt(x)
        x = x-128
        return x.type(torch.int8)
    

class FLOATLinear(Module):
    def __init__(self, in_channels, out_channels):
        super(FLOATLinear,self).__init__()
        self.weight = torch.randint(-128,127,(out_channels, in_channels), dtype=torch.int8)

    def forward(self,x):
        # weight [OUT, IN} - > [IN, OUT]
        # input [BATCH, IN]
        y = int8mm_cuda.int8_mm(x,self.weight.transpose(1,0).contiguous())
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()


class FLOATPool(Module):
    # mode 0 -> max pooling , mode 1 -> average pooling
    def __init__(self,kernel_size = 2, stride = 2, padding=0, mode=0):
        super(FLOATPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
    
    def forward(self,x):
        y = int8pool_cuda.int8_pool(x,self.kernel_size, self.stride, self.padding, self.mode)
        return y
    

class FLOATConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =1, padding =1):
        super(FLOATConv2d,self).__init__()
        self.weight = torch.randint(-128,127,(out_channels, kernel_size, kernel_size, in_channels), dtype=torch.int8)
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        y = int8conv_cuda.cu_int8_conv(x,self.weight,self.stride, self.padding,1)
        return y
    
    def cuda(self):
        self.weight = self.weight.cuda()


class QuantReLU(Module):
    def __init__(self):
        super(QuantReLU,self).__init__()
    
    def forward(self,x):
        x = x + 2**15
        x = torch.clamp(x, min=0, max=2**16-1)
        x = torch.sqrt(x)
        x = x-128
        return x.type(torch.int8)