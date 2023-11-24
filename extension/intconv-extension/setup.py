from torch.utils import cpp_extension
from setuptools import setup, Extension

setup(name='int8_conv',
      ext_modules=[
          cpp_extension.CUDAExtension(name ='int8conv_cuda',
                                      sources= ['int8conv.cpp', 'int8conv_kernel.cu'],
                                      library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                                      libraries=['cudnn'])
      ],
      cmdclass={
          'build_ext' : cpp_extension.BuildExtension
      })