from torch.utils import cpp_extension
from setuptools import setup, Extension

setup(name='int8_mm',
      ext_modules=[
          cpp_extension.CUDAExtension(name ='int8mm_cuda',
                                      sources= ['int8mm.cpp', 'int8mm_kernel.cu'],
                                      library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                                      libraries=['cudnn'])
      ],
      cmdclass={
          'build_ext' : cpp_extension.BuildExtension
      })