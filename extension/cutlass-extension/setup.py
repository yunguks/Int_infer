from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# git clone cutlass
# cd cutlass
# mkdir build && cd build
# export CUDA_PATH="/usr/local/cuda" 
# cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_OPERATIONS=conv2d -DCMAKE_INSTALL_PREFIX=/usr/local
# make -j8
# make install 
setup(
    name='cutlassconv',
    ext_modules=[
        CUDAExtension('cutlassconv',
                      ['cutlassconv.cpp',
                       'cutlassconv_kernel.cu',],
                      library_dirs=['/usr/lib/x86_64-linux-gnu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
