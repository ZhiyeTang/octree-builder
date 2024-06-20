#
# Copyright (C) 2024, Shenzhen University
# Immersive Media Laboratory, Institute of Future Media Computing
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
# 

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="octree_builder",
    ext_modules=[
        CUDAExtension(
            name="octree_builder._C",
            sources=[
            "octree_builder.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
