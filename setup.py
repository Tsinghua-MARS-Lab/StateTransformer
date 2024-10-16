import os
from setuptools import find_packages
from distutils.core import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='transformer4planning',
    version='1.0.0',
    author='QiaoSun & Shiduo-zh',
    license="MIT",
    packages=find_packages(),
    author_email='',
    description='',
    install_requires=[],
    cmdclass={
            'build_ext': BuildExtension,
    },
)