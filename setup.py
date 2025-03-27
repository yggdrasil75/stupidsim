from setuptools import setup, Extension
import pybind11
import sys

# MSVC-specific compilation flags
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/O2', '/fp:fast']
else:
    extra_compile_args = ['-std=c++17', '-O3', '-ffast-math']

ext_modules = [
    Extension(
        'utils_cpp',
        sources=['utils.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        'storms_cpp',
        sources=['storms.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='utils_cpp',
    version='0.1',
    ext_modules=ext_modules,
)