from setuptools import setup, Extension
import pybind11
import sys
import platform

# Detect if using MSVC on Windows
if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17"]
else:
    extra_compile_args = ["-std=c++17"] #, "-pedantic", "-Wall", "-Wextra", "-fno-fast-math"]

setup(
    name="icosphere",
    version="0.1",
    ext_modules=[
        Extension(
            "icosphere",
            ["icosphere.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    python_requires=">=3.6",
)
