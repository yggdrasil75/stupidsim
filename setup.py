from setuptools import setup, Extension
import pybind11

setup(
    name="icosphere",
    version="0.1",
    ext_modules=[
        Extension(
            "icosphere",
            ["icosphere.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],
        )
    ],
    python_requires=">=3.6",
)