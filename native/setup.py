# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Build the optional collector against the installed PyTorch, without JIT compilation."""

import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="torchscan-native",
    version="0.2.0.dev0",
    python_requires=">=3.11",
    license="Apache-2.0",
    license_files=["LICENSE"],
    install_requires=[f"torch=={torch.__version__.split('+')[0]}"],
    ext_modules=[
        CppExtension(
            "_torchscan_native", ["collector.cpp"], extra_compile_args=["/O2" if sys.platform == "win32" else "-O2"]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False, force=True)},
)
