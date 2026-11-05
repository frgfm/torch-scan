# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Build the C++ collector and shared Rust counter at installation time."""

import sys

import torch
from setuptools import setup
from setuptools_rust import Binding, RustBin, RustExtension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    install_requires=[f"torch=={torch.__version__.split('+')[0]}"],
    rust_extensions=[
        RustExtension(
            "torchscan._analysis",
            path="native/count/Cargo.toml",
            features=["python"],
            binding=Binding.PyO3,
            cargo_manifest_args=["--locked"],
            debug=False,
        ),
        RustBin("torchscan-count", path="native/count/Cargo.toml", cargo_manifest_args=["--locked"], debug=False),
    ],
    ext_modules=[
        CppExtension(
            "_torchscan_native",
            ["native/collector.cpp"],
            extra_compile_args=["/O2", "/utf-8"] if sys.platform == "win32" else ["-O2", "-g0"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False, force=True)},
)
