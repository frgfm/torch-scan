# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Build the PyTorch collector and independent C++ counting library/executable."""

import sys
from distutils.command.build_scripts import build_scripts
from pathlib import Path
from runpy import run_path

import torch
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_COUNTER = ["native/count/count.cpp", "native/count/formulas.cpp"]
_FLAGS = ["/O2", "/utf-8"] if sys.platform == "win32" else ["-O2", "-g0"]
_COUNTER_FLAGS = [*_FLAGS, *(["/std:c++17", "/EHsc"] if sys.platform == "win32" else ["-std=c++17"])]


class NativeBuild(BuildExtension):
    """Share compiled counting objects between the binding and executable."""

    def build_extensions(self) -> None:
        generated = Path(self.build_temp) / "tables"
        generated.mkdir(parents=True, exist_ok=True)
        template = Path("native/count/tables.hpp.in").read_text(encoding="utf-8")
        for name in ("rules", "ignored"):
            template = template.replace(
                f"@{name.upper()}@", Path(f"native/count/{name}.json").read_text(encoding="utf-8")
            )
        (generated / "tables.hpp").write_text(template, encoding="utf-8")
        analysis = next(ext for ext in self.extensions if ext.name == "torchscan._analysis")
        analysis.include_dirs.append(str(generated))
        include = run_path("_build_deps.py")["prepare_headers"](Path(self.build_temp) / "dependencies")
        analysis.include_dirs.append(str(include))
        super().build_extensions()
        objects = self.compiler.object_filenames(_COUNTER, output_dir=self.build_temp)
        objects += self.compiler.compile(
            ["native/count/main.cpp"],
            output_dir=self.build_temp,
            include_dirs=analysis.include_dirs,
            extra_postargs=analysis.extra_compile_args,
        )
        target = str(Path(self.build_temp) / "torchscan-count")
        self.compiler.link_executable(objects, target, target_lang="c++")
        self.counter_path = self.compiler.executable_filename(target)


class NativeScripts(build_scripts):
    """Install the real executable into the wheel's scripts directory."""

    def run(self) -> None:
        self.run_command("build_ext")
        native = self.get_finalized_command("build_ext")
        self.mkpath(self.build_dir)
        self.outfiles = [self.copy_file(native.counter_path, self.build_dir)[0]]

    def get_source_files(self) -> list[str]:
        return []  # Sources are included by MANIFEST.in.

    def get_outputs(self) -> list[str]:
        return self.outfiles


setup(
    license_files=["LICENSE", "native/count/THIRD_PARTY_LICENSES.txt"],
    install_requires=[f"torch=={torch.__version__.split('+')[0]}"],
    ext_modules=[
        CppExtension("_torchscan_native", ["native/collector.cpp"], extra_compile_args=_FLAGS.copy()),
        Extension(
            "torchscan._analysis",
            [*_COUNTER, "native/count/python.cpp"],
            language="c++",
            extra_compile_args=_COUNTER_FLAGS.copy(),
        ),
    ],
    scripts=["torchscan-count"],
    cmdclass={"build_ext": NativeBuild.with_options(use_ninja=False, force=True), "build_scripts": NativeScripts},
)
