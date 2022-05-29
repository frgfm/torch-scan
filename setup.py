# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

version = '0.1.2.dev0'
sha = 'Unknown'
package_name = 'torchscan'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    if sha != 'Unknown':
        version += '+' + sha[:7]
print(f"Building wheel {package_name}-{version}")

with open(cwd.joinpath('torchscan', 'version.py'), 'w', encoding='utf-8') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

_deps = [
    "torch>=1.5.0",
    # Testing
    "pytest>=5.3.2",
    "coverage>=4.5.4",
    # Quality
    "flake8>=3.9.0",
    "isort>=5.7.0",
    "mypy>=0.812",
    "pydocstyle>=6.0.0",
    # Docs
    "sphinx<=3.4.3",
    "sphinx-rtd-theme==0.4.3",
    "sphinxemoji>=0.1.8",
    "sphinx-copybutton>=0.3.1",
    "docutils<0.18",
    "Jinja2<3.1",  # cf. https://github.com/readthedocs/readthedocs.org/issues/9038
]

# Borrowed from https://github.com/huggingface/transformers/blob/master/setup.py
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


install_requires = [
    deps["torch"],
]

extras = {}

extras["testing"] = deps_list(
    "pytest",
    "coverage",
)

extras["quality"] = deps_list(
    "flake8",
    "isort",
    "mypy",
    "pydocstyle",
)

extras["docs"] = deps_list(
    "sphinx",
    "sphinx-rtd-theme",
    "sphinxemoji",
    "sphinx-copybutton",
    "docutils",
    "Jinja2",
)

extras["dev"] = (
    extras["testing"]
    + extras["quality"]
    + extras["docs"]
)


setup(
    # Metadata
    name=package_name,
    version=version,
    author='François-Guillaume Fernandez',
    author_email='fg-feedback@protonmail.com',
    description='Useful information about your Pytorch module',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/torch-scan',
    download_url='https://github.com/frgfm/torch-scan/tags',
    license='Apache',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['pytorch', 'deep learning', 'summary', 'memory', 'ram'],

    # Package info
    packages=find_packages(exclude=('tests',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
    package_data={'': ['LICENSE']}
)
