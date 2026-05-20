#!/usr/bin/env python
# coding=utf-8

"""python api for ascend_kernel."""

import os
from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from torch_npu.utils.cpp_extension import NpuExtension


WORKING_DIR = Path(__file__).resolve().parent
config = ConfigParser()
config.read(WORKING_DIR / "ascend_kernel" / "config.ini")
_version = config.get("global", "version")


setuptools.setup(
    name="ascend-kernel",
    version=_version,
    description="python api for ascend_kernel",
    packages=find_namespace_packages(exclude=("tests*",)),
    ext_modules=[NpuExtension("ascend_kernel._C", sources=[])],
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"ascend_kernel": ["lib/**", "VERSION"]},
)
