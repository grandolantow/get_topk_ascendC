#!/bin/bash
set -e

# Recompile kernel
bash build.sh -a kernels

# Run NPU perf benchmark
python tests/bench_set_difference_npu.py
