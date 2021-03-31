#!/bin/bash

set -e

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

USE_CUDA=1 CUDA_HOME=$CUDATOOLKIT_HOME HOST_CC=gcc HOST_CXX=g++ "$root_dir"/scripts/setup_env.py --llvm-version 60
