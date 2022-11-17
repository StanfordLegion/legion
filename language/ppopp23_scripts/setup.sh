#!/bin/bash

set -e

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

REALM_NETWORKS=gasnet1 GASNET_VERSION=GASNet-1.30.0 USE_CUDA=1 CUDA_HOME=$CUDATOOLKIT_HOME HOST_CC=gcc HOST_CXX=g++ "$root_dir"/scripts/setup_env.py --llvm-version 38 --terra-cmake --terra-url https://github.com/terralang/terra.git --terra-branch release-1.0.6
