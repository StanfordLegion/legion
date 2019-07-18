#!/bin/bash

set -e

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

if [[ ! -d "$root_dir"/conda ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-$(uname -p).sh -O conda-installer.sh
    bash ./conda-installer.sh -b -p "$root_dir"/conda
    rm conda-installer.sh
    cat > "$root_dir"/env.sh <<EOF
source "$root_dir"/conda/etc/profile.d/conda.sh
conda activate
export PYTHON_LIB="$root_dir"/conda/lib/libpython3.7m.so
export PYTHON_VERSION_MAJOR=3
EOF
    source "$root_dir"/env.sh
    conda install -y numpy cffi
else
    source "$root_dir"/env.sh
fi

if [[ ! -f $PYTHON_LIB ]]; then
    echo "Failed to locate $PYTHON_LIB"
    exit 1
fi

USE_CUDA=1 CUDA_HOME=$CUDATOOLKIT_HOME USE_PYTHON=1 HOST_CC=gcc HOST_CXX=g++ "$root_dir"/scripts/setup_env.py --llvm-version 38
LG_RT_DIR="$root_dir"/../runtime make -C "$root_dir"/../bindings/python clean
LG_RT_DIR="$root_dir"/../runtime make -C "$root_dir"/../bindings/python cached_legion.h
