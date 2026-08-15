#!/bin/bash

set -e
source "$PWD/.gitlab/env_frontier.sh"
set -x

# job directory
echo "Running tests in $PWD"

# download Terra
if [[ ${TEST_REGENT:-1} -eq 1 ]]; then
    mkdir -p "$CACHE_DIR"
    pushd "$CACHE_DIR"
    if ! echo "7359c60f056a0300c1f3cbc11c26f370b62f6b8216190a78739b362ccf432593  terra-Linux-x86_64-bb02b25.tar.xz" | shasum -a 256 -c; then
        wget -nv https://github.com/terralang/terra/releases/download/release-1.2.2/terra-Linux-x86_64-bb02b25.tar.xz
    fi
    if ! echo "a0bfebc31391ef0b31197c68deb5e3bf0a0e18aa55279cc39e85a18909e60663  clang+llvm-22.1.8-x86_64-linux-gnu.tar.xz" | shasum -a 256 -c; then
        wget -nv https://github.com/terralang/llvm-build/releases/download/llvm-22.1.8/clang+llvm-22.1.8-x86_64-linux-gnu.tar.xz
    fi
    popd
    tar xf "$CACHE_DIR/terra-Linux-x86_64-bb02b25.tar.xz"
    ln -s "$PWD/terra-Linux-x86_64-bb02b25" language/terra
    tar xf "$CACHE_DIR/clang+llvm-22.1.8-x86_64-linux-gnu.tar.xz"
    export REGENT_LLVM_PATH="$PWD/clang+llvm-22.1.8-x86_64-linux-gnu"
fi

# download GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
    if [[ "$GASNET_DEBUG" -eq 1 ]]; then
        export GASNet_ROOT="$PWD/gasnet/debug"
    else
        export GASNet_ROOT="$PWD/gasnet/release"
    fi
fi

# build GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    set +x # makes the build very noisy
    CONDUIT=$GASNET_CONDUIT make -C gasnet -j${THREADS:-16}
    set -x
fi

if [[ "$REALM_NETWORKS" != "" ]]; then
    RANKS_PER_NODE=4
    export LAUNCHER="srun -n$(( RANKS_PER_NODE * SLURM_JOB_NUM_NODES )) --cpus-per-task $(( 56 / RANKS_PER_NODE )) --gpus-per-task $(( 8 / RANKS_PER_NODE )) --cpu-bind cores"
    if [[ SLURM_JOB_NUM_NODES -eq 1 ]]; then
        export LAUNCHER+=" --network=single_node_vni"
    fi
fi

# required for machine_config test to pin NUMA memory
ulimit -l $(( 1024 * 1024 )) # KB

# get backtraces if necessary
export REALM_BACKTRACE=1

# run test script
./tools/add_github_host_key.sh
grep 'model name' /proc/cpuinfo | uniq -c || true
which cmake
cmake --version
which $CXX
$CXX --version
free

if [[ -z "$TEST_PYTHON_EXE" ]]; then
    export TEST_PYTHON_EXE=`which python3 python | head -1`
fi
$TEST_PYTHON_EXE ./test.py -j${THREADS:-16}
