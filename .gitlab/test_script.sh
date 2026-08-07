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
    if ! echo "32f6420330de4d7176396aa36929a76733fe5a1fbc5a0cf8b9a6d270f9630d8d  terra-Linux-x86_64-cc543db.tar.xz" | shasum -a 256 -c; then
        wget -nv https://github.com/terralang/terra/releases/download/release-1.2.0/terra-Linux-x86_64-cc543db.tar.xz
    fi
    if ! echo "e86847217fdb8fa3bdde964540a3e945c171ea0a01004fa9b95fd41e8e9f20ac  clang+llvm-18.1.7-x86_64-linux-gnu.tar.xz" | shasum -a 256 -c; then
        wget -nv https://github.com/terralang/llvm-build/releases/download/llvm-18.1.7/clang+llvm-18.1.7-x86_64-linux-gnu.tar.xz
    fi
    popd
    tar xf "$CACHE_DIR/terra-Linux-x86_64-cc543db.tar.xz"
    ln -s terra-Linux-x86_64-cc543db terra
    tar xf "$CACHE_DIR/clang+llvm-18.1.7-x86_64-linux-gnu.tar.xz"
    ln -s clang+llvm-18.1.7-x86_64-linux-gnu llvm
    export REGENT_LLVM_PATH="$PWD/llvm"
fi

# download Thrust
git clone https://github.com/ROCm/Thrust.git
export THRUST_PATH="$PWD/Thrust"

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
