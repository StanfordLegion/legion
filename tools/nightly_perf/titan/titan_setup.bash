#!/bin/bash

if [[ "${LG_RT_DIR}" == "" ]]
then
  echo must define LG_RT_DIR
  exit
fi

export GASNET=${LG_RT_DIR}/../language/gasnet/release
export LLVM_CONFIG=${LG_RT_DIR}/../language/llvm/install/bin/llvm-config
export CLANG=${LG_RT_DIR}/../language/llvm/install/bin/clang
export TERRA_DIR=${LG_RT_DIR}/../language/terra

module unload PrgEnv-pgi
module load PrgEnv-gnu
module load python

export CC=gcc
export CXX=CC
export HOST_CC=gcc
export HOST_CXX=g++
export USE_GASNET=1
export CONDUIT=gemini
export PERF_CORES_PER_NODE=12
export LAUNCHER=aprun
export USE_RDIR=1
