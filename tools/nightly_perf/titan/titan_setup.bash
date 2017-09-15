#!/bin/bash
export GASNET=${LG_RT_DIR}/../language/gasnet/release
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
