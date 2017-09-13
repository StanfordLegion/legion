#!/bin/bash
export GASNET=${LG_RT_DIR}/../language/gasnet/release
module unload PrgEnv-pgi
module load PrgEnv-gnu
module load python/3.5.1
export CC=cc
export CXX=CC
export HOST_CC=gcc
export HOST_CXX=g++
export USE_GASNET=1
export CONDUIT=gemini
