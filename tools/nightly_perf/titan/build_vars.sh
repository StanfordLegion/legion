#!/bin/bash

# Variables needed to build Legion and Regent.

module unload PrgEnv-pgi
module load PrgEnv-gnu
module load python

export CC=gcc
export CXX=CC
export HOST_CC=gcc
export HOST_CXX=g++

export MARCH=barcelona
export USE_GASNET=1
export USE_LIBDL=0
export CONDUIT=gemini
export RDIR=auto

unset LG_RT_DIR
