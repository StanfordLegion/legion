#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

# Hack: -ffuture 0 is a workaround for blocking on a future with the trace loop

USE_FOREIGN=0 SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.idx $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 0 -fopenmp 0 -foverride-demand-cuda 1 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal -findex-launch 1 -ffuture 0

cp $root_dir/*_stencil*.sh .
