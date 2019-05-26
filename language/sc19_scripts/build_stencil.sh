#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

USE_FOREIGN=0 SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.manual time $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 0 -fopenmp 0 -foverride-demand-cuda 1 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal -flog 1  2>&1 |& tee compile_manual

SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.auto time $root_dir/../regent.py $root_dir/../examples/stencil_sequential.rg -fflow 0 -fparallelize-cache-incl-check 0 -fcuda-offline 1 -fcuda-arch pascal -flog 1  2>&1 |& tee compile_auto

cp $root_dir/*_stencil*.sh .
