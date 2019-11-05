#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

USE_FOREIGN=0 SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.scr $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fopenmp 0 -foverride-demand-cuda 1 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal
USE_FOREIGN=0 SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.dcr $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 0 -fopenmp 0 -foverride-demand-cuda 1 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

cp $root_dir/*_stencil*.sh .
