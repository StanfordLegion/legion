#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 0
mv stencil stencil.none

for c in 10; do
    SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c"
    mv stencil stencil.spmd"$c"
done

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libstencil.so .
cp $root_dir/../examples/libstencil_mapper.so .

cp $root_dir/../scripts/*_stencil.sh .
