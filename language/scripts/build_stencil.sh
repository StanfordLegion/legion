#!/bin/bash

set -e

mkdir "$1"
cd "$1"

SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 0
mv stencil stencil.none

# SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 1
# mv stencil stencil.spmd1
# SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 10
# mv stencil stencil.spmd10
# SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 12
# mv stencil stencil.spmd12
SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 14
mv stencil stencil.spmd14
# SAVEOBJ=1 ../regent.py ../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 16
# mv stencil stencil.spmd16

cp ../../bindings/terra/liblegion_terra.so .
cp ../examples/libstencil.so .
cp ../examples/libstencil_mapper.so .

cp ../scripts/run_stencil.sh .
