#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for c in 14; do
    SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit_sparse_unroll2.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fopenmp 0
    mv circuit circuit.spmd"$c"
done

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libcircuit.so .

cp $root_dir/../scripts/*_circuit*.sh .
cp $root_dir/../scripts/summarize.py .
