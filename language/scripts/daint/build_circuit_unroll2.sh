#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for c in 9; do
    SAVEOBJ=1 STANDALONE=1 OBJNAME=./circuit.spmd"$c" $root_dir/../../regent.py $root_dir/../../examples/circuit_sparse_unroll2.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fopenmp 0
done

cp $root_dir/*_circuit*.sh .
cp $root_dir/../summarize.py .
