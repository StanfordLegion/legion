#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 0
mv circuit circuit.none

for c in 10; do
    SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c"
    mv circuit circuit.spmd"$c"
done

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libcircuit.so .

cp $root_dir/../scripts/*_circuit.sh .
