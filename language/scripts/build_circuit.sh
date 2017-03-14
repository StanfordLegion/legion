#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 0
mv circuit circuit.none

# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 1
# mv circuit circuit.spmd1
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 10
# mv circuit circuit.spmd10
SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 10
mv circuit circuit.spmd10
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 12
# mv circuit circuit.spmd12
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 14
# mv circuit circuit.spmd14
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/circuit.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 16
# mv circuit circuit.spmd16

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libcircuit.so .

cp $root_dir/../scripts/*_circuit.sh .
