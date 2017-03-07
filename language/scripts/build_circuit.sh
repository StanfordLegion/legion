#!/bin/bash

set -e

mkdir "$1"
cd "$1"

SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 0
mv circuit circuit.none

# SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 1
# mv circuit circuit.spmd1
# SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 10
# mv circuit circuit.spmd10
SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 20
mv circuit circuit.spmd20
# SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 12
# mv circuit circuit.spmd12
# SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 14
# mv circuit circuit.spmd14
# SAVEOBJ=1 ../regent.py ../examples/circuit_sparse.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 16
# mv circuit circuit.spmd16

cp ../../bindings/terra/liblegion_terra.so .
cp ../examples/libcircuit.so .

cp ../scripts/*_circuit.sh .
