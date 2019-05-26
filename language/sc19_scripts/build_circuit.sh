#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./circuit.manual time $root_dir/../regent.py $root_dir/../examples/circuit_sparse.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal -flog 1  2>&1 |& tee compile_manual

SAVEOBJ=1 STANDALONE=1 OBJNAME=./circuit.auto   time $root_dir/../regent.py $root_dir/../examples/circuit_sparse_sequential.rg -fflow 0 -fcuda-offline 1 -fcuda-arch pascal -flog 1 -fparallelize-use-colocation 0  2>&1 |& tee compile_auto

cp $root_dir/*_circuit*.sh .
