#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./circuit.normal $root_dir/../regent.py $root_dir/../examples/circuit_sparse.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

SAVEOBJ=1 STANDALONE=1 OBJNAME=./circuit.python $root_dir/../regent.py $root_dir/../../apps/circuit/python/circuit_sparse.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

cp $root_dir/*_circuit*.sh .

cp $root_dir/../../apps/circuit/python/*.py .
cp $root_dir/../../apps/circuit/python/*.h .
