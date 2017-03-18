#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/pennant_fast.rg -fflow 0
mv ./pennant pennant.none

# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/pennant_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 4
# mv ./pennant pennant.spmd4
SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/pennant_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 10
mv ./pennant pennant.spmd10

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libpennant.so .

cp -r $root_dir/../examples/pennant.tests .

cp $root_dir/../scripts/*_pennant.sh .
