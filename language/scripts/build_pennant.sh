#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/pennant_fast_vectorized.rg -fflow 0 -fvectorize-unsafe 1
mv ./pennant pennant.none

for c in 10; do
    SAVEOBJ=1 $root_dir/../regent.py $root_dir/../examples/pennant_fast_vectorized.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fvectorize-unsafe 1
    mv ./pennant pennant.spmd"$c"
done

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../examples/libpennant.so .

cp -r $root_dir/../examples/pennant.tests .

cp $root_dir/../scripts/*_pennant.sh .
cp $root_dir/../scripts/summarize.py .
