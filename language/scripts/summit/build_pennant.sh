#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for c in 9; do
    SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.spmd"$c" $root_dir/../../regent.py $root_dir/../../examples/pennant_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fvectorize-unsafe 1 -fopenmp 0 -fcuda 0
done

cp -r $root_dir/../../examples/pennant.tests .

cp $root_dir/*_pennant*.sh .
cp $root_dir/../summarize.py .

cp $root_dir/env.sh .
