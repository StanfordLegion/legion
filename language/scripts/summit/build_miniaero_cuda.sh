#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for c in 3; do
    SAVEOBJ=1 $root_dir/../../regent.py $root_dir/../../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fopenmp 0 -fcuda 1 -fcuda-offline 1
    mv miniaero_rdir_1ghost miniaero.spmd"$c"
done

cp $root_dir/../../../bindings/regent/libregent.so .
cp $root_dir/../../miniaero/librdir_1ghost.so .

cp $root_dir/*_miniaero_cuda*.sh .
cp $root_dir/../summarize.py .

cp $root_dir/env.sh .
