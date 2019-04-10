#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 $root_dir/../../regent.py $root_dir/../../miniaero/rdir_1ghost.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal #-fparallelize-time-deppart 1
mv miniaero_rdir_1ghost miniaero.manual

SAVEOBJ=1 $root_dir/../../regent.py $root_dir/../../miniaero/sequential.rg  -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal #-fparallelize-time-deppart 1
mv miniaero_sequential miniaero.auto

cp $root_dir/../../../bindings/regent/libregent.so .
cp $root_dir/../../miniaero/librdir_1ghost.so .
cp $root_dir/../../miniaero/libsequential.so .

cp $root_dir/*_miniaero*.sh .
cp $root_dir/../summarize.py .
