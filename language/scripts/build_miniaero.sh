#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

# CC_FLAGS='-DDEFER_ARRIVALS_LOCALLY' ./setup_env.py

# DETAILED_MESSAGE_TIMING

SAVEOBJ=1 $root_dir/../regent.py $root_dir/../miniaero/rdir_1ghost.rg -fflow 0
mv $root_dir/../miniaero/miniaero_rdir_1ghost miniaero.none

# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 1
# mv $root_dir/../miniaero/miniaero_rdir_1ghost miniaero.spmd1
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 4
# mv $root_dir/../miniaero/miniaero_rdir_1ghost miniaero.spmd4
SAVEOBJ=1 $root_dir/../regent.py $root_dir/../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 8
mv $root_dir/../miniaero/miniaero_rdir_1ghost miniaero.spmd8
# SAVEOBJ=1 $root_dir/../regent.py $root_dir/../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 16
# mv $root_dir/../miniaero/miniaero_rdir_1ghost miniaero.spmd16

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../miniaero/librdir_1ghost.so .

cp $root_dir/../scripts/*_miniaero.sh .
