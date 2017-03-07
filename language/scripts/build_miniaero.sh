#!/bin/bash

set -e

mkdir "$1"
cd "$1"

# CC_FLAGS='-DDEFER_ARRIVALS_LOCALLY' ./setup_env.py

# DETAILED_MESSAGE_TIMING

SAVEOBJ=1 ../regent.py ../miniaero/rdir_1ghost.rg -fflow 0
mv ../miniaero/miniaero_rdir_1ghost miniaero.none

SAVEOBJ=1 ../regent.py ../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd-shardsize 1
mv ../miniaero/miniaero_rdir_1ghost miniaero.spmd1
SAVEOBJ=1 ../regent.py ../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd-shardsize 4
mv ../miniaero/miniaero_rdir_1ghost miniaero.spmd4
SAVEOBJ=1 ../regent.py ../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd-shardsize 8
mv ../miniaero/miniaero_rdir_1ghost miniaero.spmd8
SAVEOBJ=1 ../regent.py ../miniaero/rdir_1ghost.rg -fflow 1 -fflow-spmd-shardsize 16
mv ../miniaero/miniaero_rdir_1ghost miniaero.spmd16

cp ../../bindings/terra/liblegion_terra.so .
cp ../miniaero/librdir_1ghost.so .

cp ../scripts/run_miniaero.sh .
