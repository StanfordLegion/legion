#!/bin/bash

set -e

mkdir "$1"
cd "$1"

SAVEOBJ=1 ../regent.py ../examples/pennant_fast.rg -fflow 0
mv ./pennant pennant.none

SAVEOBJ=1 ../regent.py ../examples/pennant_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize 14
mv ./pennant pennant.spmd14

cp ../../bindings/terra/liblegion_terra.so .
cp ../examples/libpennant.so .

cp -r ../examples/pennant.tests .

cp ../scripts/*_pennant.sh .
