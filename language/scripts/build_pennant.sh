#!/bin/bash

set -e

mkdir "$1"
cd "$1"

SAVEOBJ=1 ../regent.py ../examples/pennant_stripmine.rg -fflow 0
mv ./pennant pennant.none

SAVEOBJ=1 ../regent.py ../examples/pennant_stripmine.rg -fflow 1 -fflow-spmd-shardsize 12
mv ./pennant pennant.spmd12

cp ../../bindings/terra/liblegion_terra.so .
cp ../examples/libpennant.so .

cp -r ../examples/pennant.tests .

cp ../run_pennant.sh .
