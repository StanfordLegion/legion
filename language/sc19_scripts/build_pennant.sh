#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.manual $root_dir/../regent.py $root_dir/../examples/pennant.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.auto   $root_dir/../regent.py $root_dir/../examples/pennant_sequential.rg -fflow 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

cp $root_dir/*_pennant*.sh .
