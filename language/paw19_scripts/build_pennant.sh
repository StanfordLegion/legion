#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.normal $root_dir/../regent.py $root_dir/../examples/pennant.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

SAVEOBJ=1 STANDALONE=0 OBJNAME=./pennant.python $root_dir/../regent.py $root_dir/../../apps/pennant/python/pennant_python.rg -fflow 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal

cp $root_dir/*_pennant*.sh .

cp $root_dir/../../apps/pennant/python/*.py .
cp $root_dir/../../apps/pennant/python/*.h .

gcc -I $root_dir/../../runtime -DLEGION_USE_PYTHON_CFFI -DLEGION_MAX_DIM=3 -DREALM_MAX_DIM=3 -E -P pennant_config.h > cached_pennant_config.h
