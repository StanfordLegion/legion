#!/bin/sh

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for i in 0; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+2)/3) ))
    ny=$(( 2 ** ((i+1)/3) ))
    nz=$(( 2 ** ((i+0)/3) ))

    for c in 3; do
        time USE_HDF=0 TERRA_PATH=$root_dir/../liszt-legion/include/?.t SAVEOBJ=1 OBJNAME=taylor-312x156x156-dop"$n" $root_dir/../regent.py $root_dir/../soleil-x/src/soleil-x.t -i $root_dir/../soleil-x/testcases/taylor_green_vortex/taylor_green_vortex_312_156_156.lua -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fparallelize-dop "$(( nx * 3 ))","$(( ny * 2 ))","$nz" -fopenmp 0 -fcuda 1 -fcuda-offline 1 &
        sleep 3
    done
done

wait

cp $root_dir/../../bindings/regent/libregent.so .
cp $root_dir/../soleil-x/src/libsoleil_mapper.so .

cp $root_dir/../scripts/*_soleil_cuda*.sh .
cp $root_dir/../scripts/summit_env.sh .
