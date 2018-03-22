#!/bin/sh

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for i in 0 1 2 3 4 5 6 7 8 9 10; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+2)/3) ))
    ny=$(( 2 ** ((i+1)/3) ))
    nz=$(( 2 ** ((i+0)/3) ))

    for c in 8; do
        time USE_HDF=0 TERRA_PATH=$root_dir/../liszt-legion/include/?.t SAVEOBJ=1 OBJNAME=taylor-256x128x128-dop"$n" $root_dir/../regent.py $root_dir/../soleil-x/src/soleil-x.t -i $root_dir/../soleil-x/testcases/taylor_green_vortex/taylor_green_vortex_256_128_128.lua -fopenmp 0 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fparallelize-dop $(( nx * 2 )),$(( ny * 2 )),$(( nz * 2 )) -fcuda 0 &
        sleep 3
    done
done

wait

cp $root_dir/../../bindings/terra/liblegion_terra.so .
cp $root_dir/../soleil-x/src/libsoleil_mapper.so .

cp $root_dir/../scripts/*_soleil*.sh .
