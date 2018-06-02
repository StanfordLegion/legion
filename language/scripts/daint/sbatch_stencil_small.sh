#!/bin/sh
#SBATCH --nodes=64
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for i in 6 5 4; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x8_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.spmd8" -nx 20480 -ny 20480 -ntx $(( nx * 2 )) -nty $(( ny * 4 )) -tsteps 400 -tprune 10 -ll:cpu 8 -ll:io 1 -ll:util 3 -ll:dma 2 -ll:csize 30000 -ll:rsize 512 -ll:gsize 0 -dm:memoize -lg:parallel_replay 3 -ll:ht_sharing 0 | tee out_"$n"x8_r"$r".log
  done
done

popd

if [[ ! -d notracing ]]; then mkdir notracing; fi
pushd notracing

for i in 6 5 4; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x8_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.spmd8" -nx 20480 -ny 20480 -ntx $(( nx * 2 )) -nty $(( ny * 4 )) -tsteps 400 -tprune 10 -ll:cpu 8 -ll:io 1 -ll:util 3 -ll:dma 2 -ll:csize 30000 -ll:rsize 512 -ll:gsize 0 -lg:no_tracing -ll:ht_sharing 0 | tee out_"$n"x8_r"$r".log
  done
done

popd
