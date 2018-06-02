#!/bin/sh
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --time=00:40:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for n in 8 4; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x8_r$r""..."
    srun -n "$n" -N "$n" --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir"/taylor-dop"$n" -ll:csize 30000 -ll:rsize 1024 -ll:gsize 0 -ll:cpu 8 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:ht_sharing 0 -dm:memoize -lg:parallel_replay 2 -lg:window 4096 | tee out_"$n"x8_r"$r".log
  done
done

popd

if [[ ! -d notracing ]]; then mkdir notracing; fi
pushd notracing

for n in 8 4; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x8_r$r""..."
    srun -n "$n" -N "$n" --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir"/taylor-dop"$n" -ll:csize 30000 -ll:rsize 1024 -ll:gsize 0 -ll:cpu 8 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:ht_sharing 0 -dm:memoize -lg:parallel_replay 2 -lg:window 4096 -lg:no_tracing | tee out_"$n"x8_r"$r".log
  done
done

popd
