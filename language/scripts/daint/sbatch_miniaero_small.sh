#!/bin/sh
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"


export LD_LIBRARY_PATH="$PWD"

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for n in 8 4 2 1; do
  for r in 0 1 2 3 4; do
    if [[ ! -f out_"$n"x8_r"$r".log ]]; then
      echo "Running $n""x8_r$r""..."
      srun -n "$n" -N "$n" --cpu_bind none --ntasks-per-node 1 /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd8" -blocks "$(( n * 8 ))" -mesh 8x8x16384 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 8 -ll:io 1 -ll:dma 2 -ll:util 3 -ll:ht_sharing 0 -ll:csize 30000 -ll:rsize 4096 -ll:gsize 0 -dm:memoize -lg:parallel_replay 3 | tee out_"$n"x8_r"$r".log
    fi
  done
done

popd

if [[ ! -d notracing ]]; then mkdir notracing; fi
pushd notracing

for n in 8 4 2 1; do
  for r in 0 1 2 3 4; do
    if [[ ! -f out_"$n"x8_r"$r".log ]]; then
      echo "Running $n""x8_r$r""..."
      srun -n "$n" -N "$n" --cpu_bind none --ntasks-per-node 1 /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd8" -blocks "$(( n * 8 ))" -mesh 8x8x16384 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 8 -ll:io 1 -ll:dma 2 -ll:util 3 -ll:ht_sharing 0 -ll:csize 30000 -ll:rsize 4096 -ll:gsize 0 -lg:no_tracing | tee out_"$n"x8_r"$r".log
    fi
  done
done

popd
