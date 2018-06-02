#!/bin/sh
#SBATCH --nodes=64
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d tracing_prof ]]; then mkdir tracing_prof; fi
pushd tracing_prof

for n in 64 32 16; do
  for r in 0; do
    echo "Running $n""x9_r$r""..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.spmd9" -npp 16 -wpp 32 -l 100 -prune 10 -p $(( 256 * 9 )) -pps $(( 256 / n )) -ll:csize 20480 -ll:rsize 512 -ll:gsize 0 -ll:cpu 9 -ll:io 1 -ll:util 3 -ll:dma 2 -dm:memoize -lg:parallel_replay 3 -ll:ht_sharing 0 -lg:prof "$n" -lg:prof_logfile out_"$n"x9_%.gz | tee out_"$n"x9_r"$r".log
  done
done

popd

if [[ ! -d notracing_prof ]]; then mkdir notracing_prof; fi
pushd notracing_prof

for n in 64 32 16; do
  for r in 0; do
    echo "Running $n""x9_r$r""..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.spmd9" -npp 16 -wpp 32 -l 100 -prune 10 -p $(( 256 * 9 )) -pps $(( 256 / n )) -ll:csize 20480 -ll:rsize 512 -ll:gsize 0 -ll:cpu 9 -ll:io 1 -ll:util 3 -ll:dma 2 -ll:ht_sharing 0 -lg:no_tracing -lg:prof "$n"-lg:prof_logfile out_"$n"x9_%.gz | tee out_"$n"x9_r"$r".log
  done
done

popd
