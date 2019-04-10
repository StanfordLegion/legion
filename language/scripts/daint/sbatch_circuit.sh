#!/bin/sh
#SBATCH --nodes=256
#SBATCH --constraint=gpu
#SBATCH --time=00:40:00

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d auto ]]; then mkdir auto; fi
pushd auto

for n in 256 128; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.auto" -npp 2500 -wpp 10000 -l 50 -p $(( $n * 10 )) -pps 10 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d manual ]]; then mkdir manual; fi
pushd manual

for n in 256 128; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.manual"  -npp 2500 -wpp 10000 -l 50 -p $(( $n * 10 )) -pps 10 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space | tee out_"$n"x1_r"$r".log
  done
done

popd
