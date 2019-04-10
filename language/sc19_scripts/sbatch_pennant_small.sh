#!/bin/sh
#SBATCH --nodes=64
#SBATCH --constraint=gpu
#SBATCH --time=00:40:00

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d auto ]]; then mkdir auto; fi
pushd auto

for n in 64 32 16; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.auto" "$root_dir"/pennant.tests/leblanc_long"$n"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d manual ]]; then mkdir manual; fi
pushd manual

for n in 64 32 16; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.manual" "$root_dir"/pennant.tests/leblanc_long"$n"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space | tee out_"$n"x1_r"$r".log
  done
done

popd
