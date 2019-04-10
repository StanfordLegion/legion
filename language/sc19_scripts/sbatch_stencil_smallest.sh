#!/bin/sh
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --time=00:40:00

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d manual ]]; then mkdir manual; fi
pushd manual

for i in 3 2 1 0; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.manual" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_sapce | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d auto ]]; then mkdir auto; fi
pushd auto

for i in 3 2 1 0; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.auto" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_sapce | tee out_"$n"x1_r"$r".log
  done
done

popd
