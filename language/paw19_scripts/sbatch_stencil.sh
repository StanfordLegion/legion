#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

source "$root_dir"/../env.sh

export LD_LIBRARY_PATH="$PWD"

nodes=$SLURM_JOB_NUM_NODES
power=$(echo "l($nodes)/l(2)" | bc -l | xargs printf '%.0f\n')

if [[ ! -d normal ]]; then mkdir normal; fi
pushd normal

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.normal" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space 1 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d python ]]; then mkdir python; fi
pushd python

export LG_RT_DIR="$root_dir"/../../runtime
export PYTHONPATH="$PYTHONPATH:$LG_RT_DIR/../bindings/python:$root_dir"

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.python" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:dma 2 -ll:csize 15000 -ll:fsize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -ll:py 1 -ll:pyimport stencil -dm:same_address_space 1 | tee out_"$n"x1_r"$r".log
  done
done

popd
