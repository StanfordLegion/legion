#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

source "$root_dir"/../env.sh

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d normal ]]; then mkdir normal; fi
pushd normal

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.normal" "$root_dir"/pennant.tests/leblanc_long"$n"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space 1 | tee out_"$n"x1_r"$r".log
    # -ll:gpu 1 -ll:fsize 15000 -ll:zsize 15000
  done
done

popd

if [[ ! -d python ]]; then mkdir python; fi
pushd python

export LG_RT_DIR="$root_dir"/../../runtime
export PYTHONPATH="$PYTHONPATH:$LG_RT_DIR/../bindings/python:$root_dir"

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.python" "$root_dir"/pennant.tests/leblanc_long"$n"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:util 2 -ll:dma 2 -ll:csize 15000 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:same_address_space 1 -ll:py 1 -ll:pyimport pennant | tee out_"$n"x1_r"$r".log
    # -ll:gpu 1 -ll:fsize 15000 -ll:zsize 15000
  done
done

popd
