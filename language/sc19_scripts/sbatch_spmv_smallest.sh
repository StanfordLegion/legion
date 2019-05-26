#!/bin/bash -eu
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00

ROOT_DIR="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d auto ]]; then mkdir auto; fi
pushd auto

for n in 8 4 2 1; do
  for r in 0 1 2 3 4; do
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none \
	"$ROOT_DIR/spmv.auto" -steps 50 -prune 30 -p $n -size $(( 2**27*n )) \
	-ll:gpu 1 -ll:util 2 -ll:dma 2 \
	-ll:csize 15000 -ll:fsize 15000 -ll:rsize 512 -ll:gsize 0 \
	-dm:same_address_space -level 5 &> out_"$n"x1_r"$r".log
  done
done

popd
