#!/bin/sh
#SBATCH --nodes=16
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

if [[ ! -d spmd10 ]]; then mkdir spmd10; fi
pushd spmd10

for i in 0 1 2 3; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+1)/2) ))
    ny=$(( 2 ** (i/2) ))
    if [[ ! -f out_"$n"x10.log ]]; then
        echo "Running $n""x10 ($n = $nx * $ny)..."
	srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.spmd10" -nx $(( nx * 40000 )) -ny $(( ny * 40000 )) -ntx $(( nx * 2 )) -nty $(( ny * 5 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 11 -ll:util 1 -ll:dma 1 -ll:csize 30000 -ll:rsize 512 -ll:gsize 0 -hl:prof 4 -level legion_prof=2 -logfile prof_"$n"x10_%.log | tee out_"$n"x10.log
    fi
done

popd
