#!/bin/bash
#MSUB -l nodes=8
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_SPAWNER=pmi

export REALM_BACKTRACE=1

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for i in 3 2 1 0; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+1)/2) ))
    ny=$(( 2 ** (i/2) ))
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x12_r"$r".log ]]; then
            echo "Running $n""x12_r$r"" ($n = $nx * $ny)..."
	    OMP_NUM_THREADS=36 srun -n $(( n * 2 )) -N $n "$root_dir/stencil.spmd12" -nx 20000 -ny 20000 -ntx $(( nx * 3 )) -nty $(( ny * 4 )) -tsteps 100 -tprune 5 -ll:cpu 12 -ll:io 1 -ll:util 1 -ll:dma 2 -ll:csize 30000 -ll:rsize 512 -ll:gsize 0 | tee out_"$n"x12_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x12_r"$r"_%.gz
        fi
    done
done

popd
