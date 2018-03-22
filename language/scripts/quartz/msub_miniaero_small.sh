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

for n in 8 4 2 1; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."

	    OMP_NUM_THREADS=36 srun -n $(( n * 2 )) -N $n "$root_dir/miniaero.spmd8" -blocks $(( n * 2 * 8 )) -mesh 256x2048x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 8 -ll:io 1 -ll:util 1 -ll:dma 2 -ll:csize 20480 -ll:rsize 1024 -ll:gsize 0 | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd
