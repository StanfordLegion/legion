#!/bin/bash
#BSUB -P CSC275Chen
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -alloc_flags smt2
#BSUB -o lsf-%J.out
#BSUB -e lsf-%J.err

root_dir="$PWD"

source "$root_dir"/env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"

export REALM_BACKTRACE=1

export GASNET_NUM_QPS=1 # Hack: workaround for GASNet bug 3447

for n in 2; do
    for r in 0; do
        if [[ ! -f out_"$n"x2_r"$r".log ]]; then
            echo "Running $n""x2_r$r""..."
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --bind rs "$root_dir"/miniaero.spmd18 -blocks $(( n * 2 * 16 )) -mesh 256x2048x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 18 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:csize 60000 -ll:rsize 512 -ll:gsize 0 -ll:show_rsrv -hl:prof $(( n * 2 )) -hl:prof_logfile prof_"$n"x2_r"$r"_%.gz -dm:memoize | tee out_"$n"x2_r"$r".log
        fi
    done
done
