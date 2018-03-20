#!/bin/bash
#BSUB -P CSC275Chen
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -alloc_flags smt2
#BSUB -o lsf-%J.out
#BSUB -e lsf-%J.err

root_dir="$PWD"

source "$root_dir"/summit_env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"

export REALM_BACKTRACE=1

for n in 2; do
    for r in 0; do
        if [[ ! -f out_"$n"x2_r"$r".log ]]; then
            echo "Running $n""x2_r$r""..."
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --bind rs "$root_dir"/circuit.spmd18 -npp 20 -wpp 80 -l 100 -p $(( 1024 * 18 )) -pps $(( 1024 / n )) -ll:cpu 18 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:csize 60000 -ll:rsize 512 -ll:gsize 0 -ll:show_rsrv -hl:prof $(( n * 2 )) -hl:prof_logfile prof_"$n"x2_r"$r"_%.gz | tee out_"$n"x2_r"$r".log
        fi
    done
done
