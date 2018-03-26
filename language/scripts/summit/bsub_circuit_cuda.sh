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
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --gpu_per_rs 3 --bind rs --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" "$root_dir"/circuit.spmd18 -npp 20 -wpp 80 -l 100 -p $(( 1024 * 18 )) -pps $(( 1024 / n )) -ll:cpu 1 -ll:gpu 3 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:rsize 0 -ll:gsize 0 -ll:fsize 8192 -ll:zsize 512 -hl:prof $(( n * 2 )) -hl:prof_logfile prof_"$n"x2_r"$r"_%.gz -dm:memoize | tee out_"$n"x2_r"$r".log
        fi
    done
done
