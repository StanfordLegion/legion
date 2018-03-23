#!/bin/bash
#BSUB -P CSC275Chen
#BSUB -W 1:00
#BSUB -nnodes 2
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
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --gpu_per_rs 3 --bind rs --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" "$root_dir"/circuit.spmd2 -ll:cpu 2 -ll:gpu 1 -ll:fsize 1024 -ll:show_rsrv -level announce=2 -dm:memoize | tee out_"$n"x2_r"$r".log
        fi
    done
done
