#!/bin/bash
#MSUB -l nodes=8
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_SPAWNER=pmi

export REALM_BACKTRACE=1

if [[ ! -d notracing ]]; then mkdir notracing; fi
pushd notracing

for n in 8 4 2 1; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x15_r"$r".log ]]; then
            echo "Running $n""x15_r$r""..."

            OMP_NUM_THREADS=36 srun -n $(( n * 2 )) -N $n "$root_dir/pennant.spmd15" "$root_dir/pennant.tests/leblanc_long16x30/leblanc.pnt" -npieces $(( $n * 15 )) -numpcx 1 -numpcy $(( $n * 15 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 15 -ll:io 1 -ll:util 1 -ll:dma 2 -ll:csize 50000 -ll:rsize 1024 -ll:gsize 0 -lg:no_physical_tracing | tee out_"$n"x15_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x15_r"$r"_%.gz
        fi
    done
done

popd
