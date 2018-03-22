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
        if [[ ! -f out_"$n"x14_r"$r".log ]]; then
            echo "Running $n""x14_r$r""..."

            OMP_NUM_THREADS=36 srun -n $(( n * 2 )) -N $n "$root_dir/circuit.spmd14" -npp 10 -wpp 40 -l 100 -p $(( 1024 * 14 )) -pps $(( 1024 / n )) -ll:csize 30000 -ll:rsize 512 -ll:gsize 0 -ll:cpu 14 -ll:io 1 -ll:util 1 -ll:dma 2 -lg:no_physical_tracing | tee out_"$n"x14_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x14_r"$r"_%.gz
        fi
    done
done

popd
