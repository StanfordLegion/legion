#!/bin/bash
#PBS -A CSC103
#PBS -l walltime=01:00:00
#PBS -l nodes=8
#PBS -m abe

cd $PBS_O_WORKDIR

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for n in 8 4 2 1; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."
            srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.spmd8" -npp 10 -wpp 40 -l 100 -p $(( 1024 * 8 )) -pps $(( 1024 / n )) -ll:csize 20000 -ll:rsize 512 -ll:gsize 0 -ll:cpu 8 -ll:io 1 -ll:util 1 -ll:dma 1 | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd
