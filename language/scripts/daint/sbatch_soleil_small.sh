#!/bin/sh
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d tracing ]]; then mkdir tracing; fi
pushd tracing

for n in 8 4 2 1; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."
            srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir"/taylor-256x128x128-dop"$n" -ll:csize 30000 -ll:rsize 1024 -ll:gsize 0 -ll:cpu 0 -ll:ocpu 1 -ll:othr 8 -ll:okindhack -ll:io 1 -ll:util 2 -ll:dma 2 | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd
