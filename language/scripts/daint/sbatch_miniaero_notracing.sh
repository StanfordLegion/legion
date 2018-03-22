#!/bin/sh
#SBATCH --nodes=64
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

if [[ ! -d notracing ]]; then mkdir notracing; fi
pushd notracing

for n in 64 32 16; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."

	    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd8" -blocks $(( n * 8 )) -mesh 256x512x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 8 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:csize 20480 -ll:rsize 1024 -ll:gsize 0 -lg:no_physical_tracing | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd
