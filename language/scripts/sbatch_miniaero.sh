#!/bin/sh
#SBATCH --nodes=16
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

if [[ ! -d spmd8 ]]; then mkdir spmd8; fi
pushd spmd8

for n in 1 2 4 8 16; do
    if [[ ! -f out_"$n"x8.log ]]; then
        echo "Running $n""x8..."

	srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd8" -blocks $(( n * 8 )) -mesh 256x$(( n * 512 ))x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched -1 -ll:cpu 9 -ll:util 2 -ll:dma 2 -ll:csize 20480 -ll:rsize 0 -ll:gsize 0 -hl:prof 4 -level legion_prof=2 -logfile prof_"$n"x8_%.log | tee out_"$n"x8.log
    fi
done

popd
