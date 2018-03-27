#!/bin/sh
#SBATCH --nodes=256
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"
export NUM_UTIL=2

if [[ ! -d tracing_util"$NUM_UTIL" ]]; then mkdir tracing_util"$NUM_UTIL"; fi
pushd tracing_util"$NUM_UTIL"

for n in 256 128; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."
            srun -n "$((n * 2))" -N "$n" -c 12 --ntasks-per-node 2 /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd4" -blocks "$(( n * 8 ))" -mesh 8x8x32768 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 4 -ll:io 1 -ll:dma 2 -ll:pin_dma -ll:util "$NUM_UTIL" -ll:ht_sharing 0 -ll:csize 30000 -ll:rsize 4096 -ll:gsize 0 -dm:memoize | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd

if [[ ! -d notracing_util"$NUM_UTIL" ]]; then mkdir notracing_util"$NUM_UTIL"; fi
pushd notracing_util"$NUM_UTIL"

for n in 256 128; do
    for r in 0 1 2 3 4; do
        if [[ ! -f out_"$n"x8_r"$r".log ]]; then
            echo "Running $n""x8_r$r""..."
            srun -n "$((n * 2))" -N "$n" -c 12 --ntasks-per-node 2 /lib64/ld-linux-x86-64.so.2 "$root_dir/miniaero.spmd4" -blocks "$(( n * 8 ))" -mesh 8x8x32768 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 100 -output_frequency 101 -ll:cpu 4 -ll:io 1 -ll:dma 2 -ll:pin_dma -ll:util "$NUM_UTIL" -ll:ht_sharing 0 -ll:csize 30000 -ll:rsize 4096 -ll:gsize 0 | tee out_"$n"x8_r"$r".log
            # -lg:prof 4 -lg:prof_logfile prof_"$n"x8_r"$r"_%.gz
        fi
    done
done

popd
