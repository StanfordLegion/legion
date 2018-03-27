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
        if [[ ! -f out_"$n"x10_r"$r".log ]]; then
            echo "Running $n""x10_r$r""..."
            srun -n "$((n * 2))" -N "$n" -c 12 --ntasks-per-node 2 /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.spmd5" "$root_dir/pennant.tests/leblanc_long16x30/leblanc.pnt" -npieces $(( $n * 10 )) -numpcx 1 -numpcy $(( $n * 10 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 5 -ll:io 1 -ll:util 2 -ll:dma 1 -ll:pin_dma -ll:ht_sharing 0 -ll:csize 20480 -ll:rsize 1024 -ll:gsize 0 | tee out_"$n"x10_r"$r".log
        fi
    done
done

popd
