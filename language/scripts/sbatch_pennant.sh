#!/bin/sh
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

if [[ ! -d spmd10 ]]; then mkdir spmd10; fi
pushd spmd10

for n in 1 2; do
    if [[ ! -f out_"$n"x10.log ]]; then
        echo "Running $n""x10..."
        srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/pennant.spmd10" "$root_dir/pennant.tests/leblanc_long$(( $n * 4 ))x30/leblanc.pnt" -npieces $(( $n * 10 )) -numpcx 1 -numpcy $(( $n * 10 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 11 -ll:util 1 -ll:dma 1 -ll:csize 30000 -ll:rsize 0 -ll:gsize 0 -hl:sched -1 -hl:prof 4 -level legion_prof=2 -logfile prof_"$n"x10_%.log | tee out_"$n"x10.log
    fi
done

popd
