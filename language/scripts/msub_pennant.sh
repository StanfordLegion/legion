#!/bin/bash
#MSUB -l nodes=134
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

root_dir="$PWD"

NODES=128

# export OMPI_MCA_mtl="^psm,psm2"

export LD_LIBRARY_PATH="$root_dir"

export GASNET_SPAWNER=pmi

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

mkdir timing
pushd timing

for (( i = 0; i < SLURM_JOB_NUM_NODES; i++ )); do
    n=1
    srun --relative $i -n $(( $n * 2 )) -N $n --ntasks-per-socket 1 --cpu_bind socket --mpibind=on --output=timing.%N.log "$root_dir/pennant.spmd16" "$root_dir/pennant.tests/leblanc_long$(( $n * 16 ))x30/leblanc.pnt" -npieces $(( $n * 2 * 16 )) -numpcx 1 -numpcy $(( $n * 2 * 16 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 17 -ll:util 1 -ll:dma 1 -ll:csize 60000 -ll:rsize 0 -ll:gsize 0 -hl:sched 1024 &
    if (( i % 64 == 0 )); then wait; fi
done
wait

"$root_dir/../scripts/summarize.py" timing.*.log | grep -v ERROR | sort -n -k 4 | cut -d. -f2 > nodelist.txt
head -n $NODES nodelist.txt | sort -n > nodelist_$NODES.txt

popd

if [[ ! -d spmd32 ]]; then mkdir spmd32; fi
pushd spmd32

for n in 128 64 32 16 8 4 2 1; do
    if [[ ! -f out_"$n"x2x16.log ]]; then
        echo "Running $n""x2x16..."
        srun --nodelist "$root_dir/timing/nodelist_$NODES.txt" -n $(( $n * 2 )) -N $n --ntasks-per-socket 1 --cpu_bind socket --mpibind=on "$root_dir/pennant.spmd16" "$root_dir/pennant.tests/leblanc_long$(( $n * 16 ))x30/leblanc.pnt" -npieces $(( $n * 2 * 16 )) -numpcx 1 -numpcy $(( $n * 2 * 16 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 17 -ll:util 1 -ll:dma 1 -ll:csize 60000 -ll:rsize 0 -ll:gsize 0 -hl:sched 1024 -hl:prof 4 -level legion_prof=2 -logfile prof_"$n"x2x16_%.log | tee out_"$n"x2x16.log
    fi
done

popd
