#!/bin/bash
#MSUB -l nodes=520
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

root_dir="$PWD"

NODES=512

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
    srun --relative $i -n $n -N $n --ntasks-per-node 1 --cpu_bind none --mpibind=off --output=timing.%N.log "$root_dir/circuit.spmd32" -npp 2500 -wpp 10000 -l 30 -p $(( $n * 32 )) -ll:gsize 0 -ll:csize 8192 -ll:cpu 33 -ll:util 2 -ll:dma 2 -hl:sched 1024 &
    if (( i % 64 == 0 )); then wait; fi
done
wait

"$root_dir/../scripts/summarize.py" timing.*.log | grep -v ERROR | sort -n -k 4 | cut -d. -f2 > nodelist.txt
head -n $NODES nodelist.txt | sort -n > nodelist_$NODES.txt

popd

if [[ ! -d spmd32 ]]; then mkdir spmd32; fi
pushd spmd32

for n in 1 2 4 8 16 32 64 128 256 512; do
    if [[ ! -f out_"$n"x32.log ]]; then
        echo "Running $n""x32..."
        srun --nodelist "$root_dir/timing/nodelist_$NODES.txt" -n $n -N $n --ntasks-per-node 1 --cpu_bind none --mpibind=off "$root_dir/circuit.spmd32" -npp 2500 -wpp 10000 -l 30 -p $(( $n * 32 )) -ll:gsize 0 -ll:csize 8192 -ll:cpu 33 -ll:util 2 -ll:dma 2 -hl:sched 1024 -hl:prof 8 -level legion_prof=2 -logfile prof_"$n"x32_%.log | tee out_"$n"x32.log
    fi
done

popd
