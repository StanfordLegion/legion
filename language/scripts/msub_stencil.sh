#!/bin/bash
#MSUB -l nodes=132
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

NODES=128


root_dir="$PWD"

export LD_LIBRARY_PATH="$root_dir"

export GASNET_SPAWNER=pmi
# export GASNET_NETWORKDEPTH=64
# export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

mkdir timing
pushd timing

for (( i = 0; i < SLURM_JOB_NUM_NODES; i++ )); do
    n=1
    nx=1
    ny=1
    OMP_NUM_THREADS=36 srun --relative $i -n $(( n * 2 )) -N $n --output=timing.%N.log "$root_dir/stencil.spmd16" -nx $(( nx * 80000 )) -ny $(( ny * 80000 )) -ntx $(( nx * 4 )) -nty $(( ny * 8 )) -tsteps 60 -tprune 5 -hl:sched -1 -ll:cpu 17 -ll:util 1 -ll:dma 2 -ll:csize 50000 -ll:rsize 512 -ll:gsize 0 &

    if (( i % 128 == 127 )); then wait; fi
done
wait

"$root_dir/summarize.py" timing.*.log | grep -v ERROR | sort -n -k 4 | cut -d. -f2 > nodelist.txt
head -n $NODES nodelist.txt | sort -n > nodelist_$NODES.txt

popd

if [[ ! -d unfiltered ]]; then mkdir unfiltered; fi
pushd unfiltered

for i in 7 6 5 4 3 2 1 0; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+1)/2) ))
    ny=$(( 2 ** (i/2) ))
    if [[ ! -f out_"$n"x2x16.log ]]; then
        echo "Running $n""x2x16..."
        OMP_NUM_THREADS=36 srun -n $(( n * 2 )) -N $n "$root_dir/stencil.spmd16" -nx $(( nx * 80000 )) -ny $(( ny * 80000 )) -ntx $(( nx * 4 )) -nty $(( ny * 8 )) -tsteps 60 -tprune 5 -hl:sched -1 -ll:cpu 17 -ll:util 1 -ll:dma 2 -ll:csize 50000 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_"$n"x2x16_%.log | tee out_"$n"x2x16.log
    fi
done

popd

if [[ ! -d sorted ]]; then mkdir sorted; fi
pushd sorted

for i in 7 6 5 4 3 2 1 0; do
    n=$(( 2 ** i))
    nx=$(( 2 ** ((i+1)/2) ))
    ny=$(( 2 ** (i/2) ))
    if [[ ! -f out_"$n"x2x16.log ]]; then
        echo "Running $n""x2x16..."
        OMP_NUM_THREADS=36 srun --nodelist "$root_dir/timing/nodelist_$NODES.txt" -n $(( n * 2 )) -N $n "$root_dir/stencil.spmd16" -nx $(( nx * 80000 )) -ny $(( ny * 80000 )) -ntx $(( nx * 4 )) -nty $(( ny * 8 )) -tsteps 60 -tprune 5 -hl:sched -1 -ll:cpu 17 -ll:util 1 -ll:dma 2 -ll:csize 50000 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_"$n"x2x16_%.log | tee out_"$n"x2x16.log
    fi
done

popd
