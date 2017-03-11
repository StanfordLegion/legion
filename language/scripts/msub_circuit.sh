#!/bin/sh
#MSUB -l nodes=128
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

# export OMPI_MCA_mtl="^psm,psm2"

export LD_LIBRARY_PATH="."

export GASNET_SPAWNER=pmi

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

for n in 1 2 4 8 16 32 64 128; do
    if [[ ! -f out_"$n"x32.log ]]; then
	echo "Running $n""x32..."
	srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd32 -npp 2500 -wpp 10000 -l 10 -p $(( $n * 32 )) -ll:gsize 0 -ll:csize 8192 -ll:cpu 33 -ll:util 2 -ll:dma 2 -hl:sched 1024 -hl:prof 8 -level legion_prof=2 -logfile prof_"$n"x32_%.log | tee out_"$n"x32.log
    fi
done
