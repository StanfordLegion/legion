#!/bin/sh
#MSUB -l nodes=16
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

export OMPI_MCA_mtl="^psm,psm2"

export LD_LIBRARY_PATH="."

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

for n in 1 2 4 8 16; do
    echo "Running $n""x2x16..."
    srun -n $(( $n * 2 )) -N $n --ntasks-per-socket 1 --cpu_bind socket --mpibind=on ./pennant.spmd16 ./pennant.tests/leblanc_long$(( $n * 4 ))x30/leblanc.pnt -npieces $(( $n * 2 * 16 )) -numpcx 1 -numpcy $(( $n * 2 * 16 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 16 -ll:util 1 -ll:dma 1 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:sched 1024 -hl:prof 1024 -logfile prof_"$n"x2x16_%.log | tee out_"$n"x2x16.log
done
