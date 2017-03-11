#!/bin/bash
#PBS -l select=512:ncpus=32
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -m abe
#PBS -M EMAIL_ADDRESS
#PBS -A ACCOUNT_ID

cd "$PBS_O_WORKDIR"

export LD_LIBRARY_PATH="$PWD"

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

export REALM_BACKTRACE=1

if [[ ! -d spmd14 ]]; then mkdir spmd14; fi
pushd spmd14

for n in 1 2 4 8 16 32 64 128 256 512; do
    if [[ ! -f out_"$n"x2x14.log ]]; then
        echo "Running $n""x2x14..."
        aprun -n$(( $n * 2 )) -S1 -cc numa_node -ss ../pennant.spmd14 ../pennant.tests/leblanc_long$(( $n * 4 ))x30/leblanc.pnt -npieces $(( $n * 2 * 14 )) -numpcx 1 -numpcy $(( $n * 2 * 14 )) -seq_init 0 -par_init 1 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 8 -level legion_prof=2 -logfile prof_"$n"x2x14_%.log | tee out_"$n"x2x14.log
    fi
done

popd
