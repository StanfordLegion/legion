#!/bin/bash
#PBS -l select=64:ncpus=32
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -m abe
#PBS -M EMAIL_ADDRESS
#PBS -A ACCOUNT_ID

cd "$PBS_O_WORKDIR"

echo "Running 1x2x12..."
time LD_LIBRARY_PATH="." aprun -n2 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long1x30/leblanc.pnt -npieces 24 -numpcx 1 -numpcy 24 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_1x2x12_%.log | tee out_1x2x12.log

echo "Running 2x2x12..."
time LD_LIBRARY_PATH="." aprun -n4 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long2x30/leblanc.pnt -npieces 48 -numpcx 1 -numpcy 48 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_2x2x12_%.log | tee out_2x2x12.log

echo "Running 4x2x12..."
time LD_LIBRARY_PATH="." aprun -n8 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long4x30/leblanc.pnt -npieces 96 -numpcx 1 -numpcy 96 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_4x2x12_%.log | tee out_4x2x12.log

echo "Running 8x2x12..."
time LD_LIBRARY_PATH="." aprun -n16 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long8x30/leblanc.pnt -npieces 192 -numpcx 1 -numpcy 192 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_8x2x12_%.log | tee out_8x2x12.log

echo "Running 16x2x12..."
time LD_LIBRARY_PATH="." aprun -n32 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long16x30/leblanc.pnt -npieces 384 -numpcx 1 -numpcy 384 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_16x2x12_%.log | tee out_16x2x12.log

echo "Running 32x2x12..."
time LD_LIBRARY_PATH="." aprun -n64 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long32x30/leblanc.pnt -npieces 768 -numpcx 1 -numpcy 768 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_32x2x12_%.log | tee out_32x2x12.log

echo "Running 64x2x12..."
time LD_LIBRARY_PATH="." aprun -n128 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long64x30/leblanc.pnt -npieces 1536 -numpcx 1 -numpcy 1536 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_64x2x12_%.log | tee out_64x2x12.log

echo "Running 256x2x12..."
time LD_LIBRARY_PATH="." aprun -n512 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./pennant.spmd12 ./pennant.tests/leblanc_long256x30/leblanc.pnt -npieces 6144 -numpcx 1 -numpcy 6144 -seq_init 0 -par_init 1 -hl:sched 256 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_256x2x12_%.log | tee out_256x2x12.log
