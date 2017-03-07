#!/bin/bash
#PBS -l select=512:ncpus=32
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -m abe
#PBS -M EMAIL_ADDRESS
#PBS -A ACCOUNT_ID

cd "$PBS_O_WORKDIR"

echo "Running 1x20..."
time LD_LIBRARY_PATH="." aprun -n1 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 20 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_1x20_%.log | tee out_1x20.log

echo "Running 2x20..."
time LD_LIBRARY_PATH="." aprun -n2 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 40 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_2x20_%.log | tee out_2x20.log

echo "Running 4x20..."
time LD_LIBRARY_PATH="." aprun -n4 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 80 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_4x20_%.log | tee out_4x20.log

echo "Running 8x20..."
time LD_LIBRARY_PATH="." aprun -n8 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 160 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_8x20_%.log | tee out_8x20.log

echo "Running 16x20..."
time LD_LIBRARY_PATH="." aprun -n16 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_16x20_%.log | tee out_16x20.log

echo "Running 32x20..."
time LD_LIBRARY_PATH="." aprun -n32 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 640 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_32x20_%.log | tee out_32x20.log

echo "Running 64x20..."
time LD_LIBRARY_PATH="." aprun -n64 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 1280 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_64x20_%.log | tee out_64x20.log

echo "Running 128x20..."
time LD_LIBRARY_PATH="." aprun -n128 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 2560 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_128x20_%.log | tee out_128x20.log

echo "Running 256x20..."
time LD_LIBRARY_PATH="." aprun -n256 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 5120 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_256x20_%.log | tee out_256x20.log

echo "Running 512x20..."
time LD_LIBRARY_PATH="." aprun -n512 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 10240 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_512x20_%.log | tee out_512x20.log
