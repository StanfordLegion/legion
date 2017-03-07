#!/bin/bash
#PBS -l select=512:ncpus=32
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -m abe
#PBS -M EMAIL_ADDRESS
#PBS -A ACCOUNT_ID

cd "$PBS_O_WORKDIR"

echo "Running 1x2x8..."
time LD_LIBRARY_PATH="." aprun -n2 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 16 -mesh 64x2048x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_1x2x8_%.log | tee out_1x2x8.log

echo "Running 2x2x8..."
time LD_LIBRARY_PATH="." aprun -n4 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 32 -mesh 64x4096x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_2x2x8_%.log | tee out_2x2x8.log

echo "Running 4x2x8..."
time LD_LIBRARY_PATH="." aprun -n8 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 64 -mesh 64x8192x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_4x2x8_%.log | tee out_4x2x8.log

echo "Running 8x2x8..."
time LD_LIBRARY_PATH="." aprun -n16 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 128 -mesh 64x16384x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_8x2x8_%.log | tee out_8x2x8.log

echo "Running 16x2x8..."
time LD_LIBRARY_PATH="." aprun -n32 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 256 -mesh 64x32768x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_16x2x8_%.log | tee out_16x2x8.log

echo "Running 32x2x8..."
time LD_LIBRARY_PATH="." aprun -n64 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 512 -mesh 64x65536x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_32x2x8_%.log | tee out_32x2x8.log

echo "Running 64x2x8..."
time LD_LIBRARY_PATH="." aprun -n128 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 1024 -mesh 64x131072x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_64x2x8_%.log | tee out_64x2x8.log

echo "Running 128x2x8..."
time LD_LIBRARY_PATH="." aprun -n256 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 2048 -mesh 64x262144x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_128x2x8_%.log | tee out_128x2x8.log

echo "Running 256x2x8..."
time LD_LIBRARY_PATH="." aprun -n512 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 4096 -mesh 64x524288x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_256x2x8_%.log | tee out_256x2x8.log

echo "Running 512x2x8..."
time LD_LIBRARY_PATH="." aprun -n1024 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 8192 -mesh 64x1048576x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:util 2 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 512 -level legion_prof=2 -logfile prof_512x2x8_%.log | tee out_512x2x8.log
