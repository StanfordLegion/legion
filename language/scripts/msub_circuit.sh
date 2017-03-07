#!/bin/sh
#MSUB -l nodes=16
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

export OMPI_MCA_mtl="^psm,psm2"

export LD_LIBRARY_PATH="."

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

echo "Running 1x20..."
srun -n 1 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 20 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_1x20_%.log | tee out_1x20.log

echo "Running 2x20..."
srun -n 2 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 40 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_2x20_%.log | tee out_2x20.log

echo "Running 4x20..."
srun -n 4 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 80 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_4x20_%.log | tee out_4x20.log

echo "Running 8x20..."
srun -n 8 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 160 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_8x20_%.log | tee out_8x20.log

echo "Running 16x20..."
srun -n 16 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_16x20_%.log | tee out_16x20.log

echo "Running 32x20..."
srun -n 32 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_32x20_%.log | tee out_32x20.log

echo "Running 64x20..."
srun -n 64 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_64x20_%.log | tee out_64x20.log

echo "Running 128x20..."
srun -n 128 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_128x20_%.log | tee out_128x20.log

echo "Running 256x20..."
srun -n 256 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_256x20_%.log | tee out_256x20.log

echo "Running 512x20..."
srun -n 512 --ntasks-per-node 1 --cpu_bind none --mpibind=off ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 320 -ll:gsize 0 -ll:csize 8192 -hl:sched 512 -ll:cpu 21 -ll:dma 2 -hl:prof 512 -level legion_prof=2 -logfile prof_512x20_%.log | tee out_512x20.log
