#!/bin/sh
#MSUB -l nodes=16
#MSUB -l walltime=1:00:00
#MSUB -q pbatch
#MSUB -m abe

export OMPI_MCA_mtl="^psm,psm2"

export LD_LIBRARY_PATH="."

export GASNET_NETWORKDEPTH=64
export GASNET_NETWORKDEPTH_TOTAL=384

echo "Running 1x2x12..."
srun -n 2 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long1x30/leblanc.pnt -npieces 24 -numpcx 1 -numpcy 24 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_1x2x12_%.log | tee out_1x2x12.log

echo "Running 2x2x12..."
srun -n 4 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long2x30/leblanc.pnt -npieces 48 -numpcx 1 -numpcy 48 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_2x2x12_%.log | tee out_2x2x12.log

echo "Running 4x2x12..."
srun -n 8 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long4x30/leblanc.pnt -npieces 96 -numpcx 1 -numpcy 96 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_4x2x12_%.log | tee out_4x2x12.log

echo "Running 8x2x12..."
srun -n 16 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long8x30/leblanc.pnt -npieces 192 -numpcx 1 -numpcy 192 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_8x2x12_%.log | tee out_8x2x12.log

echo "Running 16x2x12..."
srun -n 32 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long16x30/leblanc.pnt -npieces 384 -numpcx 1 -numpcy 384 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_16x2x12_%.log | tee out_16x2x12.log

echo "Running 32x2x12..."
srun -n 64 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long32x30/leblanc.pnt -npieces 768 -numpcx 1 -numpcy 768 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_32x2x12_%.log | tee out_32x2x12.log

echo "Running 64x2x12..."
srun -n 128 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long64x30/leblanc.pnt -npieces 1536 -numpcx 1 -numpcy 1536 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_64x2x12_%.log | tee out_64x2x12.log

echo "Running 128x2x12..."
srun -n 256 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long128x30/leblanc.pnt -npieces 6144 -numpcx 1 -numpcy 6144 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_128x2x12_%.log | tee out_128x2x12.log

echo "Running 256x2x12..."
srun -n 512 --ntasks-per-socket 1 --cpu_bind socket --mpibind=off ./pennant.spmd12 ./pennant.tests/leblanc_long256x30/leblanc.pnt -npieces 6144 -numpcx 1 -numpcy 6144 -seq_init 0 -par_init 1 -ll:cpu 13 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 0 -ll:gsize 0 -hl:prof 1024 -logfile prof_256x2x12_%.log | tee out_256x2x12.log
