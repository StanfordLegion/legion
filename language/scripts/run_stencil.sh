#!/bin/bash
#PBS -l select=512:ncpus=32
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -m abe
#PBS -M EMAIL
#PBS -A PROJECT

cd "$PBS_O_WORKDIR"

# Problem size: 40000^2 per node

mkdir spmd14
pushd spmd14

echo "Running 1x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 1 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 1 * 40000 )) -ny $(( 1 * 40000 )) -ntx $(( 1 * 4 )) -nty $(( 1 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_1x2x14_%.log | tee out_1x2x14.log

echo "Running 2x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 2 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 2 * 40000 )) -ny $(( 1 * 40000 )) -ntx $(( 2 * 4 )) -nty $(( 1 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_2x2x14_%.log | tee out_2x2x14.log

echo "Running 4x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 4 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 2 * 40000 )) -ny $(( 2 * 40000 )) -ntx $(( 2 * 4 )) -nty $(( 2 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_4x2x14_%.log | tee out_4x2x14.log

echo "Running 8x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 8 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 4 * 40000 )) -ny $(( 2 * 40000 )) -ntx $(( 4 * 4 )) -nty $(( 2 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_8x2x14_%.log | tee out_8x2x14.log

echo "Running 16x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 16 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 4 * 40000 )) -ny $(( 4 * 40000 )) -ntx $(( 4 * 4 )) -nty $(( 4 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_16x2x14_%.log | tee out_16x2x14.log

echo "Running 32x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 32 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 8 * 40000 )) -ny $(( 4 * 40000 )) -ntx $(( 8 * 4 )) -nty $(( 4 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_32x2x14_%.log | tee out_32x2x14.log

echo "Running 64x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 64 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 8 * 40000 )) -ny $(( 8 * 40000 )) -ntx $(( 8 * 4 )) -nty $(( 8 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_64x2x14_%.log | tee out_64x2x14.log

echo "Running 128x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 128 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 16 * 40000 )) -ny $(( 8 * 40000 )) -ntx $(( 16 * 4 )) -nty $(( 8 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_128x2x14_%.log | tee out_128x2x14.log

echo "Running 256x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 256 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 16 * 40000 )) -ny $(( 16 * 40000 )) -ntx $(( 16 * 4 )) -nty $(( 16 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_256x2x14_%.log | tee out_256x2x14.log

echo "Running 512x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 512 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.spmd14 -nx $(( 32 * 40000 )) -ny $(( 16 * 40000 )) -ntx $(( 32 * 4 )) -nty $(( 16 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_512x2x14_%.log | tee out_512x2x14.log

popd



mkdir prk
pushd prk

time aprun -n$(( 1 * 32 )) -cc cpu ../../prk/MPI1/Stencil/stencil 50 $(( 1 * 40000 )) | tee prk_1x32.log

time aprun -n$(( 4 * 32 )) -cc cpu ../../prk/MPI1/Stencil/stencil 50 $(( 2 * 40000 )) | tee prk_4x32.log

time aprun -n$(( 16 * 32 )) -cc cpu ../../prk/MPI1/Stencil/stencil 50 $(( 4 * 40000 )) | tee prk_16x32.log

time aprun -n$(( 64 * 32 )) -cc cpu ../../prk/MPI1/Stencil/stencil 50 $(( 8 * 40000 )) | tee prk_64x32.log

time aprun -n$(( 256 * 32 )) -cc cpu ../../prk/MPI1/Stencil/stencil 50 $(( 16 * 40000 )) | tee prk_256x32.log

popd



mkdir none
pushd none

echo "Running 1x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 1 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 1 * 40000 )) -ny $(( 1 * 40000 )) -ntx $(( 1 * 4 )) -nty $(( 1 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_1x2x14_%.log | tee out_1x2x14.log

echo "Running 2x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 2 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 2 * 40000 )) -ny $(( 1 * 40000 )) -ntx $(( 2 * 4 )) -nty $(( 1 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_2x2x14_%.log | tee out_2x2x14.log

echo "Running 4x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 4 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 2 * 40000 )) -ny $(( 2 * 40000 )) -ntx $(( 2 * 4 )) -nty $(( 2 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_4x2x14_%.log | tee out_4x2x14.log

echo "Running 8x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 8 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 4 * 40000 )) -ny $(( 2 * 40000 )) -ntx $(( 4 * 4 )) -nty $(( 2 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_8x2x14_%.log | tee out_8x2x14.log

echo "Running 16x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 16 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 4 * 40000 )) -ny $(( 4 * 40000 )) -ntx $(( 4 * 4 )) -nty $(( 4 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_16x2x14_%.log | tee out_16x2x14.log

echo "Running 32x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 32 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 8 * 40000 )) -ny $(( 4 * 40000 )) -ntx $(( 8 * 4 )) -nty $(( 4 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_32x2x14_%.log | tee out_32x2x14.log

echo "Running 64x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 64 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 8 * 40000 )) -ny $(( 8 * 40000 )) -ntx $(( 8 * 4 )) -nty $(( 8 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_64x2x14_%.log | tee out_64x2x14.log

echo "Running 128x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 128 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 16 * 40000 )) -ny $(( 8 * 40000 )) -ntx $(( 16 * 4 )) -nty $(( 8 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_128x2x14_%.log | tee out_128x2x14.log

echo "Running 256x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 256 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 16 * 40000 )) -ny $(( 16 * 40000 )) -ntx $(( 16 * 4 )) -nty $(( 16 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_256x2x14_%.log | tee out_256x2x14.log

echo "Running 512x2x14..."
time LD_LIBRARY_PATH=".." aprun -n$(( 512 * 2 )) -S1 -cc numa_node -ss -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ../stencil.none -nx $(( 32 * 40000 )) -ny $(( 16 * 40000 )) -ntx $(( 32 * 4 )) -nty $(( 16 * 7 )) -tsteps 60 -tprune 5 -hl:sched 1024 -ll:cpu 14 -ll:util 1 -ll:dma 2 -ll:csize 32768 -ll:rsize 512 -ll:gsize 0 -hl:prof 1024 -level legion_prof=2 -logfile prof_512x2x14_%.log | tee out_512x2x14.log

popd
