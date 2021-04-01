#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

nodes=$SLURM_JOB_NUM_NODES
power=$(echo "l($nodes)/l(2)" | bc -l | xargs printf '%.0f\n')

if [[ ! -d dcr_idx ]]; then mkdir dcr_idx; fi
pushd dcr_idx

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.idx" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 1 -dm:same_address_space -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d dcr_noidx ]]; then mkdir dcr_noidx; fi
pushd dcr_noidx

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.noidx" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 1 -dm:same_address_space -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d nodcr_idx ]]; then mkdir nodcr_idx; fi
pushd nodcr_idx

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.idx" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 0 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d nodcr_noidx ]]; then mkdir nodcr_noidx; fi
pushd nodcr_noidx

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"" ($n = $nx * $ny)..."
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/stencil.noidx" -nx $(( nx * 30000 )) -ny $(( ny * 30000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 0 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd
