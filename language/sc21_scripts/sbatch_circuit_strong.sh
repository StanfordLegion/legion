#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

if [[ ! -d dcr_idx ]]; then mkdir dcr_idx; fi
pushd dcr_idx

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.idx" -npp 2500 -wpp 10000 -l 50 -p $(( 512 )) -pps $(( 512 / n )) -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:pin_util -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5-dm:replicate 1 -dm:same_address_space 1 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d dcr_noidx ]]; then mkdir dcr_noidx; fi
pushd dcr_noidx

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.noidx" -npp 2500 -wpp 10000 -l 50 -p $(( 512 )) -pps $(( 512 / n )) -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:pin_util -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 1 -dm:same_address_space 1 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d nodcr_idx ]]; then mkdir nodcr_idx; fi
pushd nodcr_idx

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.idx" -npp 2500 -wpp 10000 -l 50 -p $(( 512 )) -pps $(( 512 / n )) -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:pin_util -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 0 -dm:same_address_space 0 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d nodcr_noidx ]]; then mkdir nodcr_noidx; fi
pushd nodcr_noidx

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none /lib64/ld-linux-x86-64.so.2 "$root_dir/circuit.noidx" -npp 2500 -wpp 10000 -l 50 -p $(( 512 )) -pps $(( 512 / n )) -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:pin_util -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -level 5 -dm:replicate 0 -dm:same_address_space 0 -dm:memoize -lg:parallel_replay 2 | tee out_"$n"x1_r"$r".log
  done
done

popd
