#!/bin/sh
#SBATCH -A d108
#SBATCH -p debug
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu

for (( i=0; i<=4; i++ )) do
  srun --cpu-bind=none python3 ~/legion/language/regent.py self-check.rg -ll:cpu 10 -ll:csize 4096
done
