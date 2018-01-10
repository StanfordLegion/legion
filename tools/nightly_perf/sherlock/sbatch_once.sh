#!/bin/bash
#SBATCH --partition=aaiken
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL

git pull --ff-only

export PERF_MIN_NODES=1
export PERF_MAX_NODES=1

srun ./nightly.sh
