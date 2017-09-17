#!/bin/bash
#SBATCH --partition=aaiken
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL

git pull --ff-only

srun ./nightly.sh
