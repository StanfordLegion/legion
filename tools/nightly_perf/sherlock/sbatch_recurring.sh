#!/bin/bash
#SBATCH --job-name=nightly-perf
#SBATCH --dependency=singleton
#SBATCH --begin=now+1days
#SBATCH --partition=aaiken
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL

srun ./nightly.sh

## Resubmit the job for the next execution
sbatch $0
