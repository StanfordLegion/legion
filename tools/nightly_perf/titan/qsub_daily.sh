#!/bin/bash
#PBS -A CSC103
#PBS -l walltime=01:00:00
#PBS -l nodes=8
#PBS -m abe

cd $PBS_O_WORKDIR

git pull --ff-only

# Submit the job to run again
qsub -a $(date -d 'now+1day' +%Y%m%d%H%M.%S) qsub_daily.sh

# Run job
export PERF_MIN_NODES=1
export PERF_MAX_NODES=8

./nightly.sh
