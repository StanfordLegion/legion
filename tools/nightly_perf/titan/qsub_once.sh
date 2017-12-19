#!/bin/bash
#PBS -A CSC103
#PBS -l walltime=01:00:00
#PBS -l nodes=1
#PBS -m abe

cd $PBS_O_WORKDIR

git pull --ff-only

export PERF_MIN_NODES=1
export PERF_MAX_NODES=1

./nightly.sh
