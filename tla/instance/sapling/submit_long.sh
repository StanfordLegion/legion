#!/bin/bash
# Submit the InstanceGC long-running verification matrix on sapling.
# Copy this directory to sapling and run: ./submit_long.sh
set -eu
for cfg in InstanceNewTree InstanceNewBig BisectTreeWaiterFix \
           BisectTreeSpawnPoison; do
  sbatch run_tlc.sbatch "$cfg"
done
squeue -u "$USER"
