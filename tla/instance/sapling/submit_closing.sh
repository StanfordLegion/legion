#!/bin/bash
# Closing-pass jobs for the InstanceGC campaign (2026-08-23).
set -eu
for cfg in InstanceNewTree4 InstanceNewMid2; do
  sbatch run_tlc.sbatch "$cfg"
done
squeue -u "$USER"
