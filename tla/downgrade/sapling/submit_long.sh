#!/bin/bash
# Submit the five long Stage-2 verification runs, one node each.
for cfg in DowngradeNew BisectReceiptChecks BisectRegistrationGate \
           BisectFlagVeto BisectCoveredByHandle BisectOwnershipVersioning \
           Downgrade4 DowngradeBig; do
  sbatch run_tlc.sbatch "$cfg"
done
squeue -u "$USER"
