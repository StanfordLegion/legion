#!/bin/bash
# Usage: sbatch run_tlc.sbatch <ConfigName>   (e.g. sbatch run_tlc.sbatch DowngradeNew)
#SBATCH --job-name=tlc-downgrade
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40        # adjust to the node's core count
#SBATCH --mem=0                   # all of the node's memory
#SBATCH --time=48:00:00
#SBATCH --output=tlc-%j-%x.out
# #SBATCH --partition=cpu         # uncomment/adjust for sapling's partition names

set -u
CONFIG=$1
cd "$SLURM_SUBMIT_DIR"

# Java: use a module if sapling provides one, else a JDK on PATH (11+)
# module load java || true
JAVA=${JAVA:-java}

# Node-local disk for TLC's on-disk state queue (the thing that
# exhausted the workstation). /tmp is node-local on sapling; /scratch is
# slow NFS -- do not use it for the queue.
SCRATCH=${TLC_SCRATCH:-/tmp/$USER-tlc-$SLURM_JOB_ID}
# Reclaim scratch leaked by earlier jobs on this node (wedged JVMs, or
# scancel killing the script before its cleanup line). Safe: the node
# is exclusively ours, so no concurrent job of ours can be using it.
rm -rf /tmp/"$USER"-tlc-* 2>/dev/null || true
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH"; exit 1; }
# Clean up even on failure or timeout (slurm sends TERM before KILL)
trap 'rm -rf "$SCRATCH"' EXIT TERM

# Heap: most of the node; TLC keeps the fingerprint set in memory.
XMX=${TLC_XMX:-100g}

echo "=== $CONFIG on $(hostname), workers=$SLURM_CPUS_PER_TASK, scratch=$SCRATCH, xmx=$XMX ==="
"$JAVA" -XX:+UseParallelGC -Xmx"$XMX" -cp tla2tools.jar tlc2.TLC \
    -deadlock -gzip -workers "$SLURM_CPUS_PER_TASK" -checkpoint 30 \
    -metadir "$SCRATCH" -config "$CONFIG.cfg" Downgrade.tla \
    2>&1 | tee "result-$CONFIG.log"

# summarize the verdict at the end of the SLURM log
grep -E "No error has been found|^Error: (Invariant|Temporal)" "result-$CONFIG.log" | head -2
rm -rf "$SCRATCH"
