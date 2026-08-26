#!/bin/bash
# Usage: sbatch run_tlc_mid.sbatch [ConfigName]   (default: DowngradeMid)
# Hardened variant of run_tlc.sbatch after job 77716 died of ENOSPC:
#  - -fpmem 0.75: keep up to ~75% of heap for the in-memory fingerprint
#    set (TLC's default is 0.25, which is why 77716 started spilling and
#    merging fingerprint files on disk around 3B distinct states).
#  - -checkpoint 60: checkpoints transiently double the queue's disk
#    footprint; halve the frequency to halve the exposure.
#  - disk guard: if node-local scratch drops below 15GB free, kill TLC
#    cleanly so the run ends as a readable bounded-clean verdict at the
#    last Progress line instead of a wedged JVM holding the node.
#SBATCH --job-name=tlc-downgrade
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40        # adjust to the node's core count
#SBATCH --mem=0                   # all of the node's memory
#SBATCH --time=48:00:00
#SBATCH --output=tlc-%j-%x.out
# #SBATCH --partition=cpu         # uncomment/adjust for sapling's partition names

set -u
CONFIG=${1:-DowngradeMid}
cd "$SLURM_SUBMIT_DIR"

# module load java || true
JAVA=${JAVA:-java}

# Node-local disk for TLC's on-disk state queue; /tmp is node-local on
# sapling, /scratch is slow NFS -- do not use it for the queue.
SCRATCH=${TLC_SCRATCH:-/tmp/$USER-tlc-$SLURM_JOB_ID}
# Reclaim scratch leaked by earlier jobs on this node (wedged JVMs, or
# scancel killing the script before its cleanup line). Safe: the node
# is exclusively ours, so no concurrent job of ours can be using it.
rm -rf /tmp/"$USER"-tlc-* 2>/dev/null || true
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH"; exit 1; }
# Clean up even on failure or timeout (slurm sends TERM before KILL)
trap 'rm -rf "$SCRATCH"' EXIT TERM

XMX=${TLC_XMX:-100g}
MIN_FREE_KB=$((15 * 1024 * 1024))   # 15 GB

echo "=== $CONFIG on $(hostname), workers=$SLURM_CPUS_PER_TASK, scratch=$SCRATCH, xmx=$XMX, fpmem=0.75 ==="

# Disk guard (node is exclusive, so pkill on tlc2.TLC is safe).
( while true; do
    avail=$(df -Pk "$SCRATCH" | awk 'NR==2 {print $4}')
    if [ "${avail:-0}" -lt "$MIN_FREE_KB" ]; then
      echo "DISK GUARD: <15GB free on $SCRATCH at $(date); killing TLC." \
           "Treat this run as BOUNDED-CLEAN at the last Progress line." \
           | tee -a "result-$CONFIG.log"
      pkill -9 -f tlc2.TLC
      break
    fi
    sleep 60
  done ) &
GUARD=$!

"$JAVA" -XX:+UseParallelGC -Xmx"$XMX" -cp tla2tools.jar tlc2.TLC \
    -deadlock -gzip -workers "$SLURM_CPUS_PER_TASK" -checkpoint 60 \
    -fpmem 0.75 -metadir "$SCRATCH" -config "$CONFIG.cfg" Downgrade.tla \
    2>&1 | tee "result-$CONFIG.log"

kill "$GUARD" 2>/dev/null

# summarize the verdict at the end of the SLURM log
grep -E "No error has been found|^Error: (Invariant|Temporal)|DISK GUARD" \
    "result-$CONFIG.log" | head -2
rm -rf "$SCRATCH"
