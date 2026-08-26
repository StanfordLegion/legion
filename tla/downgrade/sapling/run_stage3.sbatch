#!/bin/bash
#SBATCH --job-name=tlc-stage3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=tlc-%j-tlc-stage3.out
# Usage: sbatch run_stage3.sbatch <ConfigName>   (e.g. DowngradeTree4)
set -u
CFG=${1:?need a config name, e.g. DowngradeTree4}
SRC=$SLURM_SUBMIT_DIR
# reclaim our own stale scratch from prior runs, then create fresh
rm -rf /tmp/$USER-tlc-* 2>/dev/null
WORK=/tmp/$USER-tlc-$SLURM_JOB_ID
mkdir -p "$WORK" || { echo "FATAL: cannot create $WORK (disk full?)"; df -h /tmp; exit 1; }
trap 'rm -rf "$WORK"' EXIT TERM
# disk guard: refuse to start with <100GB free on /tmp
FREE_KB=$(df -Pk /tmp | awk 'NR==2 {print $4}')
if [ "$FREE_KB" -lt 104857600 ]; then
  echo "DISK GUARD: only $((FREE_KB/1024/1024))GB free on /tmp; clean up first"; exit 1
fi
cp "$SRC/Downgrade.tla" "$SRC/$CFG.cfg" "$SRC/tla2tools.jar" "$WORK/"
cd "$WORK"
java -XX:+UseParallelGC -Xmx220g -Djava.io.tmpdir="$WORK" -cp tla2tools.jar \
  tlc2.TLC -deadlock -workers $SLURM_CPUS_PER_TASK -fpmem 0.75 -gzip \
  -checkpoint 60 -metadir "$WORK/states" -config "$CFG.cfg" Downgrade.tla
