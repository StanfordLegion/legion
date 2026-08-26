#!/bin/zsh
# Self-resuming model-checking matrix driver (v2).
# Safe to relaunch any number of times: completed runs are skipped via
# done-<name> markers; an interrupted run resumes from its TLC checkpoint.
set -u
cd "$(dirname "$0")"
JAVA=/opt/homebrew/opt/openjdk/bin/java
JAR=../tla2tools.jar
R=RESULTS.md

# single-instance guard
if [ -f driver.lock ] && kill -0 "$(cat driver.lock)" 2>/dev/null; then
  echo "driver already running (pid $(cat driver.lock)); exiting"
  exit 0
fi
echo $$ > driver.lock

note() { echo "$@" >> $R; }

[ -f $R ] || { echo "# Downgrade protocol v2 matrix — $(date)" > $R; echo >> $R; }

run_one() {  # <name> <cfg> <expectation>
  local name=$1 cfg=$2 expect=$3
  [ -f "done-$name" ] && { echo "skip $name (done)"; return; }
  echo "run $name ($(date '+%H:%M:%S'))"
  local rec=()
  local ckpt
  ckpt=$(ls -dt states-$name/*/ 2>/dev/null | head -1 || true)
  if [ -n "${ckpt:-}" ] && [ -n "$(ls ${ckpt%/}/*.st.chkpt 2>/dev/null)" ]; then
    rec=(-recover "${ckpt%/}")
    echo "  recovering from ${ckpt%/}"
  else
    rm -rf "states-$name"
  fi
  $JAVA -XX:+UseParallelGC -Xmx8g -cp $JAR tlc2.TLC -deadlock -gzip \
        -workers 10 -checkpoint 2 -metadir "states-$name" "${rec[@]}" \
        -config "$cfg" Downgrade.tla > "log-$name.out" 2>&1
  local viol
  viol=$(grep -m1 -E "^Error: (Invariant .* is violated|Temporal properties were violated)" "log-$name.out" || true)
  if [ -z "$viol" ] && ! grep -q "No error has been found" "log-$name.out"; then
    if grep -qE "TLC threw an unexpected exception|No such file or directory|when writing the disk|when reading the disk" "log-$name.out"; then
      echo "  $name failed on a bad checkpoint/disk error; clearing state for a fresh retry"
      rm -rf "states-$name"
    else
      echo "  $name interrupted; will resume from checkpoint on next launch"
    fi
    grep -m1 -E "No space left" "log-$name.out" >> $R || true
    return
  fi
  note "## $name  ($(date '+%H:%M:%S'))"
  note "config: $cfg — expectation: $expect"
  if [ -n "$viol" ]; then
    local tlen; tlen=$(grep -c '^State' "log-$name.out" || true)
    note "**VIOLATION**: $viol (trace length $tlen)"
  else
    note "**PASS** (exhaustive, no violation)"
  fi
  grep -m1 "depth of the complete state graph" "log-$name.out" >> $R || true
  tail -20 "log-$name.out" | grep -m1 "states generated" >> $R || true
  grep -m1 "^Finished in" "log-$name.out" >> $R || true
  note ""
  touch "done-$name"
  rm -rf "states-$name"
}

( while true; do echo "heartbeat $(date '+%H:%M:%S') $(df -h / | tail -1 | awk '{print $4" free"}')"; sleep 60; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

# ---- fast, high-value first ----
run_one live-base DowngradeNewLive.cfg "pass expected: full design liveness"
run_one bisectlive-RestartForwarding BisectLiveRestartForwarding.cfg "violation: dropped restarts leak"
run_one bisectlive-RegHandshake      BisectLiveRegHandshake.cfg      "violation: registration nudge missing"
run_one bisectlive-OwnershipVersioning BisectLiveOwnershipVersioning.cfg "violation: stale forwarded-restart transfers"
run_one bisectlive-RegistrationGate  BisectLiveRegistrationGate.cfg  "unknown: gate may be redundant"
run_one old-liveness DowngradeOldLive.cfg "violation expected: current protocol leaks"

# ---- base safety, then safety bisections ----
run_one safety-exhaustive DowngradeNew.cfg "pass expected: full design safety"
run_one bisect-FlagVeto        BisectFlagVeto.cfg        "violation: F3 count cancellation"
run_one bisect-CoveredByHandle BisectCoveredByHandle.cfg "violation: uncovered by-handle zombie"
run_one bisect-OwnershipVersioning BisectOwnershipVersioning.cfg "violation: update resurrection"
run_one bisect-ReceiptChecks   BisectReceiptChecks.cfg   "open: receipts vs state-matching for stragglers (Stage 2)"
run_one bisect-RegistrationGate BisectRegistrationGate.cfg "open: gate necessity at Stage 2"

# ---- big configurations last ----
run_one safety-4node Downgrade4.cfg "pass expected: relay aggregation with >=2 children"
run_one safety-bigbudget DowngradeBig.cfg "pass expected: larger budgets"


note "---"
note "matrix pass finished at $(date) (relaunch to resume anything interrupted)"
echo "driver finished $(date)"
