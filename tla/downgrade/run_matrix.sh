#!/bin/zsh
# Overnight model-checking matrix for the downgrade protocol.
# Run from tla/downgrade:  nohup ./run_matrix.sh > run_matrix.log 2>&1 &
# Results accumulate incrementally in RESULTS.md; raw TLC output in log-*.out.
set -u
cd "$(dirname "$0")"
JAVA=/opt/homebrew/opt/openjdk/bin/java
JAR=../tla2tools.jar
R=RESULTS.md

note() { echo "$@" >> $R; }

run_one() {  # <name> <cfg> <expectation> [extra tlc args...]
  local name=$1 cfg=$2 expect=$3; shift 3
  note "## $name  ($(date '+%H:%M:%S'))"
  note "config: $cfg — expectation: $expect"
  rm -rf "states-$name"
  $JAVA -XX:+UseParallelGC -Xmx10g -cp $JAR tlc2.TLC -deadlock -workers 10 \
        -metadir "states-$name" "$@" -config "$cfg" Downgrade.tla \
        > "log-$name.out" 2>&1
  local viol; viol=$(grep -m1 -E "^Error: (Invariant .* is violated|Temporal properties were violated)" "log-$name.out" || true)
  if [ -n "$viol" ]; then
    local tlen; tlen=$(grep -c '^State' "log-$name.out" || true)
    note "**VIOLATION**: $viol (trace length $tlen)"
  else
    if grep -q "No error has been found" "log-$name.out"; then
      note "**PASS** (exhaustive, no violation)"
    else
      note "**INCONCLUSIVE** (run did not complete cleanly — see log-$name.out)"
    fi
  fi
  grep -m1 "depth of the complete state graph" "log-$name.out" >> $R || true
  tail -20 "log-$name.out" | grep -m1 "states generated" >> $R || true
  grep -m1 "^Finished in" "log-$name.out" >> $R || true
  note ""
  rm -rf "states-$name"
}

echo "# Downgrade protocol model-checking matrix — $(date)" > $R
note ""

# ---- Phase 0: wait for / finish the exhaustive New-config safety run ----
note "## safety-exhaustive (DowngradeNew.cfg, 3 nodes, symmetric)"
while pgrep -f "tlc2.TLC.*DowngradeNew.cfg" > /dev/null; do sleep 60; done
if grep -q "No error has been found" tlc-safety-sym.log 2>/dev/null; then
  note "**PASS** (completed by the in-flight run)"
  grep -m1 "depth of the complete state graph" tlc-safety-sym.log >> $R || true
  tail -20 tlc-safety-sym.log | grep -m1 "states generated" >> $R || true
elif grep -qE "^Error: Invariant" tlc-safety-sym.log 2>/dev/null; then
  note "**VIOLATION** in the in-flight run — see tlc-safety-sym.log"
else
  # recover from the last checkpoint, looping until complete
  note "(in-flight run interrupted; recovering from checkpoints)"
  for attempt in 1 2 3 4 5 6 7 8; do
    ckpt=$(ls -dt states-safety/*/ 2>/dev/null | head -1)
    [ -z "$ckpt" ] && break
    $JAVA -XX:+UseParallelGC -Xmx10g -cp $JAR tlc2.TLC -deadlock -workers 10 \
          -metadir states-safety -recover "${ckpt%/}" -checkpoint 5 \
          -config DowngradeNew.cfg Downgrade.tla > tlc-safety-sym.log 2>&1
    grep -qE "No error has been found|^Error:" tlc-safety-sym.log && break
  done
  if grep -q "No error has been found" tlc-safety-sym.log; then
    note "**PASS** (exhaustive after recovery)"
  else
    viol=$(grep -m1 -E "^Error: Invariant" tlc-safety-sym.log || true)
    [ -n "$viol" ] && note "**VIOLATION**: $viol" || note "**INCONCLUSIVE**"
  fi
  grep -m1 "depth of the complete state graph" tlc-safety-sym.log >> $R || true
  tail -20 tlc-safety-sym.log | grep -m1 "states generated" >> $R || true
fi
note ""

# ---- Phase 1: safety bisections (symmetric, full workers) ----
run_one bisect-FlagVeto        BisectFlagVeto.cfg        "violation: F3 count cancellation (DeadOwnerClean)"
run_one bisect-CoveredByHandle BisectCoveredByHandle.cfg "violation: zombie by-handle replica (DeadOwnerClean)"
run_one bisect-RegistrationGate BisectRegistrationGate.cfg "unknown: gate may be redundant given nudge+records"
run_one bisect-OwnershipVersioning BisectOwnershipVersioning.cfg "violation: update resurrection/clobber (DiedStaysDead/OwnerUnique)"
run_one bisect-RoundTagging    BisectRoundTagging.cfg    "unknown: rid staleness may be subsumed by clocks in Stage 1"
run_one bisect-ReceiptChecks   BisectReceiptChecks.cfg   "expected redundant in Stage 1 (single level => at most one commit); load-bearing at Stage 2"

# ---- Phase 2: liveness bisections (NO symmetry) ----
run_one bisectlive-RestartForwarding BisectLiveRestartForwarding.cfg "violation: dropped restarts leak (EventualCollection)"
run_one bisectlive-RegHandshake      BisectLiveRegHandshake.cfg      "violation: registration wedge / no nudge (EventualCollection)"
run_one bisectlive-OwnershipVersioning BisectLiveOwnershipVersioning.cfg "violation: ownership clobber leak (EventualCollection)"
run_one bisectlive-RegistrationGate  BisectLiveRegistrationGate.cfg  "unknown"

# ---- Phase 3: old-protocol liveness (model validation) ----
run_one old-liveness DowngradeOldLive.cfg "violation expected: current protocol leaks"

# ---- Phase 4: bigger configurations (may run for hours; last on purpose) ----
run_one safety-4node Downgrade4.cfg "pass expected: relay aggregation with >=2 children"
run_one safety-bigbudget DowngradeBig.cfg "pass expected: larger reference/round budgets"

note "---"
note "matrix complete at $(date)"
