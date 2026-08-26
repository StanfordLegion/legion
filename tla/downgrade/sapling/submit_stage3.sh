#!/bin/bash
# Stage-3 verification, round 2 (F16 fix: RegClockBump). Expectations:
#   DowngradeTree4     - depth-2 tree, fix ON:  PASS (was StrictLevelOrder
#                        violation in job 77779 with the fix off)
#   DowngradeTreeSmoke - star tree + probes, fix ON: PASS
#   DowngradeTreeMid   - star tree + 2 outsiders, fix ON: PASS
#   DowngradeNew       - flat exhaustive with the bump enabled: PASS
# BisectF16 (fix OFF) is intentionally NOT submitted: job 77779 already
# is that verdict (RegClockBump=FALSE is provably identical to the
# pre-fix spec). Submit it manually only if you want the re-confirmation:
#   sbatch run_stage3.sbatch BisectF16
cd "$(dirname "$0")"
for c in DowngradeTree4 DowngradeTreeSmoke DowngradeTreeMid DowngradeNew; do
  sbatch run_stage3.sbatch $c
done
