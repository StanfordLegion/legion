# Stage-2 downgrade-protocol verification: long runs for sapling

Self-contained: Downgrade.tla, five .cfg files, tla2tools.jar, and the
SLURM scripts. Needs only a JDK (11+) on the node.

    scp -r sapling/ sapling.stanford.edu:tlc-downgrade/
    ssh sapling.stanford.edu
    cd tlc-downgrade && chmod +x submit_long.sh && ./submit_long.sh

Knobs (env vars or edit run_tlc.sbatch): JAVA (jdk path), TLC_XMX
(heap, default 100g), TLC_SCRATCH (state-queue dir, defaults to node-local /tmp; needs 100GB+
free for DowngradeNew/Downgrade4/DowngradeBig -- check `df -h /tmp` on a
node; if /tmp is a small RAM-backed tmpfs, point TLC_SCRATCH at whatever
local SSD mount the nodes have instead. Avoid NFS /scratch for the queue. Adjust --cpus-per-task and --partition to the
cluster layout.

The eight runs and their expectations:
  DowngradeNew           - full design, two levels: PASS expected
  BisectReceiptChecks    - receipts off: open question (straggler handling)
  BisectRegistrationGate - registration gate off: open question
  Downgrade4             - 4 nodes, multi-child relay: PASS expected
  DowngradeBig           - larger budgets: PASS expected
  BisectFlagVeto         - flag veto off: VIOLATION expected (count cancellation)
  BisectCoveredByHandle  - covering off: VIOLATION expected (zombie replica)
  BisectOwnershipVersioning - versioning off: VIOLATION expected (resurrection)

Each job's verdict is the last lines of its tlc-<jobid>-*.out and in
result-<Config>.log ("No error has been found" = exhaustive PASS;
"Error: Invariant X is violated" = counterexample, with the full trace
in the log). Copy the result-*.log files back for the record.

## Post-matrix follow-up
| DowngradeMid | run_tlc_mid.sbatch | PASS expected | packs=3 (F10 budget), rounds=4; replaces disk-capped DowngradeBig; ~1 day est. |
