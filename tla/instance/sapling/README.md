# InstanceGC sapling matrix (PhysicalManager instance-collection protocol)

Bundle: `InstanceGC.tla` + configs + `tla2tools.jar` (v2.19) +
`run_tlc.sbatch` (hardened: `-fpmem 0.75`, `-checkpoint 60`, disk
guard on node-local /tmp) + `submit_long.sh`.

All configs run the NEW DESIGN baseline (F1 record contract, F2
SeparateAccums, F3 SpawnPoison, F4 WaiterFix) unless a toggle is
bisected off. The flat-topology matrix already completed locally (see
../RESULTS.md): flat exhaustive PASS, all five mechanism bisections
violate as required, master regressions reproduce F2.

| Job | Expectation |
|---|---|
| InstanceNewTree | PASS: 4 nodes, collective tree {n0,n1,n2} + 1 spawnable. Local 10-min probe reached 171M generated / 38.8M distinct at depth 21, queue still growing. |
| InstanceNewBig | PASS: flat 3 nodes, budgets acqs/packs/rounds = 3, spawns/users = 2 (larger budgets are load-bearing, per the downgrade F10 lesson). |
| BisectTreeWaiterFix | VIOLATION expected (SafeDeletion): F4 necessity at tree scope. |
| BisectTreeSpawnPoison | VIOLATION expected (SafeDeletion): F3 necessity at tree scope. |

Verdicts land in `result-<Config>.log`; the SLURM `.out` tail has a
one-line summary. An expected-PASS job that violates is a finding:
bring the trace back for decoding. A DISK GUARD line means
bounded-clean at the last Progress line, not a pass.

## Closing pass (2026-08-23)
NOTE: InstanceGC.tla updated (liveness properties added) -- re-copy the
spec along with the new configs. Submit with ./submit_closing.sh.

| Job | Expectation |
|---|---|
| InstanceNewTree4 | PASS: depth-2 tree {n0..n3}, the config that actually exercises the interior-node subtree-skip (a VALID interior node fails the round without forwarding; its child goes unasked). Local 14-min probe: 149M generated / 31.8M distinct at depth 21, queue growing. |
| InstanceNewMid2 | PASS: flat, packs=3 + spawns=2 (the full large-pack budget; the spawns=1 variant already passed exhaustively locally). Local probe: 277M / 59.3M at depth 26, queue ~10M. |

## F1-correction reruns (2026-08-23)
The model's user-recording semantics were corrected (covering
references, pinned forwards; see FINDINGS.md "F1 CORRECTED"). All
users>0 verdicts from the previous spec are invalidated. Re-copy
InstanceGC.tla and ALL configs, then rerun:
  for cfg in InstanceNewTree InstanceNewTree4 InstanceNewMid2 \
             BisectTreeWaiterFix BisectTreeSpawnPoison; do
    sbatch run_tlc.sbatch $cfg
  done
Expectations unchanged: the first three PASS, the two bisections
violate SafeDeletion.
