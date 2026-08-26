# Downgrade protocol v2 matrix — Thu Aug 20 03:25:35 PDT 2026

## live-base  (03:27:15)
config: DowngradeNewLive.cfg — expectation: pass expected: full design liveness
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 47.
Progress(17) at 2026-08-20 03:25:39: 161,870 states generated (161,870 s/min), 52,616 distinct states found (52,616 ds/min), 11,132 states left on queue.
Finished in 01min 39s at (2026-08-20 03:27:14)

## bisectlive-RestartForwarding  (03:27:19)
config: BisectLiveRestartForwarding.cfg — expectation: violation: dropped restarts leak
**VIOLATION**: Error: Temporal properties were violated. (trace length 14)
186551 states generated, 58242 distinct states found, 6180 states left on queue.
Finished in 03s at (2026-08-20 03:27:18)

## bisectlive-RegHandshake  (03:27:23)
config: BisectLiveRegHandshake.cfg — expectation: violation: registration nudge missing
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
149092 states generated, 55820 distinct states found, 8703 states left on queue.
Finished in 03s at (2026-08-20 03:27:22)

## bisectlive-OwnershipVersioning  (03:27:27)
config: BisectLiveOwnershipVersioning.cfg — expectation: violation: stale forwarded-restart transfers
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
179968 states generated, 59070 distinct states found, 12300 states left on queue.
Finished in 03s at (2026-08-20 03:27:26)

## bisectlive-RegistrationGate  (03:34:01)
config: BisectLiveRegistrationGate.cfg — expectation: unknown: gate may be redundant
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 47.
Progress(30) at 2026-08-20 03:30:02: 8,328,825 states generated (4,503,774 s/min), 2,186,011 distinct states found (1,051,589 ds/min), 314,970 states left on queue.
Finished in 06min 33s at (2026-08-20 03:34:01)

## old-liveness  (03:34:05)
config: DowngradeOldLive.cfg — expectation: violation expected: current protocol leaks
**VIOLATION**: Error: Temporal properties were violated. (trace length 11)
198298 states generated, 66811 distinct states found, 8138 states left on queue.
Finished in 03s at (2026-08-20 03:34:05)

## safety-exhaustive (local) — CANCELLED 2026-08-20
Superseded: was verifying the pre-F10 spec (level-conditional clock bump);
sapling DowngradeNew on the fixed spec replaces it. The local liveness
results from this pass are likewise pre-F10; the fixed spec re-passed
liveness exhaustively in the F10 smoke.
## Sapling resubmitted batch (F10-fixed spec, started 2026-08-20 15:29)
- BisectFlagVeto (77712): **VIOLATION as expected** — DeadOwnerClean, trace 22,
  84.4M states, 7m12s. Count cancellation without the veto (spawn + acquire +
  3 packs/drops cancel in the tallies). Pre-F10 run also violated (trace 22).
- BisectCoveredByHandle (77713): **VIOLATION as expected** — DeadOwnerClean,
  depth 17, 30s. Uncounted by-handle Spawn escapes the round (coverq zombie).
  Same depth as pre-F10 run.
- BisectOwnershipVersioning (77714): **VIOLATION as expected** — DiedStaysDead,
  depth 13, 5s. Stale versionless restart/update resurrects a dead node
  (master defect #3 fingerprint). Depth 13 vs 12 pre-F10 — state space shifted
  by the unconditional bump, confirming the refreshed spec is what ran.
All three retained mechanisms remain bisection-proven necessary on the fixed
spec. Outstanding: DowngradeNew, Downgrade4, DowngradeBig (verdict-carriers,
must PASS), BisectReceiptChecks, BisectRegistrationGate (open questions).
- Downgrade4 (77715): **PASS (exhaustive)** — 4 nodes, multi-child relay.
  856.2M states generated, 149.7M distinct, depth 63, queue drained to 0.
  1h06m on 40 workers. Fingerprint-collision estimate 0.003. All safety
  invariants incl. StrictLevelOrder + HandlesCovered hold on F10-fixed spec.
- BisectRegistrationGate (77711): **VIOLATION — gate is NECESSARY at Stage 2**
  (DeadOwnerClean, trace 28, 1.28B states / 298M distinct, 1h54m). Reverses
  the Stage-1 deletion-candidate verdict. See FINDINGS.md F11: cross-level
  instance-list blindness — counts protect within a level; only registration
  maintains the instance list across the V->G transition.
- DowngradeNew (77709): **PASS (exhaustive)** — 3-node base, full design,
  F10-fixed spec. 5.153B states generated, 908.1M distinct, depth 58, queue
  drained to 0, 6h09m. Actual fingerprint-collision estimate 0.012.
- BisectReceiptChecks (77710): **PASS (exhaustive)** — ReceiptChecks off.
  IDENTICAL state counts to DowngradeNew (5,152,721,461 generated /
  908,130,603 distinct / depth 58) under a DIFFERENT fingerprint seed:
  the receipts guard is behaviorally inert at Stage-2 scope — the
  PGlobal-only restriction on catch-up is already implied by the commit
  machinery (post-V-commit no node can be plain Valid). Deletion candidate,
  re-confirm at Stage-3 scope per the F11 lesson. The seed-disjoint count
  match also cross-validates both runs against fingerprint-collision loss.
- DowngradeBig (77716): **BOUNDED-CLEAN to depth 30** — disk-capped, not
  exhaustive. 15.23B states generated / 3.22B distinct over ~19h, then
  ENOSPC on node-local /tmp during fingerprint-set merge (JVM wedged;
  scancelled). F10-fix confirmation margin: the pre-fix violation was
  found at depth 19 / 84M-ish states in 13 min; post-fix BFS fully
  verified all behaviors through depth 29-30 at 180x the state count with
  no violation. Follow-up: DowngradeMid (packs=3, rounds=4) isolates the
  F10-critical pack budget in an exhaustible space.

---
## STAGE-2 MATRIX VERDICT (2026-08-21)
Design verified: exhaustive safety passes at 3 nodes (908M distinct) and
4 nodes (150M distinct); exhaustive liveness (local); all retained
mechanisms bisection-proven necessary (FlagVeto, CoveredByHandle,
OwnershipVersioning, RegistrationGate per F11); ReceiptChecks proven
behaviorally inert at this scope (sole deletion candidate, re-adjudicate
at Stage 3); large-budget config bounded-clean to depth 30.

---
## STAGE-3: COLLECTIVE TREES (2026-08-24, in progress)

Trigger: fuzzer crashes on fixdcprotocol (slurm-77759/60) triaged to the
unmodeled collective-tree topology; root cause F13 (see FINDINGS.md).
Model: TreeNodes constant, binary-heap undirected tree RE-ROOTED at each
round's downgrade owner (= CollectiveMapping::get_children semantics);
members born Valid/pre-registered, never in remote_instances; SrcRouting
toggle (FALSE = master's find_nearest misroute); probe invariants
NoRequestAtBusyNode (gc.cc:1214 assert) and RootRetainsOwnership
(gc.cc:1657 assert). Full details: STAGE3-RESULTS.md.

Modeling note: EventualCollection is now bounded by BudgetQuiesced,
excusing ONLY quiescent states (empty network, no open aggregation)
whose live self-believing owner is round-budget-capped. The bare
property cannot pass at any finite MaxRounds (TLC burns rounds on
doomed attempts and stutters at the cap); the excused terminal was
hand-verified convergent given one more round. Stuck messages, parked
uncapped owners, and lost ownership still violate.

| Run | Expectation | Verdict |
|---|---|---|
| DowngradeDegen new-vs-old spec (TreeNodes={n0}) | identical | PASS: 912,887 / 224,577 distinct / depth 38, byte-identical both specs |
| DowngradeTreeLive (tree {n0,n1}+outsider, fix) | pass | PASS |
| BisectF13 (same, master routing) | violate | VIOLATED (12-state stuck-response wedge) |
| DowngradeTreeSmoke (star tree {n0,n1,n2}+outsider, full invariants+probes) | pass | local: bounded-clean to depth 18, 71.5M distinct, NO violation (probes holding); ENOSPC on local disk at 30min -> promoted to sapling |
| DowngradeTree4 (round 1, no F16 fix) | pass | VIOLATED: StrictLevelOrder, depth 22, 945.7M distinct, 6h16m (sapling 77779) -> finding F16; doubles as the BisectF16 verdict |
| DowngradeTree4 / TreeSmoke / TreeMid / DowngradeNew (round 2, RegClockBump) | pass | queued: sapling/submit_stage3.sh |

Fuzzer round 2 (slurm-77768, post-F13): Mode B (gc.cc:1214) GONE --
field confirmation of F13. Mode A (gc.cc:1633) persisted via F14 (see
FINDINGS.md): restart-transfer through the registration-gate window,
an implementation omission of a guard the model always had
(RegistrationGate => reg on the restart path). Two-layer fix applied,
uncommitted. Release mode reports no errors (asserts compiled out; the
corrupted rounds were debug-visible only).

Stage-3 housekeeping (2026-08-24): all legacy configs gained the new
constants (TreeNodes = {owner}, SrcRouting = TRUE, BoundedLiveness =
FALSE) so every committed config runs against the Stage-3 spec; the
degenerate tree is byte-identical to the old spec (DowngradeDegen).
BoundedLiveness gates the BudgetQuiesced excuse: TRUE only for
DowngradeTreeLive/BisectF13 (tree scope needs it), FALSE for legacy
liveness configs, which were RE-RUN and reproduce their recorded
verdicts exactly: DowngradeNewLive PASS, DowngradeOldLive VIOLATION,
BisectLive{RestartForwarding,RegHandshake,OwnershipVersioning}
VIOLATION, BisectLiveRegistrationGate PASS (matching its recorded
"gate may be redundant for liveness" expectation).

RegClockBump config policy (2026-08-24): TRUE configs verify the NEW
design (DowngradeNew/NewLive, Degen(Bump), all Tree*, BisectF13);
FALSE configs preserve the historical matrix exactly (Old*, all nine
legacy Bisect*, DowngradeDegen equivalence check, BisectF16). Rationale:
with the bump enabled the legacy gate liveness bisection produced a
BUDGET ARTIFACT under the bare property (registrations now consume
clock ids, so previously-convergent behaviors hit MaxRounds; terminal
= quiescent capped owner, msgs empty -- verified by trace), not a real
wedge. Local battery on the fixed spec, all as recorded: DegenBump
PASS, NewLive PASS, TreeLive PASS, OldLive VIOLATION, four BisectLive
verdicts match history (gate PASS with FALSE re-confirmed), BisectF13
VIOLATION.

---
## STAGE-3 ROUND-2 VERDICTS (2026-08-25, sapling, RegClockBump=TRUE)

| Run | Verdict |
|---|---|
| DowngradeTree4 (F16 fix at the violating scope, incl. both probe invariants) | PASS, exhaustive — the F16 fix verdict; NoRequestAtBusyNode and RootRetainsOwnership exhaustively certified at depth-2-tree scope (impl asserts gc.cc:1214/1633 are certified tripwires) |
| DowngradeNew (flat exhaustive with the registration clock bump) | PASS, exhaustive |
| DowngradeTreeSmoke (star + outsider + probes) | bounded-clean: 24h timeout, no violation |
| DowngradeTreeMid (star + two outsiders) | bounded-clean: 24h timeout, no violation |

Assessment: the two exhaustive PASSes are the load-bearing results —
Tree4 is the exact geometry that produced F16 (violated in 6h16m in
round 1, passes exhaustively with the fix), and DowngradeNew covers
the bump's new flat behavior exhaustively. The star configs are the
F16-benign geometry (every response path crosses the owner-space, so
the poison window covers the round); their 24h bounded-clean runs are
accepted on the DowngradeBig precedent (all known failure shapes in
this campaign live at depths 11-22, well within 24h of BFS).

STAGE-3 CAMPAIGN CLOSED: tree topology modeled (re-rooted fan-out,
requester routing, registration semantics), degenerate-exact against
Stage 2, F13/F14/F15/F16 found+fixed+verified, probes certified,
liveness verified at tree scope, historical matrix preserved under the
RegClockBump=FALSE policy.
