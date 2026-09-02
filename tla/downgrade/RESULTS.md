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

## F19/F20/F21 acquire-protocol campaign (overnight 2026-08-31 -> 09-01)
Final-spec local verdicts (AcquireMode x StaleUpdateProbe; all small
configs, metadirs cleaned -- deep runs belong on sapling):
| Config | Mode | Expectation | Verdict |
|---|---|---|---|
| BisectF19 | deny (master) | spurious deny | VIOLATED AcquireContract (14 states) |
| ProbeAcqResolved | park | F20 requester hang | VIOLATED AcqResolved |
| BisectF21 | hybrid, probe OFF (committed F18) | entry leak | VIOLATED UpdsDrain (22 states) |
| ProbeRequesterAcqSmoke | requester | pass | PASS exhaustive 1,791,246 distinct |
| ProbeHybridAcqSmoke | hybrid, probe ON | pass | PASS exhaustive 1,791,246 distinct (8,010,890 gen; AcquireContract + AcqResolved + UpdsDrain) |
| DowngradeDegen | hybrid, probe ON | pass | PASS 78,094 distinct |
| DowngradeDegenBump | hybrid, probe ON | pass | PASS |
Sapling overnight batch (submitted by Mike ~04:00) runs the PRE-probe
hybrid spec: DowngradeNew/4/Big, Tree{Smoke,4,Mid}, DowngradeNewLive,
ProbeHybridAcq, ProbeRequesterAcq. A follow-up batch with the probe
spec (this repo's tla/downgrade, already staged in sapling/) is
recommended for the deep F21 configs.
Implementation (uncommitted, master): see FINDINGS F19/F21
IMPLEMENTATION record. Full runtime rebuild clean (0 warnings);
test/lightweight passes (10K tasks, clean shutdown).

## Sapling batch 1 (hybrid mode + model-only regresp adoption),
## jobs 77920-77928, submitted 2026-09-01 ~04:00, adjudicated ~13:00
| Job | Config | Verdict |
|---|---|---|
| 77920 | DowngradeNew (flat exhaustive + AcquireContract) | PASS exhaustive, 2,555,203,454 generated |
| 77921 | Downgrade4 | PASS exhaustive, 389,449,597 generated |
| 77922 | DowngradeBig | running clean @ 6.96B gen / 1.51B distinct |
| 77923 | DowngradeTreeSmoke | running clean @ 5.71B gen / 1.27B distinct |
| 77924 | DowngradeTree4 | running clean @ 5.69B gen / 1.28B distinct |
| 77925 | DowngradeTreeMid | running clean @ 6.23B gen / 1.31B distinct |
| 77926 | DowngradeNewLive (EventualCollection) | PASS exhaustive, 7,804,362 generated |
| 77927 | ProbeHybridAcq (contract+drain, depth 2) | PASS, 177,969,351 generated |
| 77928 | ProbeRequesterAcq (requester mode, depth 2) | PASS, 167,652,139 generated |
Caveat: this batch predates the RegRespOwnership fidelity finding and
the final-deny design; it verifies the hybrid design under the model-
only handshake adoption. Batch 2 (impl-faithful "final" mode, submitted
2026-09-01 ~13:00) is the design-of-record verification.

## Sapling batch 2 (DESIGN OF RECORD: AcquireMode "final" +
## RegRespOwnership FALSE), jobs 77929-77937, adjudicated 2026-09-02
| Job | Config | Verdict |
|---|---|---|
| 77929 | DowngradeNew (flat exhaustive + AcquireContract) | PASS EXHAUSTIVE, 1,564,052,600 gen / 315,750,424 distinct, depth 50 |
| 77930 | Downgrade4 | PASS EXHAUSTIVE, 254,024,782 gen / 55,235,878 distinct |
| 77931 | DowngradeBig | bounded-clean at 24h: 18.67B gen / 4.02B distinct |
| 77932 | DowngradeTreeSmoke | bounded-clean at 24h: 16.33B gen / 3.49B distinct |
| 77933 | DowngradeTree4 | bounded-clean at 24h: 15.38B gen / 3.38B distinct |
| 77934 | DowngradeTreeMid | bounded-clean at 24h: 16.47B gen / 3.47B distinct |
| 77935 | DowngradeNewLive (EventualCollection) | PASS EXHAUSTIVE, 3,346,140 gen / 886,446 distinct |
| 77936 | DowngradeTreeLive | PASS EXHAUSTIVE, 102,928 gen / 34,584 distinct |
| 77937 | ProbeFinalAcq (AcquireContract + AcqResolved + UpdsDrain, chase depth 2) | PASS EXHAUSTIVE, 70,889,710 gen / 15,877,097 distinct |
Zero violations across both batches. Batch-1 stragglers (77922-77925,
hybrid mode) ended bounded-clean at 5.7-7.0B gen / 1.27-1.51B distinct.
The final-deny design is verified at the campaign's full scope:
exhaustive flat safety with the acquire contract, exhaustive flat and
tree liveness, exhaustive deep acquire probes, and multi-billion-state
bounded-clean tree/big-budget safety.
