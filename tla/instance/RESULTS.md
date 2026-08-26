# InstanceGC verification results

## Smoke scale (3 nodes; acqs 2, packs 2, rounds 2, spawns 2, users 1)
- InstanceNewSmoke (F1 contract + SeparateAccums + SpawnPoison, all
  mechanisms on): **PASS (exhaustive)** -- 30.5M states generated,
  6.59M distinct, depth 35, queue drained. All invariants incl.
  SafeDeletion + UsersGathered hold ACROSS non-monotonic
  valid/invalid/valid cycles and failed-round retries, answering the
  F2 non-monotonicity question at this scope.
- InstanceSmoke / InstanceSmokeNoUsers (master mode): **VIOLATION as
  expected** -- F2 regression traces (SafeDeletion).
- Pre-poison new design (SeparateAccums only): **VIOLATION** = F3
  discovery trace (log-new-smoke.out superseded; trace in FINDINGS.md).
- BisectClockCheck (new design, clock off): **VIOLATION as expected**
  (SafeDeletion, depth 20) -- the collect lamport clock remains
  necessary in the new design.
- BisectContract (new design, uncovered add_valid_ref allowed):
  **VIOLATION as expected** (SafeDeletion, depth 11) -- the acquire
  discipline is load-bearing, so the LEGION_DEBUG contract-check
  machinery guards real protocol soundness (Q6 adjudicated: keep it;
  do not model the debug messages themselves).

Full matrix remains GATED on MODEL_PLAN.md Q1, Q3, Q5, Q7, Q8.

## Multi-waiter extension (2026-08-22)
- InstanceNewSmoke (F1 contract + SeparateAccums + SpawnPoison +
  WaiterFix): **PASS (exhaustive)** -- 40.9M generated / 8.86M
  distinct, depth 35, queue drained. Includes joins, wake orders, and
  orphaned-round interleavings.
- BisectWaiterFix (waiter fix off, all else on): **VIOLATION**
  (SafeDeletion, depth 11) -- F4 machine-confirmed via the
  stranded-waiter early-decision variant (see FINDINGS.md).

## Collective-tree stage + matrix split (2026-08-22, 15-min local rule)
- Flat regression after the tree refactor (TreeNodes = {n0} degenerate):
  **PASS (exhaustive)** with byte-identical counts (40,883,863 /
  8,858,081, depth 35) -- the tree extension is backward-compatible.
- BisectSpawnPoison (flat): **VIOLATION as expected** (SafeDeletion,
  depth 18) -- F3 necessity isolated.
- BisectSeparateAccums (flat): **VIOLATION as expected** (SafeDeletion,
  depth 15) -- F2 necessity isolated.
- All five mechanism bisections now confirmed at flat scale
  (ClockCheck, Contract, SeparateAccums, SpawnPoison, WaiterFix).
- InstanceNewTree (4 nodes, tree {n0,n1,n2} + 1 spawnable): exceeds the
  local budget (10-min probe: 171M generated / 38.8M distinct, depth
  21, queue growing) -> SAPLING (sapling/ bundle, submit_long.sh, with
  InstanceNewBig + the two tree-scope bisections).

## Sapling results (2026-08-22)
- BisectTreeWaiterFix (77727): **VIOLATION as expected** (SafeDeletion,
  trace 11, 16s). The F4 stranded-waiter variant at tree scope: round 1
  fans to the tree; the tree leaf n2 acquires mid-round and gcfails it;
  an owner acquire-save + release flips the owner back to COLLECTABLE
  before the round-1 starter wakes (counter leak); round 2 fans; the
  ORPHANED round-1 starter then wakes and decides round 2 SUCCESS while
  round 2's requests are still in flight and n2 is VALID holding a
  reference. Confirms WaiterFix necessity at tree scope.
- BisectTreeSpawnPoison (77728): **VIOLATION as expected**
  (SafeDeletion, trace 17, 4m37s, 83.8M states). F3 at tree scope: n3
  spawned mid-round (born PENDING, unasked); the tree leaf n2 --
  legally VALID mid-round -- packs to n3, n3 flips VALID and packs
  back; both end individually balanced (1,1) with clock stamps of 0
  (neither ever committed, so no bump was armed); n2 releases, then
  commits cleanly to the still-in-flight round; the round decides
  SUCCESS while n3 is VALID holding a reference. Confirms SpawnPoison
  necessity at tree scope.
Outstanding: InstanceNewTree, InstanceNewBig (the two expected-PASS
verdict-carriers).
- InstanceNewTree (77725): **PASS (exhaustive)** -- 4 nodes, collective
  tree {n0,n1,n2} + 1 spawnable. 1.568B states generated, 303.3M
  distinct, depth 37, queue drained to 0, 1h06m on 40 workers.
  Fingerprint-collision estimate 0.0013. All invariants (SafeDeletion,
  UsersGathered, NoFatal, RefsImplyValid, DeadStaysDead) hold for the
  full fixed design at the collective-mapping topology.
- InstanceNewBig (77729): **BOUNDED-CLEAN to depth 30** (disk-capped,
  not exhaustive). 24.85B states generated / 4.29B distinct over 13h;
  the state queue (1.12B states, still growing) exhausted node-local
  /tmp and the disk guard ended the run cleanly (guard + cleanup trap
  worked as designed -- no wedged JVM, no stranded scratch). No
  violation through depth 30 at budgets acqs/packs/rounds = 3,
  spawns/users = 2: every known failure shape in this protocol was
  found at depths 11-18, so the margin is large. Optional follow-up:
  an InstanceNewMid config isolating the pack budget (packs = 3,
  rounds = 2) would be exhaustible if a fully exhaustive large-budget
  verdict is wanted.

---
## INSTANCE-COLLECTION PROTOCOL MATRIX VERDICT (2026-08-22)
Fixed design (F1 record contract + F2 SeparateAccums + F3 SpawnPoison
+ F4 WaiterFix) verified: exhaustive safety PASS at flat 3-node scope
(40.9M/8.9M, depth 35) and at the collective-tree scope (1.568B/303M,
depth 37); large-budget flat config bounded-clean to depth 30 (4.29B
distinct). All seven mechanism bisections violate as required
(ClockCheck, Contract, SeparateAccums, SpawnPoison, WaiterFix at flat
scope; WaiterFix, SpawnPoison at tree scope). Master regressions
reproduce F2/F4. Four genuine master bugs found and fixes verified
(F1-F4, FINDINGS.md); implementation notes accumulated therein.

## Closing pass (2026-08-23; items 1-4, Mike ruled 5/6 out of scope)
- Item 3 (implementation lock-window audit): COMPLETE, no violations of
  the re-validation rule (FINDINGS.md).
- Item 2 (liveness): InstanceLiveNew (fixed design): **PASS
  (exhaustive)** -- WaitersDrain + OwnerUnparks hold (9,947 distinct
  states). InstanceLiveMaster (master waiter machinery,
  properties-only): **TEMPORAL VIOLATION as expected** -- the
  parked-owner counterexample, the liveness half of F4.
- Item 4: InstanceNewMid (packs=3, rounds=2, spawns=1): **PASS
  (exhaustive)** locally -- 3.07M generated / 766K distinct, depth 32.
  Mid2 (spawns=2) running; supersedes InstanceNewBig's bounded-clean if
  it exhausts.
- Item 1: InstanceNewTree4 (depth-2 tree, interior-node subtree-skip
  actually exercised) probing locally; sapling if it exceeds 15 min.
- Item 4 addendum: Mid2 (packs=3, spawns=2) exceeds the local budget
  (277M/59.3M at depth 26, queue growing) -> SAPLING.
- Item 1: Tree4 exceeds the local budget (149M/31.8M at depth 21,
  queue growing) -> SAPLING. Both in sapling/submit_closing.sh.
- InstanceNewTree4 (77731): **PASS (exhaustive)** -- 458.9M generated /
  94.3M distinct, depth 35, queue drained, 21m19s. The interior-node
  subtree-skip (VALID interior node fails without forwarding; unasked
  child; subtree-wide wait resolution) is now exhaustively covered.
- InstanceNewMid2 (77732): **PASS (exhaustive)** -- 477.3M generated /
  95.0M distinct, depth 38, queue drained, 21m57s. Fully exhaustive
  large-pack-budget verdict (packs=3, spawns=2), superseding
  InstanceNewBig's bounded-clean.

---
## CAMPAIGN CLOSED (2026-08-23)
The PhysicalManager instance-collection protocol (fixed design: F1
record contract + F2 separate accumulators + F3 spawn poison + F4
waiter discipline) is verified: exhaustive safety at flat (8.9M
distinct), collective-tree depth-1 (303M) and depth-2 (94.3M, the
subtree-skip), and large-pack (95M) scopes; exhaustive liveness
(WaitersDrain + OwnerUnparks) with master's temporal counterexample
documented; seven necessity bisections; master regressions for F2/F4;
implementation lock-window audit clean. Out of scope by ruling:
cross-protocol composition (local GLOBAL-reference pin reasoning),
external/attached and unbound instances. Implementation uncommitted in
instances/physical.{h,cc} pending review.

## F1 CORRECTION re-verification (2026-08-23; see FINDINGS.md)
The CI counterexample showed the F1 assert (and the model's RecordUser
guard) over-literalized the contract: covering references may be held
on a DIFFERENT node than the recording one. Model corrected
(RecordUser(n, r, u) with pinned forwards; RecordCover toggle); code
assert relaxed to the covering contract. All users>0 verdicts on the
previous spec invalidated and re-run:
- InstanceNewSmoke: **re-PASS (exhaustive)** -- 55.7M / 11.4M, depth 35
  (space grew with the forward+pin machinery).
- InstanceNewMid: **re-PASS (exhaustive)**.
- All five flat bisections (ClockCheck, Contract, SeparateAccums,
  SpawnPoison, WaiterFix): **VIOLATION as required** on the corrected
  spec. (BisectClockCheck/BisectContract configs also repaired: they
  had never gained the SpawnPoison/WaiterFix/RecordCover constants.)
- NEW BisectRecordCover: **VIOLATION as required** (UsersGathered,
  depth 11) -- the covering discipline is load-bearing; F1 is
  reclassified as a caller contract, same category as the acquire
  discipline.
- SAPLING reruns required (previous verdicts stale): InstanceNewTree,
  InstanceNewTree4, InstanceNewMid2, BisectTreeWaiterFix,
  BisectTreeSpawnPoison (see sapling/README.md).
- Liveness configs (users=0) unaffected and remain valid.

## Sapling reruns on the corrected model (2026-08-23, jobs 77751-77755)
- InstanceNewTree (77751): **re-PASS (exhaustive)** -- 2.304B generated
  / 413.7M distinct, depth drained, 1h33m (was 303M distinct on the
  pre-correction spec; growth = the forward+pin machinery).
- InstanceNewTree4 (77752): **re-PASS (exhaustive)** -- 692.6M / 130.6M,
  30m21s. The interior-node subtree-skip re-verified under the
  corrected user-recording semantics.
- InstanceNewMid2 (77753): **re-PASS (exhaustive)** -- 659.8M / 123.4M,
  28m53s.
- BisectTreeWaiterFix (77754): **VIOLATION as required** (SafeDeletion,
  14s).
- BisectTreeSpawnPoison (77755): **VIOLATION as required**
  (SafeDeletion, 7m47s).

---
## CAMPAIGN RE-CLOSED (2026-08-23)
Every verdict re-established on the corrected (covering-semantics)
spec: exhaustive safety at flat (11.4M distinct), tree depth-1
(413.7M), tree depth-2 (130.6M), and large-pack (123.4M) scopes;
exhaustive liveness (unchanged, users=0); EIGHT necessity bisections
(the original seven plus BisectRecordCover for the covering
discipline); master regressions intact. Implementation (with the
relaxed F1 assert) uncommitted in instances/physical.{h,cc} pending
review.
