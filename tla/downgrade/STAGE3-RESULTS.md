# Stage-3: collective-tree verification of the downgrade protocol

Date: 2026-08-24. Trigger: fuzzer crashes on `fixdcprotocol`
(slurm-77759/77760, 33 instances, two assert signatures), triaged to the
collective-tree topology — the documented Stage-3 gap. This file is the
detailed Stage-3 record; summary verdicts also live in RESULTS.md and
the F13 finding in FINDINGS.md (both tracked on fixdcprotocol).

## F13 — misrouted downgrade responses for non-member instances
## of collective mappings (LATENT IN MASTER)

Root cause of BOTH fuzzer failure modes, found statically and confirmed
by the model (below):

- Requests to NON-TREE remote instances of a collective-mapped DC are
  fanned directly by the tracking node (owner-space, or any member with
  its own `remote_instances`), which counts them in ITS
  `remaining_responses`. The instance's RESPONSE, however, routed via
  `get_downgrade_target` -> `CollectiveMapping::find_nearest(local)` —
  the member nearest BY ADDRESS-SPACE ID (collectives.cc:2243) — which
  is generally NOT the sender.
- The misrouted response is not expected by its receiver: the sender's
  aggregation never drains (wedged round) and, if the receiver is
  mid-aggregation, its counts are corrupted (early/incorrect decisions,
  late responses landing on retried/transferred rounds).
- Fuzzer mode A (gc.cc:1657 `downgrade_owner == local_space` at the
  decision) and mode B (gc.cc:1214 `remaining_responses == 0` at
  `check_for_downgrade`) are both downstream of this.
- MASTER has identical routing and asserts but fuzzes clean: a wedged
  round there just sits forever (restarts dropped, no nudges, no
  retries) — a SILENT LEAK. The new liveness machinery re-agitates
  wedged rounds over corrupted state, turning the leak into asserts.
- FIX (uncommitted, kernel/garbage_collection.{h,cc}): new
  `round_parent` member records the requester in
  `process_downgrade_request`; all five response paths use it;
  `get_downgrade_target` deleted. This matches the model, which always
  routed responses to the request's source (`rd.par`).

## The Stage-3 model

`Downgrade.tla` extended with:
- `TreeNodes` constant: binary-heap-shaped UNDIRECTED tree over the
  collective members (OwnerSpace at index 1), RE-ROOTED at each round's
  downgrade owner (`ChildrenRR`/`NextHop` = the impl's
  `CollectiveMapping::get_children(origin, local)`).
- Members born Valid + pre-registered, never in `remote_instances`
  (packs to members count but do not register or poison); re-born
  members stay members. Spawns restricted to non-members. The hairy
  first-registration transfer gated on `TreeNodes = {OwnerSpace}`
  (impl gates it on `collective_mapping == nullptr`).
- Request relay at every member: re-rooted tree children plus, at the
  owner-space, the flat `instSet`. Success waves forwarded down the
  receiver-oriented tree (stale orientation under-delivery is faithful;
  catch-up covers it).
- `SrcRouting` toggle: TRUE = the F13 fix (responses to the requester);
  FALSE = master's `find_nearest` (abstracted: a fixed member other
  than the owner-space).
- Probe invariants for the impl's two tripwire asserts:
  `NoRequestAtBusyNode` (gc.cc:1214) and `RootRetainsOwnership`
  (gc.cc:1657).
- `EventualCollection` bounded with `BudgetQuiesced`: excuses ONLY
  quiescent states (empty network, no open aggregation) where a live
  self-believing owner is round-budget-capped. Wedges with stuck
  messages, parked uncapped owners, or lost ownership still violate.
  (The bare property cannot pass at any finite MaxRounds: TLC burns
  rounds on doomed attempts and stutters at the cap; the excused
  terminal was hand-verified convergent given one more round.)

## Verdicts (local, 2026-08-24)

| Run | Config | Expectation | Result |
|---|---|---|---|
| Degenerate equivalence | DowngradeDegen (TreeNodes={n0}) vs old spec, same budgets | identical | PASS: both 912,887 generated / 224,577 distinct / depth 38 |
| Tree liveness, fix | DowngradeTreeLive (tree {n0,n1}+outsider, SrcRouting TRUE, rounds 5) | pass | PASS |
| F13 bisection | BisectF13 (same, SrcRouting FALSE) | temporal violation | VIOLATED: minimal 12-state wedge — non-member n2's not-ready response delivered to n1 (nearest-by-ID) instead of requester n0; n0's round never drains. Master's silent leak, machine-checked. |
| Tree safety + probes | DowngradeTreeSmoke (4 nodes, star tree {n0,n1,n2}+outsider, DowngradeNew budgets) | pass | (running) |
| Depth-2 tree | DowngradeTree4 (5 nodes, tree {n0,(n1->n3),n2}+outsider) | pass | sapling: submit_stage3.sh |
| Two outsiders | DowngradeTreeMid (star tree + {n3,n4}) | pass | sapling: submit_stage3.sh |

Probe semantics: if `NoRequestAtBusyNode` / `RootRetainsOwnership` HOLD
at tree scope with SrcRouting=TRUE, the impl's asserts at gc.cc:1214 and
gc.cc:1657 are certified tripwires (at modeled scope) under the F13 fix.
A violation trace = a legitimate scenario the impl must handle
(deferral), not assert.

## ROUND-2 CLOSURE (2026-08-25)
Tree4 (fix ON) PASS exhaustive incl. probes; DowngradeNew (flat+bump)
PASS exhaustive; TreeSmoke/TreeMid bounded-clean at 24h. Campaign
closed; see RESULTS.md for the verdict table and FINDINGS.md F13-F16.
