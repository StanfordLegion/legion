# Downgrade protocol implementation notes (model -> code)

Implementation of the Stage-2-verified design on branch `fixdcprotocol`,
confined to `runtime/legion/kernel/garbage_collection.{h,cc}` (+469/-118).
Uncommitted, for review. Each mechanism below maps a spec construct to its
code location.

## Wire format changes (all internal to the downgrade messages)
- **DowngradeRequest**: + `owner_version` (after owner).
- **DowngradeUpdate**: + `owner_version` (after state).
- **DowngradeRestart**: + `candidate`, `candidate_version` (after did).
  The candidate was previously implicit (the network source); explicit so
  restarts can be forwarded.
- **Packed global refs**: the serialized LamportClock is now a *token* —
  `(clock << 1) | valid_stamp_bit`. Encode/decode live entirely inside
  pack_global_ref / unpack_global_ref, so all ~35 call sites (including
  the two-step pack -> store -> serialize -> peek -> unpack pipelines) are
  untouched and treat it opaquely. Valid-ref clocks stay raw (stamp ==
  level there, no bit needed).

## Mechanism map
| Spec construct | Code |
|---|---|
| F10 unconditional bump | pack_global_ref + acquire_global_remote: level conditions on the bump deleted |
| Stamp-counting (Pack dual count, RecvRef dual receipt, born-in-creator-state) | `record_valid_stamp` (sender: counts sent_valid, returns stamp bit) / `apply_valid_stamp` (receiver: counts received_valid, promotes a GLOBAL-born fresh replica to VALID) — virtual, base = no-stamp/abort |
| FlagVeto at EVERY vote (`FlagVeto => ~rflag`) | `pending_downgrade_restart` checked at: leaf vote (check_for_downgrade ready), relay re-vote + root decision (process_downgrade_response); set on unpack mid-round (new `else` in both unpacks), on restart mid-round, and on parked restarts |
| rflag clear rules (`~can`, rdyAt-else, root retry/transfer, RecvUpd) | cleared exactly at: not-ready leaf/relay answers, root decision else-paths, ownership adoption in process_downgrade_update; **kept** when remaining_responses > 0 (can_delete drain now conditioned on it) |
| OwnershipVersioning (`ver`, adopt iff strictly newer) | `downgrade_owner_version`; increments at all three transfer sites (response transfer, restart transfer, update_remote_instances hairy); DowngradeUpdate handler drops stale versions; request adoption gated in process_downgrade_request |
| rnd.ow vs own (round vs belief) | new `round_owner` field: response aggregation/routing keys off it; `downgrade_owner` stays the belief and is only version-adopted |
| RestartForwarding (forward/park, never drop) | check_for_downgrade_restart rewritten: forward when our version is strictly fresher (re-tagged), else park in pending_downgrade_restart; the old silent drops (stale notready, unknown-instance) deleted |
| RegistrationGate (F11: `RegistrationGate => reg`) | blocking wait on `remote_registered` at check_for_downgrade entry (mirrors pack_global_ref's wait) — gates leaf votes AND round starts by an unregistered owner; blocking = no lost wakeup, replaces the model's RegRespMsg->TryRound |
| Registration nudge (RecvReg nudgeOut) | update_remote_instances gains `registration` flag (set only by DistributedRemoteRegistration::handle): outside a round it nudges the downgrade owner (self: check_for_downgrade_restart; remote: restart with cand=owner); mid-round the existing notready poison covers it |
| CatchUp = commit-proof only (replaces master's while-loop, defect #12) | process_downgrade_request: `if (to_check < current_state)` + assert PENDING_GLOBAL (the receipts-proven tripwire) + single perform_downgrade; below-level tripwire assert after |
| Level-aware rollback/commit on RecvUpd (rollV/commitV/rollG) | Valid process_downgrade_update: PENDING_GLOBAL + V-stamp -> VALID (rollback); PENDING_GLOBAL + G-stamp -> perform_downgrade (commit proof, ordered before ownership adoption so the assert holds); PENDING_LOCAL -> GLOBAL; never moves state down otherwise. Base: PENDING_LOCAL -> GLOBAL only. Both guarded by remaining_responses == 0 before re-checking |
| Transfer restartOut (root hands its veto to the new owner) | process_downgrade_response transfer branch: restart(cand=local) follows the update when the flag was set |
| Root retry (`retry == ~causal \/ rflag`) | process_downgrade_response: retry on causality violation OR veto |
| TryRound parts={} per-level counts | virtual `has_packed_references()` (valid level checks both levels' counts) replaces the global-only checks in can_delete + update_remote_instances hairy + the owner-alone assert |
| Blocking find in RecvUpd/RecvReq | unchanged (find_distributed_collectable) — restart-transfer comment documents reliance on it |

## Deliberately NOT implemented (per verification verdicts)
RoundTagging, creation references, defunct replies, abort broadcast,
objLvl/MixedLevelReady, ReceiptChecks-as-a-branch (present only as the
PENDING_GLOBAL assert tripwire; re-adjudicate at Stage 3 before deleting
the assert).

## Option A / by-handle audit (no code changes needed)
- Owner-originated sends (send_node pattern) already count
  (pack_valid_ref/pack_global_ref in the response) and pre-register
  (update_remote_instances(target) at send time, `registration=false` so
  no spurious nudges). Index tree replicas skip trailing registration.
- Requester-side creations (FutureImpl api/future.cc:2560, FutureMapImpl
  api/future_map.cc:605, remote expressions kernel/runtime.cc:10100 —
  which already skips when the source is the owner) keep trailing
  send_remote_registration; they are the model's ref-created path and are
  covered by the gate + nudge + veto.
- Covering invariant stays a caller obligation (model pins are
  bookkeeping only); the by-handle response audit list = the
  send_remote_registration/update_remote_instances call sites.

## Known-separate scope (flagged, untouched)
- PhysicalManager runs its own shadow collection protocol
  (gc_state/collect_lamport_clock under inst_lock, instances/physical.cc
  ~620-673) — not a ValidDistributedCollectable. Its gc refs use the base
  protocol (token-compatible); its valid-ref machinery should eventually
  get the same F10/F3-class review.
- Collective mappings (Stage 3): the tree fan-out paths compile and carry
  versions, but tree-specific interleavings are unverified until Stage 3.
- LEGION_DEBUG_GC pre-existing: header declares acquire_global_remote
  without the clock param (h:288) while the .cc defines only the
  clocked one — mismatch predates this change.

## Verification of the change itself
All 29 affected translation units pass -fsyntax-only with the regent
defines (LEGION_DEBUG enabled). No call sites outside
garbage_collection.cc required changes.

---
## F13 fix (2026-08-24, committed 3fdb88c529)

Model rd.par -> impl round_parent (AddressSpaceID, guarded by gc_lock):
set from the message source at the top of process_downgrade_request;
consumed by ALL FIVE response paths (gate raced-voter, leaf ready vote,
leaf not-ready, catch-up raced response, relay aggregation).
get_downgrade_target DELETED — it recomputed a topological target and
disagreed with the sender for non-members of a collective mapping
(find_nearest = nearest by address-space ID), which is finding F13.
Deviation from model: none; the model always responded to the request's
source.

## F14 fix (2026-08-24, uncommitted at time of writing)

Restores the model's `RegistrationGate => reg` guard on the restart
path (check_for_downgrade_restart) and re-validates ownership belief in
check_for_downgrade's post-gate raced check (the F12 check tests only
remaining_responses; a restart-transfer through the gate window leaves
it zero). DEVIATION: at an unregistered owner the model DROPS the
restart (liveness is carried by the registration handshake's nudge);
the implementation PARKS it (pending_downgrade_restart = true). Parking
is strictly conservative -- an extra veto costs at most one retried
round -- and preserves the never-drop-a-restart doctrine used
everywhere else in the implementation.
