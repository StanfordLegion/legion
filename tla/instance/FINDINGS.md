# Findings: PhysicalManager instance-collection protocol model

Numbering continues per-model (this file is for tla/instance; the
downgrade model's F1-F11 live in tla/downgrade/FINDINGS.md).

## F1 — Forwarded user events race the collection decision (UsersGathered)

Smoke trace (depth 8, InstanceSmoke.cfg, log-smoke.out): a replica on n1
is born, the owner's collection round fans to n1, n1 commits (shipping
its gc_events, which are empty), the owner decides SUCCESS and performs
the deletion. THEN record_instance_user runs at n1 -- still
PENDING_COLLECTED, the notification is in flight -- and forwards the
user's ApEvent to the owner (GarbageCollectionRecordEvent, the
physical.cc:502 path). The token is live but was never gathered:
the deferred deletion does not wait on it (use-after-free of the Realm
instance if the user is still running). When the forward lands at the
now-COLLECTED owner, the impl assert `gc_state != COLLECTED_GC_STATE`
in record_instance_user fires (debug) -- in release the event is added
to a gc_events set that was already consumed.

Nothing orders a PENDING remote's record-forward before the owner's
decision: only the RECORDING CALLER waits on the applied event; the
round waits only on events shipped at acquire_collect commit time.
Design question Q2: what should order these? (Options include failing
the round on any record-at-PENDING, having the forward carry an
acquire-like guard, or ruling record-at-PENDING without a covering
valid reference an API violation -- but the message path exists, so
presumably it is meant to be legal.)

## F2 — The failure-path count restore erases mid-round reference traffic (SafeDeletion)

Smoke trace (depth 14, InstanceSmokeNoUsers.cfg, log-smoke-nousers.out):
1. Replica exists on n1. The owner starts round 1 (snapshot of its
   sent/received counts taken; round clock = 1); n1 commits to the round.
2. A mapper acquire SAVES the owner (PENDING -> VALID) mid-round --
   legal, the decision will observe it.
3. The saved owner packs a valid reference to n1: the armed bump
   advances the clock to 2, the vref is stamped 2, sent becomes 1.
   The owner then releases (back to COLLECTABLE).
4. Round 1 decides: owner is COLLECTABLE -> "an acquire won" -> restore
   the count snapshot. **The restore erases sent=1 for the still
   in-flight reference.**
5. Round 2 starts: its snapshot clock is ++clock = 3, ABSORBING the
   pack's clock trail (the stamp 2 is now below the new snapshot). It
   fans to n1, which is still PENDING from round 1 and commits with
   balanced (0,0) counts.
6. Round 2 decides SUCCESS -- counts balanced (erased), clock 3 <= 3 --
   and deletes while the stamped vref is still in flight to n1. When it
   lands, n1 flips VALID holding a reference to a deleted instance (in
   the other interleaving, n1 receives it pre-commit, later rounds see
   received=1 vs erased sent=0 and can NEVER balance again -- a
   permanent leak instead of a use-after-free).

Root cause: sent/received_valid_references serve two roles -- the
node's cumulative pack/unpack log AND the round's accumulation target
for folded remote mismatch reports. The failure restore exists to undo
the folds, but it also undoes legitimate local counts recorded
mid-round, and the next round's ++clock erases the clock evidence.
The downgrade protocol avoids exactly this by accumulating round
reports in SEPARATE counters (total_sent/total_received_references)
and never restoring the primary ones. Candidate fix: same separation
here (fold mismatch reports into round-local accumulators; never touch
the cumulative counters; delete the snapshot/restore machinery).
DESIGN QUESTION for Mike before any fix is modeled as the new design.

## Model-fidelity corrections made during these smokes
- The owner's covering REMOTE_DID_REF during an acquire grant is a pin
  only the ack can release (Release requires refs > pins); first draft
  let the application release it.
- An empty-instance-set collection round starts and decides under a
  single lock hold in the implementation (collection_ready never
  exists); modeled as one atomic action.

## F3 — A replica created mid-round escapes the collection round entirely (SafeDeletion)

Found by the new-design smoke (log-new-smoke.out, depth 16) once F2's
restore bug was fixed, but present in master's design too: the
mechanism involves neither the restore nor the clock.

1. n1 exists and acquires locally (Coll -> Valid, legal reuse). The
   owner starts a collection round; the fan-out set is {n1}.
2. n2 is spawned MID-ROUND (a manager request while the owner is
   PENDING): pack_garbage_collection_state sends payload PENDING, and
   update_remote_instances registers n2 -- but the round's wait set was
   snapshotted at fan-out, so the round will never ask n2.
3. n1 (VALID, holding refs) packs a valid reference to n2; n2's
   unpack+add flips it PENDING -> VALID (legal notify_valid). n2 packs
   a reference back to n1. Both nodes now have sent=1, recv=1 --
   INDIVIDUALLY BALANCED. Neither ever committed to a round, so no
   bump was armed and both packs are stamped with clock 0.
4. n1 releases everything (-> Coll), then processes the round's
   acquire: commits with balanced counts, silent clean response.
5. The owner decides SUCCESS -- no failures, totals balanced, clock
   equal to the snapshot -- and deletes while n2 is VALID holding a
   reference. The notification arriving at VALID n2 fires the
   notify_remote_deletion assert (release mode: the node's instance is
   deleted out from under its valid references).

Root cause: the round's membership is snapshotted at fan-out and the
decision never reconciles against instances registered afterwards. The
counts cannot catch it (a pack/pack-back pair between the new replica
and an existing one is self-balancing), and the clock cannot catch it
(bumps are only armed by round commits, which neither node performed).
This is the instance-protocol analog of the downgrade protocol's
registration findings: there, update_remote_instances POISONS the
in-flight round (notready_owner) -- but that poison sets
downgrade-protocol state only; the instance-collection round has no
equivalent.

Candidate fix (needs Mike's ruling): the same poison, e.g.
pack_garbage_collection_state (owner, under i_lock, when gc_state ==
PENDING_COLLECTED) increments failed_collection_count so the in-flight
round fails and retries against the full instance list. Alternative:
re-check the registered-instance set at decision time and re-fan to
the difference.

## Resolutions (2026-08-21, Mike's rulings encoded)
- F1: contract ruling -- record_instance_user must be called with a held
  valid reference the FIRST time from an external call; later
  reinvocations (the owner-side handler, the round-commit shipping call
  at PENDING) need not be valid. Model: external RecordUser requires a
  held reference; the unordered forward is no longer an action.
  Implementation note: the strengthened assert (VALID at external
  entry) requires distinguishing the external entry from the internal
  shipping/handler entries (split entry or flag).
- F2: separate round-local accumulators (downgrade-style), no
  snapshot/restore. Encoded as SeparateAccums.
- F3: Mike's ruling -- same technique as the downgrade protocol:
  sending a manager while the owner is PENDING poisons the in-flight
  round (impl: pack_garbage_collection_state at a PENDING owner bumps
  failed_collection_count). Encoded as SpawnPoison.

## Q1 audit (2026-08-22, delegated by Mike): add-before-unpack holds
All four PhysicalManager valid-ref unpack sites verified:
1. UnboundPool::unpack (managers/memory.cc:1632-1635):
   add_base_valid_ref (debug: acquire workaround for the "overzealous"
   debug check on in-flight packed refs) THEN unpack. OK.
2. Deferred-buffer response (managers/memory.cc:3852-3855):
   acquire_instance THEN unpack. OK.
3. ExternalDetachRequest (managers/memory.cc:5971-5972): unpack paired
   with a REMOVAL (detach consumes the ref) -- receipt booked after the
   removal is the conservative direction (a round in the window sees
   the sender's sent unmatched and fails); external instances are
   excluded from scope anyway. OK.
4. The eq-set/view map chain (equivalence_set.cc apply_state tail
   10222-10227 -> views/individual.cc:1186): manager valid refs are
   added during state installation (the view-validity cascade) before
   the leftover-clock drain at the function tail; when the view was
   already valid at the destination no add is needed (cover exists)
   and the unpack is receipt-only. OK.
Conclusion: add/acquire/cover always precedes the unpack in program
order. The impl's window (valid-but-receipt-unbooked) makes a
concurrent round fail at that node -- same outcome as the model's
atomic receive, so the atomic RecvVref abstraction is sound.

## F4 — pending_changes accounting (suspected implementation bug, Mike concurs)
The VALID/COLLECTABLE wake branch of collect() (physical.cc:1586-1596)
returns without decrementing pending_changes; only the PENDING branch
decrements (1619). Hand-trace: a leaked count makes the failing
decrement miss zero, so gc_state parks at PENDING_COLLECTED forever;
every later collect() joins a phantom round (1566-1572), waits on the
long-triggered collection_ready, and re-runs the decision on stale
failed_collection_count and current counts WITHOUT re-fanning --
possible spurious deletion, at minimum a permanently parked owner.
NOT yet machine-checked: requires modeling the multi-waiter dedup
(next model increment). Mike: "definitely seems like a bug...
unbalanced and in need of greater scrutiny."

## F4 machine-confirmed (2026-08-22): stranded waiter decides a half-collected round

The multi-waiter (pending_changes) machinery is now modeled: waiters
per round, joiners, wake-order interleavings, orphaned rounds (waiters
stranded when a new fan-out overwrites the round whose
collection_ready they captured), and the SHARED pending_changes /
failed_collection_count members spanning rounds.

BisectWaiterFix (F1 contract + SeparateAccums + SpawnPoison on, waiter
fix off) violates SafeDeletion at depth 11 -- a variant WORSE than the
hand-traced parked-owner leak:
1. Round 1 fans to n1; n1 responds; before the starter wakes, a mapper
   acquire SAVES the owner (PENDING -> VALID) and releases (-> COLL).
   The starter never woke: pending_changes leaks and the waiter stays
   blocked on round 1's already-triggered collection_ready.
2. A new collect() starts round 2 from COLLECTABLE (master gates only
   on gc_state): failed_collection_count RESET, fresh fan-out in
   flight. The round-1 waiter is now stranded against overwritten
   round state.
3. The stranded waiter wakes (its captured event has long triggered)
   and re-runs the decision switch against ROUND 2's HALF-COLLECTED
   state: fails=0 (just reset), counts balanced, clock == snapshot ->
   it decides SUCCESS and performs the deletion BEFORE round 2's
   responses arrive. SafeDeletion violated.

In implementation terms: any collect() waiter that wakes after the
round it joined was decided-by-acquire can run the decision switch
against a LATER round's partially collected state, because the wake
event, pending_changes, and failed_collection_count are not tied to a
round identity.

## F4 fix (verified): WaiterFix semantics
(1) EVERY wake path decrements pending_changes (including the
VALID/COLLECTABLE and COLLECTED cases at physical.cc:1586-1596 and
1698-1703); (2) a round is decided exactly once -- the first waker to
run the PENDING decision (or observe the acquire-win) marks it
decided, later wakers only drain the counter (last one resets
gc_state to COLLECTABLE); (3) a new round may only start once
pending_changes == 0 (all prior waiters drained). With these plus the
F1-F3 fixes the design passes EXHAUSTIVELY at smoke scale: 40.9M
states generated / 8.86M distinct, depth 35, all invariants.

## IMPLEMENTED (2026-08-22, uncommitted, instances/physical.{h,cc})
F1: record_instance_user split into the external entry (asserts
gc_state == VALID_GC_STATE under the recording lock hold; always
records locally) and record_instance_user_internal (the owner-side
GarbageCollectionRecordEvent handler and the round-commit shipping
call in GarbageCollectionAcquire; keeps the relaxed != COLLECTED
assert and the PENDING forward). Local recording factored into
record_gc_event (lock-held helper).
F2: total_sent/received_valid_references round accumulators (reset at
round start; process_remote_reference_mismatch folds into them); the
decision compares accumulators + primaries; the stack
snapshot/restore of the primaries is deleted.
F3: pack_garbage_collection_state's PENDING_COLLECTED case bumps
failed_collection_count (the poison).
F4: every collect() wake path decrements pending_changes; decide-once
via the new collection_decided member; new rounds gated on
pending_changes == 0 (a collect() at COLLECTABLE with undrained
waiters joins-and-drains, returning false, instead of fanning out).
No round-identity was added to messages: with the drain gate a live
round can never be overwritten, and each round's failure/mismatch/
record messages are ordered before its decision by its ready_events,
so the model's gen field has no implementation counterpart to guard.

## Item-3 closing audit (2026-08-23): implementation lock windows
All inst_lock releases in the paths touched by the F1-F4 implementation
verified against the re-validation rule from the downgrade F12 lesson:
- collect()'s wait (physical.cc:1634-1636): the entire decision -- the
  gc_state switch, collection_decided, failed_collection_count, the
  accumulator/primary balance, and clocks -- reads under the single
  reacquired hold; the decide-once flag guards wake reordering.
- perform_deletion's releases: --pending_changes and collection_decided
  are written under the hold BEFORE the call; gc_state = COLLECTED is
  set before the release, so re-entrant collect() calls early-return.
- collection_ready = NO_RT_EVENT: Realm defines NO_EVENT as
  always-triggered (event_impl.cc:41-42), so no wait/release occurs and
  no-remote rounds remain atomic under one hold.
- Pre-existing patterns not touched by the diff (the non-owner request
  round-trip, notify_valid's debug wait-while-holding) left as-is.
No violations of the rule found.

## F1 CORRECTED (2026-08-23, from a CI counterexample)
CI falsified the F1 assert (`gc_state == VALID_GC_STATE` at the
external record_instance_user): a view on a REMOTE node holds the
manager valid there, and its copy-user registration
(ViewAddCopyUserMessage -> IndividualView::add_copy_user) legally
lands at the manager's owner, whose local replica may be COLLECTABLE.
Both the implementation assert and the model's RecordUser guard had
over-literalized the ruling "hold a valid reference" into a LOCAL
state predicate. The corrected contract is a COVERING obligation: a
valid reference is held on SOME node and retained until the record's
applied events trigger, which is what prevents a collection round
from committing while the registration is in flight (the round fails
at the covering node).
- Implementation: the external entry now routes through
  record_instance_user_internal with no local-validity assert (the
  != COLLECTED assert remains); a comment documents the covering
  contract.
- Model: RecordUser(n, r, u) -- record at n covered by refs at r; a
  PENDING-remote record forwards to the owner with the cover PINNED
  until RecvRecUser applies it. RecordCover toggle FALSE reproduces
  the original F1 trace (BisectRecordCover: UsersGathered violation,
  depth 11) -- F1 is thereby RECLASSIFIED from a protocol bug to a
  load-bearing caller discipline, the same category as the acquire
  contract.
- All UsersGathered verdicts from the previous model required
  re-verification: flat exhaustive re-PASSED (55.7M/11.4M, depth 35);
  remaining re-runs tracked in RESULTS.md.
