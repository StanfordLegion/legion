# TLA+ model plan: PhysicalManager instance-collection protocol

Target: the valid-reference / garbage-collection protocol on
`PhysicalManager` (`instances/physical.{h,cc,inl}`), which is SEPARATE
from the DistributedCollectable downgrade protocol (`tla/downgrade`).
Key difference: **non-monotonic**. An instance oscillates between valid
(has valid data, cannot be collected) and collectable arbitrarily many
times; only COLLECTED is a one-way door. The protocol arbitrates the
race between mappers acquiring instances (the fast, common case) and
the garbage collector reclaiming them.

## Protocol summary (from code)

Per-node state `gc_state`:
`VALID` <-> `COLLECTABLE` -> `PENDING_COLLECTED` -> (`COLLECTED` | back)
with `PENDING -> VALID` possible via owner-arbitrated saves.
Validity is NODE-LOCAL (unlike downgrade levels): each node's state
tracks whether IT holds valid references; the instance is globally
protected if any node is valid, enforced by collection rounds that ask
every node.

Mapper side (`acquire_instance` / `acquire_internal`):
- Lock-free CAS fast path: refs > 0 -> refs++ (no lock, no state read).
- VALID: refs++. COLLECTABLE: flip to VALID locally (no messages!).
- PENDING: owner may flip itself back to VALID ("save it"); a remote
  must ask the owner (AcquireRequest -> owner acquires its own valid
  ref as cover -> AcquireResponse adds an UNCOUNTED reference at the
  requester -> ack releases the owner's covering ref).
- COLLECTED: fail; a remote whose owner-ask fails goes COLLECTED and
  notifies subscribers.

GC side (`collect`, owner-arbitrated; remote requests forward to owner):
- Owner COLLECTABLE -> PENDING; new round: snapshot counts (for restore),
  `pending_collect_lamport_clock = ++collect_lamport_clock`, arm bump;
  fan `GarbageCollectionAcquire(round clock)` to all remote instances;
  wait for every response.
- Remote VALID -> `GarbageCollectionFailed` (owner counts failures).
  Otherwise -> PENDING; swap its gc_events to the owner; report
  (sent,received,clock) via `GarbageCollectionMismatch` when unbalanced
  or when its clock exceeds the round snapshot; arm bump.
- Owner decision (all responses in): fail if any acquire failed, or
  accumulated sent != received, or clock > snapshot; on failure restore
  the owner's local count snapshot (folded remote reports are
  discarded; clock stays monotonic). On success: perform deletion
  (deferred on gathered gc_events) and broadcast
  `GarbageCollectionNotification` (remotes -> COLLECTED).
- Mid-round acquires flip the owner to VALID/COLLECTABLE, which the
  decision observes as failure.

Valid references travel between nodes (piggybacked on analysis
messages) via `pack_valid_ref`/`unpack_valid_ref` with sent/received
counts and a lamport clock stamped on each pack (bump-after-commit),
exactly the downgrade-style counting but for a single, resettable
level. Note the count/clock machinery here was recently added and has
NOT been through a verification campaign.

## Abstractions (same rules as tla/downgrade)
- One action == one `inst_lock`-held handler execution; the CAS fast
  path is its own atomic action (sound: the counter linearizes it).
- Message set: no loss, no duplication, arbitrary order.
- Flat topology first (collective-mapping trees are a later stage,
  like downgrade Stage 3).
- The owner-side MemoryManager driver, GC priorities, and eager
  collection reduce to nondeterministic triggers: NEVER priority == an
  acquire that never releases; EAGER == a collect attempt after any
  invalid transition. Both are subsumed by unconstrained Acquire /
  CollectStart actions with budgets. (Q7)
- `pending_changes` deduplication of concurrent collects: modeled as a
  single round at a time initially. (Q8: is more needed?)
- Remote-initiated collection = a message that triggers the owner's
  CollectStart; folded into the nondeterministic trigger.
- gc_events / deferred deletion: modeled as USER TOKENS. RecordUser
  places a live token at a node (forwarded to the owner when a remote
  is PENDING, per record_instance_user); collection responses carry the
  node's tokens to the owner; the deletion-safety invariant is that
  every still-live token is in the owner's gathered set at the moment
  of deletion. This models use-after-free of the Realm instance.
- The round's per-remote completion (done event + optional Mismatch +
  optional RecordEvent, all of which the round waits on) is modeled as
  ONE atomic `gcdone` message carrying counts+clock+tokens. Sound
  because the impl's decision cannot run until all three land.
- Replica creation: owner-only sends (managers are requested from and
  sent by the owner), pre-registered via update_remote_instances at
  send time; the payload state follows pack_garbage_collection_state
  (VALID/COLLECTABLE -> COLLECTABLE, PENDING/COLLECTED sent as-is), so
  replicas born mid-round are born PENDING.
- Excluded from stage 1: external/attached instances (detach makes
  notify_invalid legal from non-VALID and COLLECTED sticky), unbound
  instances, instance redistricting (`hole` plumbing), padded
  reservations. (Q5)

## Invariants
- RefsImplyValid: refs[n] > 0 => st[n] = Valid.
- DeadStaysDead: COLLECTED is terminal (aux history flag).
- SafeDeletion: the owner deletes => no node holds refs, no node is
  VALID/COLLECTABLE, and no valid-ref or acquire-grant message is in
  flight. (The mapper/GC race soundness claim.)
- UsersGathered: the owner deletes => every live user token has been
  gathered at the owner. (No use-after-free.)
- Impl assertions as reachability invariants (tripwires we must prove
  unreachable, since the code raises Fatal/asserts on them):
  - a packed valid ref unpacked at a COLLECTED node (the "internal
    garbage collection race" fatal error in notify_valid)
  - GarbageCollectionAcquire arriving at a COLLECTED node
  - GarbageCollectionNotification arriving at a VALID/COLLECTABLE node
- Liveness: with fair message delivery, budgets exhausted, and all
  references released, a collect attempt eventually succeeds (no
  permanently-uncollectable garbage); and symmetric mapper progress:
  acquire attempts on a never-collected instance eventually succeed.

## Suspected races to probe (from code reading, unconfirmed)
- R1: unpack_valid_ref + its paired add_valid_reference can flip a
  PENDING remote to VALID with no owner arbitration; soundness rests
  entirely on counts+clock (the packer must be VALID and thus fail the
  round, or the counts must not balance). This is the heart of the
  protocol; the round-interleaved variants are exactly what TLC is for.
- R2: record_instance_user at a PENDING remote forwards the user event
  to the owner, but only the CALLER waits on the applied event -- an
  in-flight forward does not block the round's decision. If the round
  can succeed while the forward is in flight, the deferred deletion
  misses that event (use-after-free). Need to establish what, if
  anything, orders the forward before the decision. (Q2)
- R3: failed rounds leave born-during-round replicas parked in PENDING
  forever (nothing un-pends them); all their acquires take the owner
  slow path until one succeeds. Correctness fine, but is this the
  intended design? (Q3)
- R4: on failure the owner restores its local count snapshot and
  discards folded remote reports; remotes never reset their counts.
  Confirm the intended invariant is "counts are only meaningful as
  accumulated at the owner within a single round" -- the restore path
  depends on it. (Q4)
- R5: the acquire-grant covering handshake (owner holds REMOTE_DID_REF
  until the requester's ack) pins the owner VALID during the grant
  flight. Interaction with a concurrent round start (owner must be
  COLLECTABLE to start) looks safe by construction; verify.

## Debug-mode messages (GarbageCollectionDebugRequest/Response)
These implement a CONTRACT check, not protocol machinery: in
LEGION_DEBUG, the first add_valid_ref on a remote node (without an
acquire) synchronously asks the owner to prove the instance is validly
held somewhere (owner does acquire+release); release mode instead
raises the fatal error only when the state is already COLLECTED.

Recommendation: do NOT model the debug messages themselves. The model
enforces the acquire discipline by construction (its action set only
contains legal reference operations), so the debug round-trip would
verify a tautology. What the model CAN do is adjudicate the contract
itself: if we add a toggle action "AddValidWithoutAcquire" (an
uncovered add_valid_ref at a remote in COLLECTABLE state) and safety
breaks, that is machine-checked proof the contract is load-bearing --
i.e., the debug check guards real soundness and must stay in the
implementation. If safety somehow holds without the contract, the
debug machinery (and its blocking round trip) is deletable. Either
verdict is cheap to obtain. Note verification of the protocol can
never render the debug check redundant as a caller-bug detector; the
question is only whether the protocol relies on the contract.

## Open design/fidelity questions (GATE: no full matrix until answered)
Answered: Q2 (F1 contract: external record_instance_user requires a
held valid reference), Q4 (F2: separate round accumulators, ruled),
Q6 (machine-answered: BisectContract violates, keep the debug checks).

- Q1: unpack/paired-add adjacency and ORDER. Example:
  PhiView::add_initial_references (views/phi.cc:124-155) adds then
  unpacks in separate lock sections, and the pairing can be deferred
  to a meta-task (phi.cc:292-295). Add-then-unpack is safe (node VALID
  before the receipt books); unpack-then-add lets a round balance on
  the receipt and commit before the add fires at COLLECTED (the Fatal
  at physical.cc:826-833). Is add-before-unpack the enforced
  convention at every manager valid-ref site?
- Q3: remotes stay parked in PENDING after failed rounds --
  acquire_collect sets PENDING (physical.cc:1133); the failure path
  resets only the owner (1619-1620). Every past round participant
  loses the lock-free acquire fast path (physical.inl:88-100) until an
  acquire succeeds. Intended, or should failure broadcast an un-pend?
- Q5: stage-1 exclusions -- external/attached (physical.cc:903-907,
  physical.inl:83-86), unbound (physical.cc:796, 903), collective
  trees (physical.cc:1282-1305, 1507-1528). Confirm.
- Q7: NEVER == acquire-save that never releases (physical.cc:1817-1850
  is transition-identical to owner acquire-save); EAGER == a
  nondeterministic CollectStart (physical.cc:912-916, 1852-1870).
  Confirm no other priority-specific transition.
- Q8: pending_changes multi-waiter dedup is NOT modeled (single round
  at a time). Hand-trace of concern: the VALID/COLLECTABLE wake branch
  (physical.cc:1586-1596) never decrements pending_changes; a leaked
  count makes the failing decrement (1619) miss zero, parking
  gc_state at PENDING permanently, after which every collect() joins a
  phantom round (1566-1572) and re-decides on a triggered
  collection_ready without re-fanning -- possible spurious deletion.
  Confirm whether 1586-1596 is a bug, and whether stage 1 should model
  the dedup.

## Verification stages
1. Master-fidelity model, small budgets, safety smokes: reproduce or
   refute the suspected races R1-R5, adjudicate the debug-contract
   toggle. SMOKE-LEVEL ONLY until the questions above are answered.
2. Full matrix (local + sapling) with per-mechanism bisections
   (ClockCheck, CountReports, SnapshotRestore, PendingSave, ...):
   expected-violation configs document necessity; expected-pass configs
   certify the design. Add liveness configs (leak freedom + mapper
   progress).
3. Collective-mapping trees; multi-round concurrency (pending_changes);
   external-instance detach if warranted.

## Rulings round 2 (2026-08-22)
- Q1: AUDITED (see FINDINGS.md) -- add-before-unpack holds at all
  four PhysicalManager sites; atomic RecvVref is sound.
- Q3: RULED intended -- failure discovery is deliberately lazy; an
  eager failure broadcast could race with acquire bumps and stale
  failures arriving after the next round starts would be unsound.
  Parked remotes recover via their next successful acquire.
- Q5: RULED -- exclude external/attach-detach and unbound instances
  (not eligible for collection); collective_mapping behavior MUST be
  modeled. Note the tree topology is mixed: GarbageCollectionAcquire
  fans down the tree with completion aggregated up via child_done
  events, but gcfail/mismatch messages go DIRECTLY to the owner.
- Q8: pending_changes = suspected impl bug (F4). Model increment:
  multi-waiter dedup (concurrent collect() callers sharing a round,
  per-waiter wake/decide, the pending_changes counter and its
  reset-to-COLLECTABLE rule) so TLC can adjudicate the sound design.
- Q7 still open, restated: confirm (a) NEVER priority's only protocol
  effect is the acquire-shaped transition at physical.cc:1817-1850
  (state flip to VALID + a valid reference that is never released
  until the priority is raised, whereupon 1871-1872 releases it), and
  (b) EAGER priority's only effect is WHEN collect() gets invoked
  (notify_invalid 912-916 locally, remote invalids via
  GarbageCollectionRequest to the owner), never any state transition
  of its own. If both hold, priorities are subsumed by the model's
  unconstrained AcquireTry/Release/CollectStart and need no actions.

## Next model increments (in order)
1. Multi-waiter collect dedup (pending_changes) -- adjudicate F4.
2. Collective-mapping tree topology (required per Q5 ruling).
3. Then the full matrix (still gated on Q7 confirmation).

## Closing-pass rulings (2026-08-23)
Items in scope: (1) tree4 coverage of the interior-node subtree-skip,
(2) liveness configs, (3) lock-window audit of the implementation,
(4) InstanceNewMid exhaustive large-budget verdict.
OUT OF SCOPE per Mike: (5) cross-protocol composition -- "each
PhysicalManager adds its GLOBAL reference locally to keep a copy alive
as long as it is live and that is easy to reason about" (the
INTERNAL_VALID_REF pin is local-only reasoning); (6) external/attached
and unbound instances.
