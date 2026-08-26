# TLA+ Model of the DistributedCollectable Downgrade Protocol

Goal: model-check the downgrade/collection protocol before changing
`runtime/legion/kernel/garbage_collection.cc`. The model must be able to
(a) reproduce the known bug class in the current protocol (validation that
the model is faithful enough to matter), and (b) verify the proposed
redesign against the same invariants. Every proposed protocol change gets
a feature toggle so TLC can attribute which mechanism closes which bug,
and so future proposals (e.g. an abort broadcast) can be checked before
any implementation work.

## Staging

- **Stage 1 (this spec, `Downgrade.tla`)**: one collectable, one
  reference level (GLOBAL -> LOCAL), flat topology (no collective
  mapping) with the owner-space relay. Covers: round tagging, vote
  receipts, ownership versioning, restart forwarding, the registration
  handshake, and reference-covered by-handle sends.
- **Stage 2**: add the VALID level (ValidDistributedCollectable): the
  request-side catch-up rule (`while (to_check < current_state)
  perform_downgrade`), inter-level round ordering, and the lamport-clock
  piggybacking on packed references.
- **Stage 3 (if needed)**: collective-mapping tree relays instead of the
  single owner-space relay, and the out-of-mapping registration routing.

## Modeling rules

1. **One TLA+ action == one `gc_lock`-held handler execution.** The lock
   is the atomicity boundary in the implementation, so it is the
   atomicity boundary in the model. In particular the root's commit
   decision and its own downgrade are ONE action — this is the property
   the stashed two-phase attempt gave up (opening the acquire window),
   and the model keeps it.
2. **Network = a set of in-flight messages, arbitrary delivery order, no
   loss, no duplication** (Realm semantics). Per-pair channel ordering is
   NOT assumed (models messages crossing virtual channels). Reference
   messages carry a serial id so two packed refs to the same target stay
   distinct set elements.
3. **Event-driven fidelity: downgrade checks run only inside handler
   actions.** There is no free-standing "check for downgrade" action.
   If the real code would lose a wakeup (e.g. nothing re-runs
   `check_for_downgrade` after some event), the model loses it too, and
   the leak shows up as a liveness violation. A naive model with
   always-enabled guards would mask this entire bug class.
4. Handlers are total where the real code uses `weak_find`: messages to
   collected replicas are consumed as no-ops. Messages the real code
   would block on (`find_distributed_collectable` waiting for creation)
   are modeled as undeliverable until the replica exists.

## Abstractions and known fidelity gaps

| Real construct | Model | Gap |
|---|---|---|
| gc/valid references | single `refs` counter per node | resource refs collapsed into the `Collect` action (LOCAL -> ABSENT) |
| sent/received global refs | `sent`/`recv` counters + totals in responses | faithful |
| downgrade lamport clocks | `clk`/`pclk`/`bump` per node, piggybacked on ref/resp/restart/upd/regresp messages | modeled in Stage 1 after TLC proved them load-bearing (FINDINGS.md F1) |
| collective mapping | none; owner space relays to `instSet` | Stage 3 |
| DID re-creation on a node | per-node `gen` counter, reset of `died` on spawn | registration messages carry `gen` so cross-life staleness is modeled |
| Realm event trigger (old registration "done" event) | direct write of `reg` flag at the owner action | trigger propagation delay not modeled (old mode only) |
| acquire chase starvation | bounded by a hop budget on the message | real code prevents this with the ordered REFERENCE virtual channel; noted, not modeled |
| by-handle request/response | `Spawn` action at the owner + spawn message | requester-side failure handling (defunct arm) abstracted |

Bounded constants (`MaxPacks`, `MaxAcqs`, `MaxRounds`, `MaxSpawns`) make
the state space finite. A liveness violation that is only "rounds
exhausted at MaxRounds" is an artifact — raise the bound and re-run.

## Feature toggles (all FALSE ~= the current protocol)

| Toggle | Mechanism | Bug it is meant to close |
|---|---|---|
| `RoundTagging` | requests/responses carry round ids; stale ones dropped or answered not-ready | stale/duplicate responses counted toward a newer round |
| `ReceiptChecks` | success applies only if the receiver voted ready in that exact round and is PENDING | unsolicited success force-downgrading a fresh/non-voting replica |
| `OwnershipVersioning` | ownership adopted only from strictly newer versions; restarts carry the sender's version and are parked, not forwarded, when the receiver knows no more; a PENDING receiver of a DowngradeUpdate rolls back to GLOBAL (transfer proves its voted round failed, FINDINGS.md F4b) | ownership clobber by reordered messages; permanent restart ping-pong; owner wedged in PENDING |
| `RestartForwarding` | restarts forwarded to the believed owner (version-tagged) instead of dropped; parked in `pending_downgrade_restart` during an active round | dropped restarts leaking the object |
| `RegistrationGate` | a replica cannot vote ready until its registration handshake completes | fresh replica voting in / being downgraded by a round the owner tallied without it |
| `RegHandshake` | registration returns (owner, version, last round, clock); `defunct` reply when the object is already gone; re-kick of the downgrade check after completion; registration NUDGES the downgrade owner (fire-and-forget restart) since a new instance is invisible to any round the owner space already voted in (FINDINGS.md F4a) | replica with stale ownership beliefs; registration wedging on a dead owner; lost wakeup after registration; information about new instances dying at the owner space |
| `CoveredByHandle` | by-handle responses carry a counted CREATION REFERENCE (the replica is born holding it, the requester releases it); merely covering the send is insufficient (FINDINGS.md F2) | referenceless/zombie replica materialized while or after the object dies |
| `FlagVeto` | an unpack during an active round sets `pending_downgrade_restart`, and EVERY ready vote (leaf, relay, root decision) refuses while it is set | count cancellation through an unregistered replica (FINDINGS.md F3), which neither counts nor clocks catch |

## Invariants (safety)

- `SafeRefs` — a node holding references is never LOCAL or collected.
- `DiedStaysDead` — a replica that reached LOCAL never returns to
  GLOBAL/PENDING within the same lifetime (resurrection).
- `DeadOwnerClean` — if the owner-space replica has reached LOCAL, then
  no node holds references, no reference message is in flight, and every
  replica is PENDING/LOCAL/ABSENT (i.e. a commit was globally justified).
- `OwnerUnique` — at most one node believes itself the downgrade owner.

## Properties (liveness, small configs, with weak fairness on message
delivery, reference drops, and collection — creation actions unfair)

- `EventualCollection` — once reference creation stops (all creation
  actions are budget-bounded), every replica is eventually collected.
  This is the property the leak bugs (dropped restarts, ownership
  clobber, registration wedge, lost wakeups) violate.

## Validation matrix (expected TLC results)

| Config | Expectation |
|---|---|
| `DowngradeOld.cfg` (all toggles FALSE) | violates `SafeRefs` and/or `DiedStaysDead` (forced success on a non-voter; update resurrection); `EventualCollection` fails (dropped restarts / registration wedge) |
| `DowngradeNew.cfg` (all toggles TRUE) | all invariants hold |
| `DowngradeNewLive.cfg` | `EventualCollection` holds |
| single-toggle bisections | attribute each violation to the mechanism that prevents it |

## How to run

```
JAVA=/opt/homebrew/opt/openjdk/bin/java
JAR=/Users/mebauer/legion/tla/tla2tools.jar
cd /Users/mebauer/legion/tla/downgrade
$JAVA -XX:+UseParallelGC -cp $JAR tlc2.TLC -deadlock -workers auto -config DowngradeNew.cfg Downgrade.tla
```

(`-deadlock` disables TLC's deadlock check: the fully-collected state has
no enabled actions by design.)
