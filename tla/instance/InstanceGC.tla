---------------------------- MODULE InstanceGC ----------------------------
(***************************************************************************)
(* Model of the PhysicalManager instance-collection protocol              *)
(* (instances/physical.{h,cc,inl}): the NON-MONOTONIC valid/collectable   *)
(* protocol arbitrating mappers acquiring instances against the garbage   *)
(* collector reclaiming them. Distinct from tla/downgrade (the            *)
(* DistributedCollectable downgrade protocol).                            *)
(*                                                                        *)
(* Modeling rules (as in tla/downgrade):                                  *)
(*   - one action == one inst_lock-held handler execution; the lock-free  *)
(*     CAS acquire fast path is its own atomic action (the atomic counter *)
(*     linearizes it against the lock holders)                            *)
(*   - network == set of in-flight messages, arbitrary order, no loss     *)
(*   - flat topology (collective-mapping trees are a later stage)         *)
(*   - impl Fatal errors / asserts == the `fatal` flag (invariant NoFatal *)
(*     checks they are unreachable)                                       *)
(*   - gc_events == user tokens; deletion must gather all live tokens     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  Nodes,          \* address spaces
  Owner,          \* the owner node of the instance (in Nodes)
  TreeNodes,      \* nodes of the collective_mapping tree (Owner is a
                  \* member; {Owner} alone models a non-collective
                  \* manager). Tree replicas exist from creation. The
                  \* tree shape is a binary heap over a fixed
                  \* linearization of TreeNodes rooted at the owner.
  MaxAcqs,        \* bound on acquire attempts
  MaxPacks,       \* bound on packed valid references
  MaxRounds,      \* bound on collection rounds
  MaxSpawns,      \* bound on replica creations
  MaxUsers,       \* bound on recorded instance users (gc_events)
  \* ---- toggles ----
  ClockCheck,     \* the collect_lamport_clock machinery (recently added):
                  \* FALSE models the protocol before the clock was added
  ContractHeld,   \* acquire discipline honored: FALSE enables an
                  \* uncovered add_valid_ref at a remote (the caller bug
                  \* the LEGION_DEBUG messages exist to catch)
  SpawnPoison,    \* the F3 fix (same technique as the downgrade
                  \* protocol's update_remote_instances poison): sending
                  \* a manager while the owner's collection round is in
                  \* flight fails the round, so it retries against the
                  \* full instance list. In the implementation:
                  \* pack_garbage_collection_state at a PENDING owner
                  \* bumps failed_collection_count.
  SeparateAccums, \* the F2 fix: remote count reports fold into
                  \* round-local accumulators (downgrade-style
                  \* total_sent/total_received) instead of the primary
                  \* counters, and the snapshot/restore machinery is
                  \* deleted. FALSE models master (expect the F2
                  \* violation).
  RecordCover,    \* the record_instance_user caller discipline (the
                  \* CORRECTED F1 contract, 2026-08-23): a recorded user
                  \* is covered by a valid reference held on SOME node
                  \* (not necessarily the recording one -- e.g. a view
                  \* holds the manager valid remotely while its copy
                  \* user registers at the owner), and the cover is held
                  \* until the record is applied. FALSE allows uncovered
                  \* records (the original F1 trace).
  WaiterFix       \* the F4 fix (proposed, to adjudicate): every wake
                  \* path decrements pending_changes; a round is decided
                  \* exactly once (a decided flag prevents later wakers
                  \* from re-deciding); and a new round may only start
                  \* once the previous round's waiters have fully
                  \* drained. FALSE models master: the VALID/COLLECTABLE
                  \* and COLLECTED wake paths leak the counter
                  \* (physical.cc:1586-1596, 1698-1703), wakers re-run
                  \* the decision switch, and a new round can strand the
                  \* previous round's waiters.

ASSUME /\ Owner \in Nodes
       /\ Owner \in TreeNodes /\ TreeNodes \subseteq Nodes
       /\ Cardinality(TreeNodes) <= 8
       /\ MaxAcqs \in Nat /\ MaxPacks \in Nat /\ MaxRounds \in Nat
       /\ MaxSpawns \in Nat /\ MaxUsers \in Nat

Max(a, b) == IF a >= b THEN a ELSE b

\* Symmetry only over nodes outside the tree (the tree shape
\* distinguishes its members)
Symm == Permutations(Nodes \ TreeNodes)

\* Binary-heap tree over a fixed linearization of TreeNodes: index 1 is
\* the owner; children of index i are 2i and 2i+1
NT == Cardinality(TreeNodes)
TreeSeq == CHOOSE s \in [1..NT -> TreeNodes] :
             /\ s[1] = Owner
             /\ \A i, j \in 1..NT : (i # j) => (s[i] # s[j])
Idx(n) == CHOOSE i \in 1..NT : TreeSeq[i] = n
Children(n) ==
  IF n \in TreeNodes
  THEN {TreeSeq[c] : c \in {x \in {2 * Idx(n), 2 * Idx(n) + 1} : x <= NT}}
  ELSE {}
\* ancestors by repeated halving of the heap index (NT <= 8 => depth 3)
AncIdx(j) == {j, j \div 2, j \div 4, j \div 8} \ {0}
Desc(n) ==
  IF n \in TreeNodes
  THEN {m \in TreeNodes : m # n /\ Idx(n) \in AncIdx(Idx(m))}
  ELSE {}

\* VALID_GC_STATE, COLLECTABLE_GC_STATE, PENDING_COLLECTED_GC_STATE,
\* COLLECTED_GC_STATE; Absent == no replica on this node yet
States == {"Absent", "Valid", "Coll", "Pend", "Dead"}

Users == 1..MaxUsers

\* snapS/snapR: master's count snapshot for the failure restore
\* tS/tR: the F2 fix's round-local accumulators for remote reports
\* gen: round generation (a fresh fan-out); decided: WaiterFix's
\* decide-once flag. fails is the SHARED failed_collection_count member
\* (reset only at round start, so stale responses pollute it -- faithful)
NoRound == [act |-> FALSE, gen |-> 0, decided |-> FALSE, wait |-> {},
            fails |-> 0, snapS |-> 0, snapR |-> 0, tS |-> 0, tR |-> 0]

\* Waiters of the CURRENT round: the starter (took the count snapshot,
\* master restores only through it) and joiners. Waiters of an ORPHANED
\* round: one a new fan-out overwrote while they were still waiting;
\* owait counts the orphaned round's still-outstanding responses (their
\* captured collection_ready triggers when it reaches zero).
\* SCOPE: at most one orphaned round at a time (CollectStart requires
\* the orphan drained before overwriting another live round).
NoWaiters == [s |-> 0, j |-> 0]
NoOrphans == [s |-> 0, j |-> 0, owait |-> {}]

VARIABLES
  node,      \* per-node replica state
  instSet,   \* owner's remote_instances (registered replicas)
  rnd,       \* the owner's current collection round state
  pend,      \* waiters of the current round (collect() callers blocked
             \* on collection_ready): [s: starter count, j: joiners]
  orph,      \* waiters of an orphaned (overwritten) round + its
             \* outstanding response count
  pcs,       \* pending_changes: ONE shared counter spanning rounds
             \* (faithful to the implementation member)
  deleted,   \* the owner performed the deletion (perform_deletion ran)
  userState, \* Users -> unused | live | done  (gc_events lifecycle)
  msgs,
  budget,
  fatal      \* an impl Fatal error / assert fired
vars == <<node, instSet, rnd, pend, orph, pcs, deleted, userState, msgs,
          budget, fatal>>

(***************************************************************************)
(* Messages                                                               *)
(***************************************************************************)
MsgBase == [t |-> "none", dst |-> Owner, src |-> Owner, clk |-> 0,
            s |-> 0, r |-> 0, mm |-> FALSE, uset |-> {}, ost |-> "Coll",
            u |-> 0, gen |-> 0]

Mk(type, over) ==
  [f \in DOMAIN MsgBase |->
     IF f = "t" THEN type
     ELSE IF f \in DOMAIN over THEN over[f] ELSE MsgBase[f]]

SpawnMsg(d, st)      == Mk("spawn",   [dst |-> d, ost |-> st])
VrefMsg(d, s, c)     == Mk("vref",    [dst |-> d, src |-> s, clk |-> c])
AcqReqMsg(s)         == Mk("acqreq",  [src |-> s])
AcqGrantMsg(d)       == Mk("acqgrant",[dst |-> d])
AcqAckMsg(s)         == Mk("acqack",  [src |-> s])
AcqFailMsg(d)        == Mk("acqfail", [dst |-> d])
GcAcqMsg(d, c, g)    == Mk("gcacq",   [dst |-> d, clk |-> c, gen |-> g])
GcFailMsg(s, g)      == Mk("gcfail",  [src |-> s, gen |-> g])
GcDoneMsg(s, sv, rv, c, m, us, g) ==
  Mk("gcdone", [src |-> s, s |-> sv, r |-> rv, clk |-> c, mm |-> m,
                uset |-> us, gen |-> g])
NotifyMsg(d)         == Mk("notify",  [dst |-> d])
RecUserMsg(uu, rr)   == Mk("recuser", [u |-> uu, src |-> rr])

(***************************************************************************)
(* Initial state: the owner's replica exists COLLECTABLE (constructor     *)
(* default); instances become valid through acquires.                     *)
(***************************************************************************)
InitNode(n) ==
  [st    |-> IF n \in TreeNodes THEN "Coll" ELSE "Absent",
   refs  |-> 0, pins |-> 0, sent |-> 0, recv |-> 0,
   clk   |-> 0, pclk |-> 0, bump |-> FALSE,
   users |-> {}]

Init == /\ node = [n \in Nodes |-> InitNode(n)]
        /\ instSet = {}
        /\ rnd = NoRound
        /\ pend = NoWaiters
        /\ orph = NoOrphans
        /\ pcs = 0
        /\ deleted = FALSE
        /\ userState = [u \in Users |-> "unused"]
        /\ msgs = {}
        /\ budget = [acqs |-> 0, packs |-> 0, rounds |-> 0, spawns |-> 0,
                     users |-> 0]
        /\ fatal = FALSE

(***************************************************************************)
(* Replica creation: managers are requested from and sent by the owner    *)
(* (find_or_request_instance_manager); the payload state follows          *)
(* pack_garbage_collection_state and the target is pre-registered in      *)
(* remote_instances at send time. Replicas born mid-round are born Pend.  *)
(***************************************************************************)
PayloadOf(st) == IF st \in {"Valid", "Coll"} THEN "Coll" ELSE st

Spawn(m) ==
  /\ budget.spawns < MaxSpawns
  /\ m \notin TreeNodes
  /\ node[m].st = "Absent"
  /\ m \notin instSet
  /\ node[Owner].st # "Absent"
  /\ instSet' = instSet \cup {m}
  /\ msgs' = msgs \cup {SpawnMsg(m, PayloadOf(node[Owner].st))}
  /\ budget' = [budget EXCEPT !.spawns = @ + 1]
  /\ IF SpawnPoison /\ node[Owner].st = "Pend"
     THEN \* F3 fix: the new replica escaped the in-flight round's
          \* fan-out, so poison the round; it will retry against the
          \* full instance list
          rnd' = [rnd EXCEPT !.fails = @ + 1]
     ELSE UNCHANGED rnd
  /\ UNCHANGED <<node, pend, orph, pcs, deleted, userState, fatal>>

RecvSpawn(m) ==
  LET d == m.dst IN
  /\ node' = [node EXCEPT ![d].st = m.ost]
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget, fatal>>

(***************************************************************************)
(* Mapper side: acquire_instance / acquire_internal. One action covers    *)
(* the CAS fast path (refs > 0) and the locked slow path.                 *)
(*   VALID: refs++.  COLLECTABLE: local flip to VALID (instance reuse,    *)
(*   the common fast case -- no messages).  PENDING at the owner: save    *)
(*   it from the collector.  PENDING at a remote: ask the owner.          *)
(*   Acquires at Dead fail without state change (not modeled: stutter).   *)
(***************************************************************************)
AcquireTry(n) ==
  /\ budget.acqs < MaxAcqs
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ \/ /\ node[n].st \in {"Valid", "Coll"}
        /\ node' = [node EXCEPT ![n].st = "Valid", ![n].refs = @ + 1]
        /\ UNCHANGED msgs
     \/ /\ node[n].st = "Pend"
        /\ n = Owner
        /\ node' = [node EXCEPT ![n].st = "Valid", ![n].refs = @ + 1]
        /\ UNCHANGED msgs
     \/ /\ node[n].st = "Pend"
        /\ n # Owner
        /\ msgs' = msgs \cup {AcqReqMsg(n)}
        /\ UNCHANGED node
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, fatal>>

\* The owner arbitrates a remote acquire during PENDING: it acquires its
\* own covering valid reference (REMOTE_DID_REF, modeled as a pin that
\* only the ack can release) -- possibly saving the instance from an
\* in-flight round -- grants, and holds the cover until the ack.
RecvAcqReq(m) ==
  /\ IF node[Owner].st \in {"Valid", "Coll", "Pend"}
     THEN /\ node' = [node EXCEPT ![Owner].st = "Valid",
                                  ![Owner].refs = @ + 1,
                                  ![Owner].pins = @ + 1]
          /\ msgs' = (msgs \ {m}) \cup {AcqGrantMsg(m.src)}
          /\ UNCHANGED fatal
     ELSE \* owner already Dead: the acquire fails
          /\ msgs' = (msgs \ {m}) \cup {AcqFailMsg(m.src)}
          /\ UNCHANGED <<node, fatal>>
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

\* GarbageCollectionAcquireResponse: an UNCOUNTED reference add at the
\* requester (covered by the owner's held reference until the ack)
RecvAcqGrant(m) ==
  LET d == m.dst IN
  /\ IF node[d].st = "Dead"
     THEN /\ fatal' = TRUE   \* add_valid_reference at COLLECTED
          /\ UNCHANGED node
     ELSE /\ node' = [node EXCEPT ![d].st = "Valid", ![d].refs = @ + 1]
          /\ UNCHANGED fatal
  /\ msgs' = (msgs \ {m}) \cup {AcqAckMsg(d)}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

RecvAcqAck(m) ==
  /\ IF node[Owner].st # "Valid" \/ node[Owner].pins = 0
     THEN /\ fatal' = TRUE   \* cover must still be held
          /\ UNCHANGED node
     ELSE /\ node' = [node EXCEPT
                        ![Owner].refs = @ - 1,
                        ![Owner].pins = @ - 1,
                        ![Owner].st = IF node[Owner].refs = 1 THEN "Coll"
                                      ELSE @]
          /\ UNCHANGED fatal
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

RecvAcqFail(m) ==
  LET d == m.dst IN
  /\ IF node[d].st \in {"Pend", "Dead"}
     THEN /\ node' = [node EXCEPT ![d].st = "Dead"]
          /\ UNCHANGED fatal
     ELSE /\ fatal' = TRUE   \* assert PENDING || COLLECTED
          /\ UNCHANGED node
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

\* remove_valid_reference -> notify_invalid at zero. The application can
\* only release references it holds, never a handler's covering pin.
Release(n) ==
  /\ node[n].refs > node[n].pins
  /\ node' = [node EXCEPT
                ![n].refs = @ - 1,
                ![n].st = IF node[n].refs = 1 THEN "Coll" ELSE @]
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, msgs, budget, fatal>>

(***************************************************************************)
(* Packed valid references (piggybacked on analysis messages):            *)
(* pack_valid_ref asserts VALID and requires a held covering reference;   *)
(* counted with the lamport clock stamped on the pack. The receive        *)
(* models unpack_valid_ref plus its paired reference add as one atomic    *)
(* step (fidelity question Q1 in MODEL_PLAN.md).                          *)
(***************************************************************************)
Pack(n, m) ==
  /\ budget.packs < MaxPacks
  /\ n # m
  /\ node[n].st = "Valid"
  /\ node[n].refs > 0
  /\ m \in instSet \cup TreeNodes
  /\ LET c2 == IF ClockCheck /\ node[n].bump THEN node[n].clk + 1
               ELSE node[n].clk
     IN /\ node' = [node EXCEPT ![n].clk = c2, ![n].bump = FALSE,
                                ![n].sent = @ + 1]
        /\ msgs' = msgs \cup {VrefMsg(m, n, c2)}
  /\ budget' = [budget EXCEPT !.packs = @ + 1]
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, fatal>>

RecvVref(m) ==
  LET d == m.dst IN
  /\ node[d].st # "Absent"   \* blocking find_or_request
  /\ IF node[d].st = "Dead"
     THEN \* the paired add raises the "internal garbage collection
          \* race" Fatal error at COLLECTED
          /\ fatal' = TRUE
          /\ UNCHANGED node
     ELSE /\ node' = [node EXCEPT ![d].st = "Valid", ![d].refs = @ + 1,
                                  ![d].recv = @ + 1,
                                  ![d].clk = Max(@, m.clk)]
          /\ UNCHANGED fatal
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

\* The caller bug the LEGION_DEBUG messages exist to catch: an
\* add_valid_ref at a remote that is covered by NOTHING -- no acquire, no
\* packed reference. Enabled only when ContractHeld is FALSE; expected to
\* break safety, proving the acquire discipline is load-bearing.
UncoveredAdd(n) ==
  /\ ~ContractHeld
  /\ budget.acqs < MaxAcqs
  /\ n # Owner
  /\ node[n].st \in {"Coll", "Pend"}
  /\ node[n].refs = 0
  /\ node' = [node EXCEPT ![n].st = "Valid", ![n].refs = @ + 1]
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, msgs, fatal>>

(***************************************************************************)
(* GC side. CollectStart covers the MemoryManager driver, eager           *)
(* collection, and remote-initiated GarbageCollectionRequest (all reduce  *)
(* to: the owner attempts a collection at a nondeterministic time).       *)
(***************************************************************************)
(* GC side. CollectStart covers the MemoryManager driver, eager           *)
(* collection, and remote-initiated GarbageCollectionRequest. A caller    *)
(* that finds a collection already pending JOINS it (CollectJoin,        *)
(* pending_changes++). Every collect() caller becomes a WAITER blocked   *)
(* on its captured collection_ready; each wakes independently and        *)
(* re-runs the decision switch (physical.cc:1581-1706) -- the wake       *)
(* interleavings are where the F4 hazards live.                          *)
(***************************************************************************)
CollectStart ==
  /\ budget.rounds < MaxRounds
  /\ node[Owner].st = "Coll"
  \* WaiterFix: a new round only starts once the previous round's
  \* waiters have fully drained
  /\ ((~WaiterFix) \/ (pcs = 0))
  \* SCOPE restriction: at most one orphaned round outstanding
  /\ orph.s + orph.j + Cardinality(orph.owait) = 0
  /\ LET c2 == node[Owner].clk + 1 IN
     IF instSet = {} /\ TreeNodes = {Owner}
     THEN \* no remote instances: start + decision under a single lock
          \* hold (collection_ready never exists). pending_changes nets
          \* to zero on the failing path -- unless a leaked counter
          \* (master) makes `--pending_changes == 0` miss, parking the
          \* owner at PENDING.
          IF node[Owner].sent = node[Owner].recv
          THEN /\ node' = [node EXCEPT ![Owner].st = "Dead",
                                       ![Owner].clk = c2,
                                       ![Owner].pclk = c2]
               /\ deleted' = TRUE
               /\ UNCHANGED <<rnd, pend, orph, pcs, msgs>>
          ELSE /\ node' = [node EXCEPT
                             ![Owner].st = IF WaiterFix \/ (pcs = 0)
                                           THEN "Coll" ELSE "Pend",
                             ![Owner].clk = c2, ![Owner].pclk = c2,
                             ![Owner].bump = ClockCheck]
               /\ UNCHANGED <<rnd, pend, orph, pcs, msgs, deleted>>
     ELSE \* fan out a fresh round; any waiters still blocked on the
          \* previous round (master's leaks) are stranded against its
          \* overwritten state and become orphans
          /\ node' = [node EXCEPT ![Owner].st = "Pend",
                                  ![Owner].clk = c2, ![Owner].pclk = c2,
                                  ![Owner].bump = ClockCheck]
          /\ rnd' = [act |-> TRUE, gen |-> rnd.gen + 1, decided |-> FALSE,
                     wait |-> (TreeNodes \ {Owner}) \cup instSet,
                     fails |-> 0,
                     snapS |-> node[Owner].sent, snapR |-> node[Owner].recv,
                     tS |-> 0, tR |-> 0]
          /\ orph' = [s |-> pend.s, j |-> pend.j, owait |-> rnd.wait]
          /\ pend' = [s |-> 1, j |-> 0]
          /\ pcs' = pcs + 1
          \* direct fan-out to non-tree instances; tree fan-out goes
          \* through the owner's tree children only (each committed
          \* tree node forwards to its own children)
          /\ msgs' = msgs \cup
               {GcAcqMsg(m, c2, rnd.gen + 1) :
                  m \in Children(Owner) \cup instSet}
          /\ UNCHANGED deleted
  /\ budget' = [budget EXCEPT !.rounds = @ + 1]
  /\ UNCHANGED <<instSet, userState, fatal>>

\* A collect() call finding a collection already pending joins it
\* (physical.cc:1566-1572). Under master this can also join a COMPLETED
\* round parked at PENDING by a leaked counter -- the phantom join.
CollectJoin ==
  /\ budget.rounds < MaxRounds
  /\ node[Owner].st = "Pend"
  /\ pcs' = pcs + 1
  /\ pend' = [pend EXCEPT !.j = @ + 1]
  /\ budget' = [budget EXCEPT !.rounds = @ + 1]
  /\ UNCHANGED <<node, instSet, rnd, orph, deleted, userState, msgs, fatal>>

\* acquire_collect at a remote (plus the messages the round waits on,
\* folded into one atomic gcdone: counts+clock report and the gc_events
\* shipped to the owner)
RecvGcAcq(m) ==
  LET d == m.dst IN
  /\ node[d].st # "Absent"   \* blocking find_or_request
  /\ IF node[d].st = "Valid"
     THEN \* a VALID node fails the round and does NOT forward to its
          \* tree children: the whole subtree goes unasked
          /\ msgs' = (msgs \ {m}) \cup {GcFailMsg(d, m.gen)}
          /\ UNCHANGED <<node, fatal>>
     ELSE IF node[d].st = "Dead"
     THEN /\ fatal' = TRUE   \* assert gc_state != COLLECTED
          /\ msgs' = msgs \ {m}
          /\ UNCHANGED node
     ELSE \* Coll or Pend: commit to the round
          LET c2 == Max(node[d].clk, m.clk)
              mm == \/ node[d].sent # node[d].recv
                    \/ ClockCheck /\ (c2 > m.clk)
          IN /\ node' = [node EXCEPT ![d].st = "Pend", ![d].clk = c2,
                                     ![d].bump = ClockCheck,
                                     ![d].users = {}]
             /\ msgs' = (msgs \ {m}) \cup
                          {GcDoneMsg(d, node[d].sent, node[d].recv, c2,
                                     mm, node[d].users, m.gen)} \cup
                          {GcAcqMsg(c, m.clk, m.gen) : c \in Children(d)}
             /\ UNCHANGED fatal
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

\* failed_collection_count is a shared member: stale failures from an
\* overwritten round pollute the current one (conservative, faithful).
\* Wait-set membership is per-round (fresh done events per fan-out).
RecvGcFail(m) ==
  /\ rnd' = [rnd EXCEPT
               !.fails = @ + 1,
               !.wait = IF m.gen = rnd.gen
                        THEN @ \ ({m.src} \cup Desc(m.src)) ELSE @]
  /\ orph' = IF m.gen < rnd.gen
             THEN [orph EXCEPT
                     !.owait = @ \ ({m.src} \cup Desc(m.src))]
             ELSE orph
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<node, instSet, pend, pcs, deleted, userState, budget,
                 fatal>>

RecvGcDone(m) ==
  /\ IF (node[Owner].st = "Dead") /\ (m.mm \/ (m.uset # {}))
     THEN \* the mismatch/record handlers assert gc_state != COLLECTED;
          \* reachable only via stale responses from an orphaned round
          /\ fatal' = TRUE
          /\ UNCHANGED <<node, rnd>>
     ELSE /\ UNCHANGED fatal
          /\ IF SeparateAccums
             THEN \* the F2 fix: remote reports fold into round-local
                  \* accumulators, never into the primary counters
                  /\ node' = [node EXCEPT
                                ![Owner].clk = IF m.mm THEN Max(@, m.clk)
                                               ELSE @,
                                ![Owner].users = @ \cup m.uset]
                  /\ rnd' = [rnd EXCEPT
                               !.tS = IF m.mm THEN @ + m.s ELSE @,
                               !.tR = IF m.mm THEN @ + m.r ELSE @,
                               !.wait = IF m.gen = rnd.gen
                                        THEN @ \ {m.src} ELSE @]
             ELSE \* master: folds into the primaries (undone by restore)
                  /\ node' = [node EXCEPT
                                ![Owner].sent = IF m.mm THEN @ + m.s ELSE @,
                                ![Owner].recv = IF m.mm THEN @ + m.r ELSE @,
                                ![Owner].clk = IF m.mm THEN Max(@, m.clk)
                                               ELSE @,
                                ![Owner].users = @ \cup m.uset]
                  /\ rnd' = [rnd EXCEPT
                               !.wait = IF m.gen = rnd.gen
                                        THEN @ \ {m.src} ELSE @]
  /\ orph' = IF m.gen < rnd.gen
             THEN [orph EXCEPT !.owait = @ \ {m.src}] ELSE orph
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, pend, pcs, deleted, userState, budget>>

DecideFailNow ==
  \/ rnd.fails > 0
  \/ IF SeparateAccums
     THEN (rnd.tS + node[Owner].sent) # (rnd.tR + node[Owner].recv)
     ELSE node[Owner].sent # node[Owner].recv
  \/ ClockCheck /\ (node[Owner].clk > node[Owner].pclk)

(***************************************************************************)
(* One collect() caller wakes from its captured collection_ready and     *)
(* re-runs the decision switch against the CURRENT protocol state.       *)
(* frm = "cur": a waiter of the current round, wakes when the current    *)
(* fan-out has fully responded. frm = "orph": a waiter stranded by an    *)
(* overwriting round, wakes when the OLD round's responses drained --    *)
(* possibly while the NEW round is still collecting responses, in which  *)
(* case it decides on partial state (a master hazard). k = "s" is the    *)
(* starter (holds master's count snapshot), "j" a joiner.                *)
(***************************************************************************)
Wake(frm, k) ==
  /\ IF frm = "cur"
     THEN /\ IF k = "s" THEN pend.s > 0 ELSE pend.j > 0
          /\ rnd.wait = {}
          /\ pend' = IF k = "s" THEN [pend EXCEPT !.s = @ - 1]
                     ELSE [pend EXCEPT !.j = @ - 1]
          /\ UNCHANGED orph
     ELSE /\ IF k = "s" THEN orph.s > 0 ELSE orph.j > 0
          /\ orph.owait = {}
          /\ orph' = IF k = "s" THEN [orph EXCEPT !.s = @ - 1]
                     ELSE [orph EXCEPT !.j = @ - 1]
          /\ UNCHANGED pend
  /\ LET starter == (frm = "cur") /\ (k = "s") IN
     IF node[Owner].st \in {"Valid", "Coll"}
     THEN \* an acquire won this round; master restores through the
          \* starter and LEAKS pending_changes (F4,
          \* physical.cc:1586-1596); the fix decrements + marks decided
          /\ node' = IF starter /\ ~SeparateAccums
                     THEN [node EXCEPT ![Owner].sent = rnd.snapS,
                                       ![Owner].recv = rnd.snapR]
                     ELSE node
          /\ IF WaiterFix
             THEN /\ pcs' = pcs - 1
                  /\ rnd' = [rnd EXCEPT !.decided = TRUE]
             ELSE UNCHANGED <<pcs, rnd>>
          /\ UNCHANGED <<deleted, msgs>>
     ELSE IF node[Owner].st = "Dead"
     THEN \* someone else already deleted (physical.cc:1698-1703);
          \* master leaks the counter here too (benign post-mortem)
          /\ IF WaiterFix
             THEN pcs' = pcs - 1
             ELSE UNCHANGED pcs
          /\ UNCHANGED <<node, rnd, deleted, msgs>>
     ELSE \* PENDING: decide, or consume an already-made decision
          IF WaiterFix /\ rnd.decided
          THEN /\ pcs' = pcs - 1
               /\ node' = IF pcs = 1
                          THEN [node EXCEPT ![Owner].st = "Coll"]
                          ELSE node
               /\ UNCHANGED <<rnd, deleted, msgs>>
          ELSE IF DecideFailNow
          THEN \* round failed: `if (--pending_changes == 0)
               \* gc_state = COLLECTABLE`; master restores the
               \* starter's snapshot (the F2 bug)
               /\ node' = IF starter /\ ~SeparateAccums
                          THEN [node EXCEPT
                                  ![Owner].st = IF pcs = 1 THEN "Coll"
                                                ELSE @,
                                  ![Owner].sent = rnd.snapS,
                                  ![Owner].recv = rnd.snapR]
                          ELSE [node EXCEPT
                                  ![Owner].st = IF pcs = 1 THEN "Coll"
                                                ELSE @]
               /\ pcs' = pcs - 1
               /\ rnd' = IF WaiterFix
                         THEN [rnd EXCEPT !.decided = TRUE] ELSE rnd
               /\ UNCHANGED <<deleted, msgs>>
          ELSE \* success: perform the deletion and notify everyone
               /\ node' = [node EXCEPT ![Owner].st = "Dead"]
               /\ deleted' = TRUE
               /\ msgs' = msgs \cup
                    {NotifyMsg(x) :
                       x \in (TreeNodes \ {Owner}) \cup instSet}
               /\ IF WaiterFix
                  THEN /\ pcs' = pcs - 1
                       /\ rnd' = [rnd EXCEPT !.decided = TRUE]
                  ELSE UNCHANGED <<pcs, rnd>>
  /\ UNCHANGED <<instSet, userState, budget, fatal>>

RecvNotify(m) ==
  LET d == m.dst IN
  /\ node[d].st # "Absent"   \* blocking find
  /\ IF node[d].st \in {"Pend", "Dead"}
     THEN /\ node' = [node EXCEPT ![d].st = "Dead"]
          /\ UNCHANGED fatal
     ELSE /\ fatal' = TRUE   \* assert COLLECTED || PENDING_COLLECTED
          /\ UNCHANGED node
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState, budget>>

(***************************************************************************)
(* Instance users (gc_events). CORRECTED F1 contract (2026-08-23, from a  *)
(* CI counterexample): the recorded user is covered by a valid reference *)
(* held on SOME node r -- NOT necessarily the recording node n, whose    *)
(* local state may be COLLECTABLE (e.g. a remote view holds the manager  *)
(* valid and its copy-user registration lands at the manager's owner).   *)
(* A record at a PENDING remote forwards to the owner                    *)
(* (record_instance_user_internal's send branch); the cover is PINNED    *)
(* until the forward is applied, which is what keeps the collection      *)
(* decision from committing while the record is in flight (the round    *)
(* fails at the covering node). Uncovered records (RecordCover FALSE)    *)
(* reproduce the original F1 violation.                                  *)
(***************************************************************************)
RecordUser(n, r, u) ==
  /\ budget.users < MaxUsers
  /\ userState[u] = "unused"
  /\ node[n].st \in {"Valid", "Coll", "Pend"}
  /\ (RecordCover => (node[r].refs > node[r].pins))
  /\ userState' = [userState EXCEPT ![u] = "live"]
  /\ IF (n # Owner) /\ (node[n].st = "Pend")
     THEN /\ msgs' = msgs \cup {RecUserMsg(u, r)}
          /\ node' = IF RecordCover
                     THEN [node EXCEPT ![r].pins = @ + 1] ELSE node
     ELSE /\ node' = [node EXCEPT ![n].users = @ \cup {u}]
          /\ UNCHANGED msgs
  /\ budget' = [budget EXCEPT !.users = @ + 1]
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, fatal>>

\* GarbageCollectionRecordEvent at the owner; unpins the cover
RecvRecUser(m) ==
  /\ IF node[Owner].st = "Dead"
     THEN \* assert gc_state != COLLECTED; unreachable when covered
          /\ fatal' = TRUE
          /\ UNCHANGED node
     ELSE /\ node' = [node EXCEPT
                        ![Owner].users = @ \cup {m.u},
                        ![m.src].pins = IF RecordCover /\ @ > 0
                                        THEN @ - 1 ELSE @]
          /\ UNCHANGED fatal
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, rnd, pend, orph, pcs, deleted, userState,
                 budget>>

\* The user's ApEvent triggers (the use completes)
UserDone(u) ==
  /\ userState[u] = "live"
  /\ userState' = [userState EXCEPT ![u] = "done"]
  /\ UNCHANGED <<node, instSet, rnd, pend, orph, pcs, deleted, msgs, budget, fatal>>

(***************************************************************************)
(* Next                                                                    *)
(***************************************************************************)
Recv(m) ==
  CASE m.t = "spawn"    -> RecvSpawn(m)
    [] m.t = "vref"     -> RecvVref(m)
    [] m.t = "acqreq"   -> RecvAcqReq(m)
    [] m.t = "acqgrant" -> RecvAcqGrant(m)
    [] m.t = "acqack"   -> RecvAcqAck(m)
    [] m.t = "acqfail"  -> RecvAcqFail(m)
    [] m.t = "gcacq"    -> RecvGcAcq(m)
    [] m.t = "gcfail"   -> RecvGcFail(m)
    [] m.t = "gcdone"   -> RecvGcDone(m)
    [] m.t = "notify"   -> RecvNotify(m)
    [] m.t = "recuser"  -> RecvRecUser(m)
    [] OTHER            -> FALSE

ProgressNext ==
  \/ \E m \in msgs : Recv(m)
  \/ \E n \in Nodes : Release(n)
  \/ \E f \in {"cur", "orph"}, k \in {"s", "j"} : Wake(f, k)
  \/ \E u \in Users : UserDone(u)

CreateNext ==
  \/ \E n \in Nodes : AcquireTry(n) \/ UncoveredAdd(n)
  \/ \E n \in Nodes, m \in Nodes : Pack(n, m)
  \/ \E m \in Nodes : Spawn(m)
  \/ CollectStart \/ CollectJoin
  \/ \E n \in Nodes, r \in Nodes, u \in Users : RecordUser(n, r, u)

Next == ProgressNext \/ CreateNext

Spec == Init /\ [][Next]_vars /\ WF_vars(ProgressNext)

(***************************************************************************)
(* Invariants                                                              *)
(***************************************************************************)
CountBound == (MaxPacks + 1) * (MaxRounds + 2)
RefBound == MaxAcqs + MaxPacks + 2
ClkBound == MaxRounds + MaxPacks + 1

TypeOK ==
  /\ node \in [Nodes ->
       [st: States, refs: 0..RefBound, pins: 0..RefBound,
        sent: 0..CountBound,
        recv: 0..CountBound, clk: 0..ClkBound, pclk: 0..ClkBound,
        bump: BOOLEAN, users: SUBSET Users]]
  /\ instSet \subseteq (Nodes \ {Owner})
  /\ rnd \in [act: BOOLEAN, gen: 0..MaxRounds, decided: BOOLEAN,
              wait: SUBSET Nodes,
              fails: 0..(Cardinality(Nodes) * (MaxRounds + 1) + MaxSpawns),
              snapS: 0..CountBound, snapR: 0..CountBound,
              tS: 0..CountBound, tR: 0..CountBound]
  /\ pend \in [s: 0..1, j: 0..MaxRounds]
  /\ orph \in [s: 0..1, j: 0..MaxRounds, owait: SUBSET Nodes]
  /\ pcs \in 0..(MaxRounds + 1)
  /\ deleted \in BOOLEAN
  /\ userState \in [Users -> {"unused", "live", "done"}]
  /\ fatal \in BOOLEAN

\* No impl Fatal error or assert is reachable
NoFatal == ~fatal

\* Local validity bookkeeping: references are only held at VALID
RefsImplyValid ==
  \A n \in Nodes : node[n].refs > 0 => node[n].st = "Valid"

\* The soundness of the mapper/GC race: a committed deletion implies
\* nobody holds a valid reference, nobody is still (re)usable, and no
\* reference-carrying message is in flight
SafeDeletion ==
  deleted =>
    /\ \A n \in Nodes : node[n].refs = 0
    /\ \A n \in Nodes : node[n].st \in {"Absent", "Pend", "Dead"}
    /\ ~\E m \in msgs : m.t \in {"vref", "acqgrant"}

\* No use-after-free: at deletion every live user token has been
\* gathered at the owner (the deferred deletion waits on exactly these)
UsersGathered ==
  deleted =>
    \A u \in Users :
      userState[u] = "live" => u \in node[Owner].users

\* COLLECTED is terminal at the owner: once deleted, always deleted
\* (non-monotonicity is allowed everywhere EXCEPT out of Dead; Dead
\* resurrection paths set `fatal` instead, so NoFatal covers remotes)
DeadStaysDead ==
  deleted => node[Owner].st = "Dead"

EventualQuiescence ==
  <>[](\/ deleted
       \/ \A m \in msgs : m.t \notin {"gcacq", "gcfail", "gcdone"})

(***************************************************************************)
(* Liveness (budget-robust: no fairness on the GC driver needed, only    *)
(* the WF on ProgressNext already in Spec). WaitersDrain is the F4       *)
(* liveness claim: every collect() caller that joined a round            *)
(* eventually wakes and drains -- master's counter leaks violate it.     *)
(* OwnerUnparks: the owner never stays PENDING forever -- master's       *)
(* missed reset (`--pending_changes == 0` never reached) violates it.    *)
(***************************************************************************)
WaitersDrain == [](pcs > 0 => <>(pcs = 0))

OwnerUnparks ==
  [](node[Owner].st = "Pend" => <>(node[Owner].st # "Pend"))

=============================================================================
