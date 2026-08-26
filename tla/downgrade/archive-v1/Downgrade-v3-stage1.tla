---------------------------- MODULE Downgrade ----------------------------
(***************************************************************************)
(* Stage-1 model of the Legion DistributedCollectable downgrade protocol. *)
(* Scope: one collectable, one reference level (GLOBAL -> LOCAL), flat    *)
(* topology (no collective mapping) with the owner-space relay.           *)
(* See MODEL_PLAN.md for the abstraction rules and fidelity gaps.         *)
(*                                                                        *)
(* Modeling rules:                                                        *)
(*   - one TLA+ action == one gc_lock-held handler execution              *)
(*   - network == set of in-flight messages, arbitrary order, no loss     *)
(*   - downgrade checks run ONLY inside handler actions (event-driven     *)
(*     fidelity: lost wakeups must surface as liveness violations)        *)
(*                                                                        *)
(* The downgrade lamport clocks (clk/pclk/bump) are modeled: TLC          *)
(* demonstrated in a 13-step trace that they are load-bearing (a          *)
(* reference passing THROUGH a node after its ready vote is only caught   *)
(* by the post-vote clock bump poisoning the root's causality check).     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  Nodes,               \* address spaces
  OwnerSpace,          \* the object's owner space (in Nodes)
  MaxPacks,            \* bound on packed references (incl. acquire grants)
  MaxAcqs,             \* bound on acquire attempts (local and remote)
  MaxRounds,           \* bound on round epochs (downgrade lamport clock)
  MaxSpawns,           \* bound on by-handle replica spawns
  \* ---- feature toggles: all FALSE ~= the current protocol ----
  RoundTagging,        \* stale requests/responses detected by round id
  ReceiptChecks,       \* success applies only to the exact round voted in
  OwnershipVersioning, \* ownership adopted only from strictly newer versions
  RestartForwarding,   \* restarts forwarded/parked instead of dropped
  RegistrationGate,    \* cannot vote ready before registration completes
  RegHandshake,        \* registration returns (owner,version); defunct reply
  CoveredByHandle,     \* by-handle responses counted + pre-registered
  FlagVeto             \* unpack mid-round parks a flag; every ready vote
                       \* (leaf, relay, root) refuses while it is set

ASSUME /\ OwnerSpace \in Nodes
       /\ MaxPacks \in Nat /\ MaxAcqs \in Nat
       /\ MaxRounds \in Nat /\ MaxSpawns \in Nat

Max(a, b) == IF a >= b THEN a ELSE b

(* Non-owner nodes are interchangeable: sound for invariant checking      *)
(* (do NOT use with liveness properties).                                 *)
Symm == Permutations(Nodes \ {OwnerSpace})

States == {"Absent", "Global", "Pending", "Local"}

(* Round-collection state at a root or at the owner-space relay.          *)
(* ow   = the downgrade owner of the round                                *)
(* par  = where the aggregate response goes (self at the root)            *)
(* nrdy = not-ready node; equal to ow while everything is still ready     *)
NoRound == [act |-> FALSE, rid |-> 0, ow |-> OwnerSpace, par |-> OwnerSpace,
            wait |-> {}, nrdy |-> OwnerSpace, tS |-> 0, tR |-> 0]

VARIABLES
  node,    \* [Nodes -> per-replica record]
  instSet, \* owner space's remote_instances (monotone)
  msgs,    \* set of in-flight messages
  budget   \* creation budgets: [packs, acqs, spawns]

vars == <<node, instSet, msgs, budget>>

(***************************************************************************)
(* Messages: one uniform record shape so heterogeneous sets are safe.     *)
(* clk carries the sender's downgrade lamport clock where the real        *)
(* protocol piggybacks it (refs, responses, restarts, updates, regresp). *)
(***************************************************************************)
MsgBase == [t |-> "none", dst |-> OwnerSpace, src |-> OwnerSpace,
            rid |-> 0, ow |-> OwnerSpace, ver |-> 0, nrdy |-> OwnerSpace,
            tS |-> 0, tR |-> 0, cand |-> OwnerSpace, id |-> 0, gen |-> 0,
            lastR |-> 0, df |-> FALSE, hops |-> 0, clk |-> 0]

Mk(type, over) ==
  [f \in DOMAIN MsgBase |->
     IF f = "t" THEN type
     ELSE IF f \in DOMAIN over THEN over[f] ELSE MsgBase[f]]

ReqMsg(d, s, rid, ow, v)  == Mk("req",  [dst |-> d, src |-> s, rid |-> rid,
                                         ow |-> ow, ver |-> v])
RespMsg(d, s, rid, nr, ts, tr, c) == Mk("resp", [dst |-> d, src |-> s,
                                         rid |-> rid, nrdy |-> nr,
                                         tS |-> ts, tR |-> tr, clk |-> c])
SuccMsg(d, s, rid)        == Mk("succ", [dst |-> d, src |-> s, rid |-> rid])
UpdMsg(d, s, v, lr, c)    == Mk("upd",  [dst |-> d, src |-> s, ver |-> v,
                                         lastR |-> lr, clk |-> c])
RestartMsg(d, c, v, ck)   == Mk("restart", [dst |-> d, cand |-> c, ver |-> v,
                                         clk |-> ck])
RefMsg(d, s, i, ck)       == Mk("ref",  [dst |-> d, src |-> s, id |-> i,
                                         clk |-> ck])
SpawnMsg(d, c, ow, v, lr, ck) == Mk("spawn", [dst |-> d, cand |-> c,
                                         ow |-> ow, ver |-> v,
                                         lastR |-> lr, clk |-> ck])
UnpinMsg(c, m)            == Mk("unpin", [dst |-> c, src |-> m])
RegMsg(s, g)              == Mk("reg",  [dst |-> OwnerSpace, src |-> s,
                                         gen |-> g])
RegRespMsg(d, g, ow, v, lr, ck, df) == Mk("regresp", [dst |-> d, gen |-> g,
                                         ow |-> ow, ver |-> v, lastR |-> lr,
                                         clk |-> ck, df |-> df])
AcqMsg(d, o, h)           == Mk("acq",  [dst |-> d, cand |-> o, hops |-> h])
DenyMsg(d)                == Mk("deny", [dst |-> d])

(***************************************************************************)
(* Initial state: the owner-space replica exists in GLOBAL holding one    *)
(* reference; nothing else exists.                                        *)
(***************************************************************************)
InitNode(n) ==
  [st    |-> IF n = OwnerSpace THEN "Global" ELSE "Absent",
   refs  |-> IF n = OwnerSpace THEN 1 ELSE 0,
   sent  |-> 0, recv |-> 0,
   reg   |-> (n = OwnerSpace),
   own   |-> OwnerSpace, ver |-> 0,
   lastR |-> 0, voted |-> 0,
   clk   |-> 0, pclk |-> 0, bump |-> FALSE,
   rflag |-> FALSE, gen |-> 0, died |-> FALSE,
   pins  |-> 0,
   rnd   |-> NoRound]

Init == /\ node = [n \in Nodes |-> InitNode(n)]
        /\ instSet = {}
        /\ msgs = {}
        /\ budget = [packs |-> 0, acqs |-> 0, spawns |-> 0]

(***************************************************************************)
(* Helpers                                                                *)
(***************************************************************************)
CanDowngradeF(f, n) ==
  /\ f[n].refs = 0
  /\ f[n].st \in {"Global", "Pending"}
  /\ (RegistrationGate => f[n].reg)

(* check_for_downgrade at a node that believes it is the downgrade owner. *)
(* The round epoch is the bumped lamport clock (as in the real code).     *)
(* Empty participants with balanced counts => immediate local downgrade.  *)
TryRound(n, f, is) ==
  IF /\ f[n].own = n
     /\ f[n].st = "Global"
     /\ f[n].refs = 0
     /\ (RegistrationGate => f[n].reg)
     /\ ~f[n].rnd.act
     /\ Max(f[n].clk, f[n].lastR) < MaxRounds
  THEN LET parts == IF n = OwnerSpace THEN is ELSE {OwnerSpace}
           rid   == Max(f[n].clk, f[n].lastR) + 1
       IN IF parts = {}
          THEN IF f[n].sent = f[n].recv
               THEN [f |-> [f EXCEPT ![n].st = "Local", ![n].died = TRUE],
                     out |-> {}]
               ELSE [f |-> f, out |-> {}]
          ELSE [f |-> [f EXCEPT
                         ![n].lastR = rid,
                         ![n].pclk = rid,
                         ![n].bump = FALSE,
                         ![n].rnd = [act |-> TRUE, rid |-> rid, ow |-> n,
                                     par |-> n, wait |-> parts, nrdy |-> n,
                                     tS |-> 0, tR |-> 0]],
                out |-> {ReqMsg(p, n, rid, n, f[n].ver) : p \in parts}]
  ELSE [f |-> f, out |-> {}]

(***************************************************************************)
(* Spontaneous local actions (reference creation; budget-bounded)         *)
(***************************************************************************)

(* pack_global_ref while holding a reference: the first pack after an     *)
(* accumulate bumps the clock past the voted epoch (bump flag), which is *)
(* what poisons a round that a reference passed through. Owner-side      *)
(* sends record the target in remote_instances BEFORE dispatch.          *)
Pack(n, m) ==
  /\ n # m
  /\ budget.packs < MaxPacks
  /\ node[n].st = "Global"
  /\ node[n].refs > 0
  /\ node[n].reg
  /\ LET c2 == IF node[n].bump THEN node[n].clk + 1 ELSE node[n].clk IN
     /\ budget' = [budget EXCEPT !.packs = @ + 1]
     /\ msgs' = msgs \cup {RefMsg(m, n, budget.packs + 1, c2)}
     /\ IF n = OwnerSpace
        THEN /\ instSet' = instSet \cup {m}
             /\ node' = [node EXCEPT ![n].sent = @ + 1,
                                     ![n].clk = c2, ![n].bump = FALSE,
                                     \* update_remote_instances only runs
                                     \* (and only poisons) for a NEW instance
                                     ![n].rnd.nrdy =
                                       IF node[n].rnd.act /\ m \notin instSet
                                       THEN m ELSE @]
        ELSE /\ instSet' = instSet
             /\ node' = [node EXCEPT ![n].sent = @ + 1,
                                     ![n].clk = c2, ![n].bump = FALSE]

(* check_global_and_increment fast/GLOBAL path: local acquire.            *)
AcquireLocal(n) ==
  /\ budget.acqs < MaxAcqs
  /\ node[n].st = "Global"
  /\ node' = [node EXCEPT ![n].refs = @ + 1]
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ UNCHANGED <<instSet, msgs>>

(* acquire from PENDING: must go through the downgrade owner.             *)
AcqStart(n) ==
  /\ budget.acqs < MaxAcqs
  /\ node[n].st = "Pending"
  /\ node[n].own # n
  /\ msgs' = msgs \cup {AcqMsg(node[n].own, n, Cardinality(Nodes) + 2)}
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ UNCHANGED <<node, instSet>>

(* By-handle send from the owner space. No reference is packed with the   *)
(* handle (design review 2026-08-17): the runtime invariant is that SOME  *)
(* node holds a global reference for the whole window from serialization  *)
(* until the remote operations using the handle complete (e.g. the        *)
(* IndexTask covering its launch space for all remote point tasks).       *)
(* Covered variant: pin a reference on any node c for that window; the    *)
(* unpin token emitted at delivery models the remote-use completion.      *)
(* Uncovered variant: spontaneous referenceless spawn -- reproduces the   *)
(* zombie, validating that the covering invariant is what is load-bearing.*)
Spawn(m, c) ==
  /\ m # OwnerSpace
  /\ node[m].st = "Absent"
  /\ m \notin instSet
  /\ budget.spawns < MaxSpawns
  /\ node[OwnerSpace].st \in {"Global", "Pending"}
  /\ instSet' = instSet \cup {m}
  /\ budget' = [budget EXCEPT !.spawns = @ + 1]
  /\ IF CoveredByHandle
     THEN \* the response is COUNTED like any packed reference and carries
          \* the registration handshake (the replica is born registered).
          \* A PENDING sender may not serve directly: the implementation
          \* uses the pack_global_ref acquire fallback (owner-serialized).
          /\ node[OwnerSpace].st = "Global"
          /\ node[c].refs > node[c].pins
          /\ LET c2 == IF node[OwnerSpace].bump
                       THEN node[OwnerSpace].clk + 1
                       ELSE node[OwnerSpace].clk
             IN /\ node' = [node EXCEPT
                             ![c].pins = @ + 1,
                             ![OwnerSpace].sent = @ + 1,
                             ![OwnerSpace].clk = c2,
                             ![OwnerSpace].bump = FALSE,
                             ![OwnerSpace].rnd.nrdy =
                               IF node[OwnerSpace].rnd.act THEN m ELSE @]
                /\ msgs' = msgs \cup
                     {SpawnMsg(m, c, node[OwnerSpace].own,
                               node[OwnerSpace].ver,
                               node[OwnerSpace].lastR, c2)}
     ELSE /\ node' = [node EXCEPT ![OwnerSpace].rnd.nrdy =
                        IF node[OwnerSpace].rnd.act THEN m ELSE @]
          /\ msgs' = msgs \cup {SpawnMsg(m, c, OwnerSpace, 0, 0, 0)}

(* remove_gc_ref -> can_delete: the only places downgrade checks run are  *)
(* here and inside message handlers (event-driven fidelity).              *)
DropRef(n) ==
  /\ node[n].refs > node[n].pins   \* pinned covering references cannot drop
  /\ budget' = budget
  /\ IF node[n].refs > 1
     THEN /\ node' = [node EXCEPT ![n].refs = @ - 1]
          /\ UNCHANGED <<instSet, msgs>>
     ELSE LET keepFlag == RestartForwarding /\ node[n].rnd.act
              f0 == [node EXCEPT ![n].refs = 0,
                                 ![n].rflag = IF keepFlag THEN @ ELSE FALSE]
              restartOut ==
                IF /\ node[n].rflag /\ ~keepFlag
                   /\ node[n].own # n /\ ~node[n].rnd.act
                THEN {RestartMsg(node[n].own, n, node[n].ver, node[n].clk)}
                ELSE {}
              tr == TryRound(n, f0, instSet)
          IN /\ node' = tr.f
             /\ msgs' = msgs \cup restartOut \cup tr.out
             /\ UNCHANGED instSet

(* LOCAL -> collected (resource references abstracted away).              *)
Collect(n) ==
  /\ node[n].st = "Local"
  /\ ~node[n].rnd.act
  /\ node' = [node EXCEPT ![n].st = "Absent"]
  /\ UNCHANGED <<instSet, msgs, budget>>

(***************************************************************************)
(* Message handlers                                                       *)
(***************************************************************************)

(* A packed reference arrives (also used for acquire grants). Unpacking   *)
(* onto an ABSENT node creates the replica, which then registers. The     *)
(* clock merge on unpack is what lets a later round see the causality.    *)
RecvRef(m) ==
  LET d == m.dst IN
  IF node[d].st = "Absent"
  THEN LET ng == node[d].gen + 1 IN
       /\ node' = [node EXCEPT ![d].st = "Global", ![d].refs = 1,
                               ![d].recv = @ + 1, ![d].reg = FALSE,
                               ![d].gen = ng, ![d].died = FALSE,
                               ![d].own = OwnerSpace, ![d].ver = 0,
                               ![d].voted = 0, ![d].clk = m.clk,
                               ![d].pclk = 0, ![d].bump = FALSE]
       /\ msgs' = (msgs \ {m}) \cup
                    (IF d = OwnerSpace THEN {} ELSE {RegMsg(d, ng)})
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET wasPend == node[d].st = "Pending"
           quiet   == node[d].refs = 0
           newSt   == IF wasPend THEN "Global" ELSE node[d].st
           c2      == Max(node[d].clk, m.clk)
           notify  == node[d].own # d /\ (wasPend \/ quiet)
                      /\ ~node[d].rnd.act
           park    == \/ (node[d].own # d /\ ~(wasPend \/ quiet)
                          /\ ~node[d].rnd.act)
                      \/ (FlagVeto /\ node[d].rnd.act)
       IN /\ node' = [node EXCEPT ![d].st = newSt, ![d].refs = @ + 1,
                                  ![d].recv = @ + 1, ![d].clk = c2,
                                  ![d].voted = IF wasPend THEN 0 ELSE @,
                                  ![d].rflag = @ \/ park]
          /\ msgs' = (msgs \ {m}) \cup
                       (IF notify
                        THEN {RestartMsg(node[d].own, d, node[d].ver, c2)}
                        ELSE {})
          /\ UNCHANGED <<instSet, budget>>

(* By-handle response arrives: replica exists now, registers. Delivery    *)
(* emits the unpin token for the covering reference; its later            *)
(* consumption models the completion of the remote use of the handle.    *)
RecvSpawn(m) ==
  LET d == m.dst
      unpin == IF CoveredByHandle THEN {UnpinMsg(m.cand, d)} ELSE {}
  IN
  IF node[d].st = "Absent"
  THEN LET ng == node[d].gen + 1 IN
       IF CoveredByHandle
       THEN \* counted + pre-registered: recv++, handshake adopted from the
            \* response, no trailing registration message; the standard
            \* quiet-unpack restart notifies the downgrade owner
            /\ node' = [node EXCEPT ![d].st = "Global", ![d].reg = TRUE,
                                    ![d].gen = ng, ![d].died = FALSE,
                                    ![d].recv = @ + 1,
                                    ![d].own = m.ow, ![d].ver = m.ver,
                                    ![d].lastR = m.lastR,
                                    ![d].voted = 0, ![d].clk = m.clk,
                                    ![d].pclk = 0, ![d].bump = FALSE]
            /\ msgs' = (msgs \ {m}) \cup unpin \cup
                 (IF m.ow # d
                  THEN {RestartMsg(m.ow, d, m.ver, m.clk)} ELSE {})
            /\ UNCHANGED <<instSet, budget>>
       ELSE /\ node' = [node EXCEPT ![d].st = "Global", ![d].reg = FALSE,
                                    ![d].gen = ng, ![d].died = FALSE,
                                    ![d].own = OwnerSpace, ![d].ver = 0,
                                    ![d].voted = 0, ![d].clk = 0,
                                    ![d].pclk = 0, ![d].bump = FALSE]
            /\ msgs' = (msgs \ {m}) \cup {RegMsg(d, ng)} \cup unpin
            /\ UNCHANGED <<instSet, budget>>
  ELSE \* replica already exists (ref-created in the interim): the counted
       \* response's reference must still be consumed
       IF CoveredByHandle
       THEN LET quiet == node[d].refs = 0
                notify == node[d].own # d /\ quiet /\ ~node[d].rnd.act
                park == \/ (node[d].own # d /\ ~quiet /\ ~node[d].rnd.act)
                        \/ (FlagVeto /\ node[d].rnd.act)
            IN /\ node' = [node EXCEPT ![d].recv = @ + 1,
                                       ![d].clk = Max(@, m.clk),
                                       ![d].rflag = @ \/ park]
               /\ msgs' = (msgs \ {m}) \cup unpin \cup
                    (IF notify
                     THEN {RestartMsg(node[d].own, d, node[d].ver,
                                      Max(node[d].clk, m.clk))} ELSE {})
               /\ UNCHANGED <<instSet, budget>>
       ELSE /\ msgs' = (msgs \ {m}) \cup unpin
            /\ UNCHANGED <<node, instSet, budget>>

(* Remote use of a by-handle transfer completed: release the covering pin.*)
RecvUnpin(m) ==
  /\ node' = [node EXCEPT ![m.dst].pins = IF @ > 0 THEN @ - 1 ELSE @]
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, budget>>

(* Registration arrives at the owner space.                               *)
(* Old mode: one-way; the owner records the instance and the "done"      *)
(* event trigger is modeled as directly setting the replica's reg flag.  *)
(* New mode: reply with (owner, version, lastR, clk); defunct if the      *)
(* object is already gone. The "hairy case" ownership handoff (owner     *)
(* alone with unbalanced counts) is preserved from the real code. A      *)
(* registration landing during an active collection poisons the round.   *)
RecvReg(m) ==
  LET s == m.src
      O == OwnerSpace
  IN
  IF node[O].st \in {"Global", "Pending"}
  THEN LET hairy == /\ instSet = {} /\ node[O].own = O
                    /\ node[O].sent # node[O].recv
                    /\ ~node[O].rnd.act
           newVer == IF hairy THEN node[O].ver + 1 ELSE node[O].ver
           newOwn == IF hairy THEN s ELSE node[O].own
           poison == node[O].rnd.act /\ node[O].rnd.wait # {}
           is2 == instSet \cup {s}
           f0 == [node EXCEPT
                    ![O].own  = newOwn,
                    ![O].ver  = newVer,
                    ![O].rnd.nrdy = IF poison THEN s ELSE @,
                    ![s].reg  = IF ~RegHandshake /\ m.gen = node[s].gen
                                THEN TRUE ELSE @]
           \* A new instance is invisible to any round the owner space has
           \* already voted in, and its counts may be what a failed round
           \* is missing: registration must nudge the downgrade owner
           \* (the non-blocking replacement for freezing registration).
           nudgeSelf == RegHandshake /\ ~hairy /\ ~poison /\ newOwn = O
           tr == IF nudgeSelf THEN TryRound(O, f0, is2)
                 ELSE [f |-> f0, out |-> {}]
           nudgeOut ==
             IF RegHandshake /\ ~hairy /\ ~poison /\ newOwn # O
             THEN {RestartMsg(newOwn, newOwn, node[O].ver, node[O].clk)}
             ELSE {}
           handshakeOut ==
             IF RegHandshake
             THEN {RegRespMsg(s, m.gen, newOwn, newVer, node[O].lastR,
                              node[O].clk, FALSE)}
             ELSE {}
           hairyOut ==
             IF hairy
             THEN {UpdMsg(s, O, newVer, node[O].lastR, node[O].clk)}
             ELSE {}
       IN /\ instSet' = is2
          /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup handshakeOut \cup hairyOut
                       \cup nudgeOut \cup tr.out
          /\ UNCHANGED budget
  ELSE \* Owner already LOCAL/collected: the blocking find never returns.
       \* Deliberate (design review): under the covering invariant this is
       \* unreachable -- HandlesCovered verifies it -- and if an invariant
       \* is broken elsewhere, a loud hang is the desired tripwire.
       FALSE

(* Registration response at the replica (new mode only).                  *)
RecvRegResp(m) ==
  LET s == m.dst IN
  IF m.gen = node[s].gen /\ node[s].st \in {"Global", "Pending"}
  THEN LET adopt == IF OwnershipVersioning
                    THEN m.ver > node[s].ver
                    ELSE TRUE
           f0 == [node EXCEPT
                    ![s].reg   = TRUE,
                    ![s].own   = IF adopt THEN m.ow ELSE @,
                    ![s].ver   = IF adopt THEN m.ver ELSE @,
                    ![s].clk   = Max(node[s].clk, m.clk),
                    ![s].lastR = Max(node[s].lastR, m.lastR)]
           tr == TryRound(s, f0, instSet)  \* the post-registration re-kick
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>
  ELSE /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>

(* Downgrade request. The real handler waits in                           *)
(* find_distributed_collectable for the replica to exist, so delivery is *)
(* guarded on existence. A request to a LOCAL replica or a stale round   *)
(* id is answered not-ready so the sender's count always resolves. A     *)
(* ready vote requires clk <= epoch (the causality guard) and performs   *)
(* the accumulate: clk = epoch, bump set.                                 *)
RecvReqBody(m) ==
  LET d == m.dst
      stale == RoundTagging /\ m.rid <= node[d].lastR
  IN
  /\ node[d].st # "Absent"     \* models the blocking find
  /\ ~node[d].rnd.act          \* real code asserts no overlapping round
  /\ IF node[d].st = "Local" \/ stale
     THEN /\ msgs' = (msgs \ {m}) \cup
                       {RespMsg(m.src, d, m.rid, d, 0, 0, node[d].clk)}
          /\ UNCHANGED <<node, instSet, budget>>
     ELSE LET adopt == IF OwnershipVersioning
                       THEN m.ver > node[d].ver
                       ELSE TRUE
              own2 == IF adopt THEN m.ow ELSE node[d].own
              ver2 == IF adopt THEN m.ver ELSE node[d].ver
              lr2  == Max(node[d].lastR, m.rid)
              can  == /\ CanDowngradeF(node, d)
                      /\ node[d].clk <= m.rid
                      /\ (FlagVeto => ~node[d].rflag)
          IN
          IF ~can
          THEN /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2,
                                       ![d].rflag = IF FlagVeto THEN FALSE
                                                    ELSE @]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(m.src, d, m.rid, d, 0, 0, node[d].clk)}
               /\ UNCHANGED <<instSet, budget>>
          ELSE IF d = OwnerSpace
          THEN \* relay: fan out to the live remote instances minus the root
               LET kids == instSet \ {m.ow} IN
               IF kids = {}
               THEN /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                            ![d].lastR = lr2,
                                            ![d].st = "Pending",
                                            ![d].voted = m.rid,
                                            ![d].clk = m.rid,
                                            ![d].pclk = m.rid,
                                            ![d].bump = TRUE]
                    /\ msgs' = (msgs \ {m}) \cup
                                 {RespMsg(m.src, d, m.rid, m.ow,
                                          node[d].sent, node[d].recv, m.rid)}
                    /\ UNCHANGED <<instSet, budget>>
               ELSE /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                            ![d].lastR = lr2,
                                            ![d].pclk = m.rid,
                                            ![d].bump = FALSE,
                                            ![d].rnd = [act |-> TRUE,
                                                        rid |-> m.rid,
                                                        ow |-> m.ow,
                                                        par |-> m.src,
                                                        wait |-> kids,
                                                        nrdy |-> m.ow,
                                                        tS |-> 0, tR |-> 0]]
                    /\ msgs' = (msgs \ {m}) \cup
                                 {ReqMsg(k, d, m.rid, m.ow, ver2) : k \in kids}
                    /\ UNCHANGED <<instSet, budget>>
          ELSE \* leaf participant: vote ready (accumulate: clk=epoch, bump)
               /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2,
                                       ![d].st = "Pending",
                                       ![d].voted = m.rid,
                                       ![d].clk = m.rid,
                                       ![d].pclk = m.rid,
                                       ![d].bump = TRUE]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(m.src, d, m.rid, m.ow,
                                     node[d].sent, node[d].recv, m.rid)}
               /\ UNCHANGED <<instSet, budget>>

(* Downgrade response at a root or the relay. The decision is made and    *)
(* the root's own downgrade performed in the SAME atomic action (this is *)
(* the property the two-phase redesign attempt gave up, opening the       *)
(* acquire window; the model keeps it and TLC verifies its sufficiency). *)
(* The causality guard: commit requires the merged clock to still be at  *)
(* the round epoch; any reference that moved after a vote bumped it.     *)
RecvResp(m) ==
  LET r == m.dst
      rd == node[r].rnd
  IN
  /\ rd.act
  /\ m.src \in rd.wait
  /\ (RoundTagging => m.rid = rd.rid)   \* old mode: counts any round's response
  /\ LET clean  == rd.nrdy = rd.ow
         mReady == m.nrdy = rd.ow
         nrdy2  == IF ~mReady THEN m.nrdy ELSE rd.nrdy
         tS2    == IF mReady /\ clean THEN rd.tS + m.tS ELSE rd.tS
         tR2    == IF mReady /\ clean THEN rd.tR + m.tR ELSE rd.tR
         wait2  == rd.wait \ {m.src}
         c2     == Max(node[r].clk, m.clk)
     IN
     IF wait2 # {}
     THEN /\ node' = [node EXCEPT ![r].rnd.wait = wait2,
                                  ![r].rnd.nrdy = nrdy2,
                                  ![r].rnd.tS = tS2, ![r].rnd.tR = tR2,
                                  ![r].clk = c2]
          /\ msgs' = msgs \ {m}
          /\ UNCHANGED <<instSet, budget>>
     ELSE IF rd.par = r
     THEN \* the root: decide, and commit ATOMICALLY with the decision
          LET causal == c2 <= node[r].pclk
              ready  == /\ nrdy2 = r
                        /\ CanDowngradeF(node, r)
                        /\ tS2 + node[r].sent = tR2 + node[r].recv
                        /\ causal
                        /\ (FlagVeto => ~node[r].rflag)
          IN
          IF ready
          THEN LET succTo == IF r = OwnerSpace THEN instSet ELSE {OwnerSpace}
               IN /\ node' = [node EXCEPT ![r].st = "Local",
                                          ![r].died = TRUE,
                                          ![r].voted = 0,
                                          ![r].clk = c2,
                                          ![r].rnd = NoRound]
                  /\ msgs' = (msgs \ {m}) \cup
                               {SuccMsg(k, r, rd.rid) : k \in succTo}
                  /\ UNCHANGED <<instSet, budget>>
          ELSE IF nrdy2 # r
          THEN \* transfer ownership to the not-ready node
               LET v2 == node[r].ver + 1
                   restartOut == IF node[r].rflag
                                 THEN {RestartMsg(nrdy2, r, v2, c2)} ELSE {}
               IN /\ node' = [node EXCEPT ![r].own = nrdy2, ![r].ver = v2,
                                          ![r].rflag = FALSE, ![r].clk = c2,
                                          ![r].bump = TRUE,
                                          ![r].rnd = NoRound]
                  /\ msgs' = (msgs \ {m}) \cup
                               {UpdMsg(nrdy2, r, v2, node[r].lastR, c2)} \cup
                               restartOut
                  /\ UNCHANGED <<instSet, budget>>
          ELSE \* self not-ready: causality violation retries immediately,
               \* a count mismatch waits for a restart (as in the real code)
               LET f0 == [node EXCEPT ![r].rnd = NoRound, ![r].clk = c2,
                                      ![r].bump = TRUE, ![r].rflag = FALSE]
                   retry == ~causal \/ node[r].rflag
                   tr == IF retry THEN TryRound(r, f0, instSet)
                         ELSE [f |-> f0, out |-> {}]
               IN /\ node' = tr.f
                  /\ msgs' = (msgs \ {m}) \cup tr.out
                  /\ UNCHANGED <<instSet, budget>>
     ELSE \* the relay: aggregate and answer the root; vote with own counts
          IF /\ CanDowngradeF(node, r) /\ c2 <= node[r].pclk
             /\ (FlagVeto => ~node[r].rflag)
          THEN /\ node' = [node EXCEPT ![r].st = "Pending",
                                       ![r].voted = rd.rid,
                                       ![r].clk = Max(c2, node[r].pclk),
                                       ![r].bump = TRUE,
                                       ![r].rnd = NoRound]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(rd.par, r, rd.rid, nrdy2,
                                     tS2 + node[r].sent,
                                     tR2 + node[r].recv,
                                     Max(c2, node[r].pclk))}
               /\ UNCHANGED <<instSet, budget>>
          ELSE /\ node' = [node EXCEPT ![r].clk = c2, ![r].rnd = NoRound,
                                       ![r].rflag = IF FlagVeto THEN FALSE
                                                    ELSE @]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(rd.par, r, rd.rid, r, 0, 0, c2)}
               /\ UNCHANGED <<instSet, budget>>

(* Downgrade success.                                                     *)
(* Old mode: force the downgrade whenever the state plausibly matches    *)
(* (the unsolicited-success bug). New mode: apply only with a matching   *)
(* vote receipt. The owner space forwards to the live instance set,      *)
(* which is exactly what the old code did -- receipts make it safe.      *)
RecvSucc(m) ==
  LET d == m.dst
      apply == IF ReceiptChecks
               THEN node[d].voted = m.rid /\ node[d].st = "Pending"
               ELSE node[d].st \in {"Global", "Pending"}
      fwd == IF d = OwnerSpace /\ apply
             THEN {SuccMsg(k, d, m.rid) : k \in instSet \ {m.src, node[d].own}}
             ELSE {}
  IN
  IF node[d].st = "Absent" \/ ~apply
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE /\ node' = [node EXCEPT ![d].st = "Local", ![d].died = TRUE,
                               ![d].voted = 0]
       /\ msgs' = (msgs \ {m}) \cup fwd
       /\ UNCHANGED <<instSet, budget>>

(* Ownership transfer (DowngradeUpdate).                                  *)
(* Old mode: adopt unconditionally and PROMOTE the state to match the    *)
(* sender (the resurrection bug). New mode: adopt only strictly newer    *)
(* versions; never touch the state; run the downgrade check; absorb a    *)
(* parked restart.                                                       *)
RecvUpd(m) ==
  LET d == m.dst IN
  \* the real handler blocks in find_distributed_collectable until the
  \* replica exists (record-before-dispatch guarantees creation is coming,
  \* and an in-flight transfer forbids a commit, so the object is alive)
  /\ node[d].st # "Absent"
  /\ IF OwnershipVersioning /\ m.ver <= node[d].ver
     THEN /\ msgs' = msgs \ {m}
          /\ UNCHANGED <<node, instSet, budget>>
     ELSE LET promote == ~OwnershipVersioning /\ node[d].st = "Local"
           \* Receiving ownership is proof that the round we voted in
           \* failed (the root transfers only on a failed decision), so a
           \* PENDING receiver rolls back to GLOBAL -- the abort rollback,
           \* piggybacked on the transfer with no extra messages.
           rollback == OwnershipVersioning /\ node[d].st = "Pending"
           f0 == [node EXCEPT ![d].own = d, ![d].ver = m.ver,
                              ![d].lastR = Max(node[d].lastR, m.lastR),
                              ![d].clk = Max(node[d].clk, m.clk),
                              ![d].st = IF promote \/ rollback
                                        THEN "Global" ELSE @,
                              ![d].voted = IF rollback THEN 0 ELSE @,
                              ![d].rflag = FALSE]
           tr == TryRound(d, f0, instSet)
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>

(* Downgrade restart. Old mode: dropped at any node that is not the      *)
(* downgrade owner and dropped during an active round (the progress      *)
(* leak). New mode: forwarded with the sender's ownership version;       *)
(* parked when the receiver does not know strictly more, or when a       *)
(* round is active; a parked flag is absorbed by DropRef/RecvUpd/        *)
(* round completion.                                                     *)
RecvRestart(m) ==
  LET d == m.dst
      c2 == Max(node[d].clk, m.clk)
  IN
  IF node[d].st \in {"Absent", "Local"}
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF node[d].own # d
  THEN IF ~RestartForwarding
       THEN /\ msgs' = msgs \ {m}    \* old: silently dropped
            /\ UNCHANGED <<node, instSet, budget>>
       ELSE IF node[d].ver > m.ver
       THEN /\ node' = [node EXCEPT ![d].clk = c2]
            /\ msgs' = (msgs \ {m}) \cup
                         {RestartMsg(node[d].own, m.cand, node[d].ver, c2)}
            /\ UNCHANGED <<instSet, budget>>
       ELSE \* we know no more than the sender: park it
            /\ node' = [node EXCEPT ![d].rflag = TRUE, ![d].clk = c2]
            /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<instSet, budget>>
  ELSE IF node[d].rnd.act
  THEN IF RestartForwarding
       THEN /\ node' = [node EXCEPT ![d].rflag = TRUE, ![d].clk = c2]
            /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<instSet, budget>>
       ELSE /\ msgs' = msgs \ {m}    \* old: dropped mid-round
            /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF ~CanDowngradeF(node, d)
  THEN /\ node' = [node EXCEPT ![d].clk = c2]
       /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<instSet, budget>>
  ELSE IF m.cand # d
  THEN LET v2 == node[d].ver + 1 IN
       /\ node' = [node EXCEPT ![d].own = m.cand, ![d].ver = v2,
                               ![d].clk = c2]
       /\ msgs' = (msgs \ {m}) \cup
                    {UpdMsg(m.cand, d, v2, node[d].lastR, c2)}
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET f0 == [node EXCEPT ![d].clk = c2]
           tr == TryRound(d, f0, instSet)
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>

(* Remote acquire chase. The grant is a packed reference (RefMsg), with  *)
(* the same conditional clock bump as any pack. No round-active guard is *)
(* needed BECAUSE the commit is atomic with the decision and the counts  *)
(* + clocks poison any round the grant crosses -- TLC verifies this.     *)
RecvAcq(m) ==
  LET d == m.dst
      orig == m.cand
  IN
  IF /\ node[d].st = "Global" /\ node[d].own = d
     /\ budget.packs < MaxPacks
  THEN LET c2 == IF node[d].bump THEN node[d].clk + 1 ELSE node[d].clk IN
       /\ node' = [node EXCEPT ![d].sent = @ + 1, ![d].clk = c2,
                               ![d].bump = FALSE]
       /\ budget' = [budget EXCEPT !.packs = @ + 1]
       /\ msgs' = (msgs \ {m}) \cup {RefMsg(orig, d, budget.packs + 1, c2)}
       /\ UNCHANGED instSet
  ELSE IF /\ node[d].st \notin {"Absent"} /\ node[d].own # d
          /\ m.hops > 0
  THEN /\ msgs' = (msgs \ {m}) \cup
                    {AcqMsg(node[d].own, orig, m.hops - 1)}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE /\ msgs' = (msgs \ {m}) \cup {DenyMsg(orig)}
       /\ UNCHANGED <<node, instSet, budget>>

RecvDeny(m) ==
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<node, instSet, budget>>

(***************************************************************************)
(* Next-state relation                                                    *)
(***************************************************************************)
Recv(m) ==
  CASE m.t = "ref"     -> RecvRef(m)
    [] m.t = "spawn"   -> RecvSpawn(m)
    [] m.t = "reg"     -> RecvReg(m)
    [] m.t = "regresp" -> RecvRegResp(m)
    [] m.t = "req"     -> RecvReqBody(m)
    [] m.t = "resp"    -> RecvResp(m)
    [] m.t = "succ"    -> RecvSucc(m)
    [] m.t = "upd"     -> RecvUpd(m)
    [] m.t = "restart" -> RecvRestart(m)
    [] m.t = "acq"     -> RecvAcq(m)
    [] m.t = "deny"    -> RecvDeny(m)
    [] m.t = "unpin"   -> RecvUnpin(m)
    [] OTHER           -> FALSE

ProgressNext ==
  \/ \E m \in msgs : Recv(m)
  \/ \E n \in Nodes : DropRef(n) \/ Collect(n)

CreateNext ==
  \/ \E n \in Nodes, m \in Nodes : Pack(n, m)
  \/ \E n \in Nodes : AcquireLocal(n) \/ AcqStart(n)
  \/ \E m \in Nodes, c \in Nodes : Spawn(m, c)

Next == ProgressNext \/ CreateNext

Spec == Init /\ [][Next]_vars /\ WF_vars(ProgressNext)

(***************************************************************************)
(* Invariants                                                             *)
(***************************************************************************)
CountBound == MaxPacks + MaxAcqs + MaxSpawns + 2

TypeOK ==
  /\ node \in [Nodes ->
       [st: States, refs: 0..CountBound, sent: 0..CountBound,
        recv: 0..CountBound, reg: BOOLEAN, own: Nodes,
        ver: Nat, lastR: Nat, voted: Nat,
        clk: Nat, pclk: Nat, bump: BOOLEAN,
        rflag: BOOLEAN, gen: Nat, died: BOOLEAN, pins: 0..MaxSpawns,
        rnd: [act: BOOLEAN, rid: Nat, ow: Nodes, par: Nodes,
              wait: SUBSET Nodes, nrdy: Nodes, tS: Nat, tR: Nat]]]
  /\ instSet \subseteq (Nodes \ {OwnerSpace})
  /\ budget \in [packs: 0..MaxPacks, acqs: 0..MaxAcqs, spawns: 0..MaxSpawns]

(* A node holding references is never LOCAL or collected.                 *)
SafeRefs ==
  \A n \in Nodes :
    node[n].refs > 0 => node[n].st \in {"Global", "Pending"}

(* A replica that reached LOCAL never comes back within one lifetime.     *)
DiedStaysDead ==
  \A n \in Nodes :
    node[n].died => node[n].st \in {"Local", "Absent"}

(* If the owner-space replica has gone LOCAL/collected, the commit was    *)
(* globally justified: nothing anywhere still holds or carries a          *)
(* reference, and no replica is still (or newly) GLOBAL.                  *)
DeadOwnerClean ==
  node[OwnerSpace].st \in {"Local", "Absent"} =>
    /\ \A n \in Nodes : node[n].refs = 0
    /\ \A n \in Nodes : node[n].st \in {"Pending", "Local", "Absent"}
    /\ ~\E m \in msgs : m.t = "ref"

(* At most one node believes itself the downgrade owner.                  *)
OwnerUnique ==
  Cardinality({n \in Nodes : node[n].own = n /\ node[n].st # "Absent"}) <= 1

(* The design-review claim, machine-checked: while any handle transfer or *)
(* registration is in flight, the object is alive at the owner space --   *)
(* so the registration handler's blocking find can never wait forever.    *)
HandlesCovered ==
  (\E m \in msgs : m.t \in {"spawn", "reg"}) =>
    node[OwnerSpace].st \in {"Global", "Pending"}

(***************************************************************************)
(* Liveness: once (budget-bounded) reference creation stops, everything   *)
(* is eventually collected on every node. The leak bugs violate this.     *)
(***************************************************************************)
EventualCollection == <>[](\A n \in Nodes : node[n].st = "Absent")

=============================================================================
