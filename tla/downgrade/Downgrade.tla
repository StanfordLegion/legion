---------------------------- MODULE Downgrade ----------------------------
(***************************************************************************)
(* Stage-3 model of the Legion DistributedCollectable downgrade protocol: *)
(* TWO reference levels (VALID -> GLOBAL -> LOCAL) over a COLLECTIVE      *)
(* TREE (TreeNodes) plus flat remote instances. The tree is re-rooted at  *)
(* each round's downgrade owner (CollectiveMapping::get_children).        *)
(* TreeNodes = {OwnerSpace} degenerates to the verified Stage-2 flat      *)
(* model. Stage-1 (single level) is archived in                           *)
(* archive-v1/Downgrade-v3-stage1.tla with its completed matrix.          *)
(*                                                                        *)
(* Modeling rules (unchanged):                                            *)
(*   - one TLA+ action == one gc_lock-held handler execution              *)
(*   - network == set of in-flight messages, arbitrary order, no loss     *)
(*   - downgrade checks run ONLY inside handler actions                   *)
(*                                                                        *)
(* Stage-2 additions under test (design questions, NOT settled):          *)
(*   - CatchUp: a node above a round's level self-downgrades on a fresh   *)
(*     request (the real `while (to_check < current) perform_downgrade`)  *)
(*   - Replica state IS the object level (design review): creations are  *)
(*     stamped with the creator's level and counted at the stamped level, *)
(*     so no replica can materialize above the object's committed level   *)
(*   - ReceiptChecks re-adjudication: stale VALID successes now coexist   *)
(*     with GLOBAL rounds                                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  Nodes,               \* address spaces
  OwnerSpace,          \* the object's owner space (in Nodes)
  TreeNodes,           \* the collective mapping (OwnerSpace is a member);
                       \* {OwnerSpace} degenerates to the flat Stage-2 model
  SrcRouting,          \* TRUE = F13 fix: responses return to the REQUESTER.
                       \* FALSE = master: a non-member responds to the
                       \* member nearest by ID (find_nearest) instead
  RegClockBump,        \* F16 fix: a registration processed while the
                       \* owner-space's aggregation is CLOSED bumps the
                       \* owner-space clock past its lastR; the handshake
                       \* response carries the bump to the registrant,
                       \* whose packs propagate it, so any voter that
                       \* received such a pack fails the round's
                       \* causality check (clk <= rid) and votes
                       \* not-ready. Closes the invisible-middleman hole
                       \* that re-rooted trees open in the mid-round
                       \* registration poison (finding F16).
  BoundedLiveness,     \* TRUE = EventualCollection excuses round-budget-
                       \* capped quiescent owners (needed at tree scope,
                       \* where TLC can always burn the round budget on
                       \* doomed attempts). FALSE = the bare property;
                       \* used by the legacy flat liveness configs so
                       \* their recorded verdicts stay under the exact
                       \* property they were established with.
  MaxPacks,            \* bound on packed references (both levels)
  MaxAcqs,             \* bound on acquire attempts (both levels)
  MaxRounds,           \* bound on round epochs (downgrade lamport clock)
  MaxSpawns,           \* bound on by-handle replica spawns
  \* ---- feature toggles: all FALSE ~= the current protocol ----
  RoundTagging,        \* stale requests/responses detected by round id
  ReceiptChecks,       \* success applies only to the exact round voted in
  OwnershipVersioning, \* ownership adopted only from strictly newer versions
  RestartForwarding,   \* restarts forwarded/parked instead of dropped
  RegistrationGate,    \* cannot vote ready before registration completes
  RegHandshake,        \* registration returns (owner,version) + nudge
  CoveredByHandle,     \* by-handle responses counted + pre-registered
  FlagVeto,            \* unpack mid-round parks a flag; every ready vote refuses
  CatchUp,             \* PENDING_GLOBAL applies its downgrade on a G-round
                       \* request (commit proof); old mode: any valid state
  RegRespOwnership,    \* FIDELITY BISECTION (2026-09-01): TRUE = the model as
                       \* verified so far, where the registration response
                       \* carries (owner, version) and the registrant adopts.
                       \* THE IMPLEMENTATION DOES NOT DO THIS: its response
                       \* carries only the F16 clock (gc.cc
                       \* process_registration_response). FALSE = impl-
                       \* faithful clock-only handshake. The F21 stale-update
                       \* leak trace rides the TRUE-only duplicate ownership
                       \* carrier; if FALSE makes UpdsDrain pass, F21 is a
                       \* model artifact, unreachable in the implementation
  StaleUpdateProbe,    \* F21 fix: parking a downgrade update for an absent
                       \* instance sends a liveness probe to the DID's owner
                       \* space, whose replica is provably the last to die
                       \* (DeadOwnerClean). A DEAD verdict (owner-space
                       \* replica Local/Absent) erases the parked entry; the
                       \* impl defers the verdict at a PENDING owner-space
                       \* replica until its round resolves, so the verdict is
                       \* computed at a stable state (modeled by the action's
                       \* enabling condition). ALIVE verdicts are safe to
                       \* ignore: entry parked + object alive implies the
                       \* target's creation is a counted pack in flight, so
                       \* registration consumes the entry. FALSE = master +
                       \* F18 as committed: the entry leaks forever
  AcquireMode          \* F19 acquire contract (ruling 2026-08-31): an acquire
                       \* may fail only if the object actually committed its
                       \* downgrade at that level. Three structures:
                       \* "deny"      = master: an Absent chase target answers
                       \*               deny, and the deny is FINAL (the F19
                       \*               spurious failure).
                       \* "park"      = responder-side park-and-replay: a chase
                       \*               at an Absent target is undeliverable
                       \*               until the replica registers. UNSOUND:
                       \*               finding F20 -- a chase arriving after
                       \*               the target collected parks forever (a
                       \*               hung requester).
                       \* "requester" = requester-side finality: responders
                       \*               answer deny at any dead end, but a deny
                       \*               is ADVISORY; the requester fails only
                       \*               when its OWN replica has committed the
                       \*               level (it holds the commit proof), and
                       \*               otherwise re-chases (budget-bounded) or
                       \*               parks the retry on its own replica's
                       \*               next protocol event. TIMELINESS HOLE:
                       \*               a chase denied at a pre-birth new owner
                       \*               that then holds refs quietly leaves the
                       \*               requester waiting on protocol traffic
                       \*               that never comes while the object lives
                       \* "final"     = the impl as proposed by Mike 2026-09-01:
                       \*               responder-side park-on-entry exactly as
                       \*               in "hybrid", but denies are FINAL at the
                       \*               requester (no local backstop): sound
                       \*               because with the impl-faithful handshake
                       \*               (RegRespOwnership FALSE) every deny is
                       \*               provably genuine -- a dead-end replica
                       \*               or bare-Absent DID implies the level
                       \*               committed
                       \* "hybrid"    = the proposed impl: an Absent target
                       \*               parks the chase IFF an ownership update
                       \*               naming it is pending there (the F18
                       \*               PendingCollectable entry; in the impl
                       \*               the ordered REFERENCE channel makes the
                       \*               update-then-chase order certain), which
                       \*               is exactly the pre-birth case; a bare
                       \*               Absent target is post-death (deny).
                       \*               Denies remain advisory: the requester
                       \*               verifies against its own replica, whose
                       \*               commit proof is guaranteed in every
                       \*               post-death deny case

ASSUME /\ OwnerSpace \in Nodes
       /\ OwnerSpace \in TreeNodes /\ TreeNodes \subseteq Nodes
       /\ Cardinality(TreeNodes) <= 8
       /\ SrcRouting \in BOOLEAN /\ BoundedLiveness \in BOOLEAN
       /\ RegClockBump \in BOOLEAN
       /\ MaxPacks \in Nat /\ MaxAcqs \in Nat
       /\ AcquireMode \in {"deny", "park", "requester", "hybrid", "final"}
       /\ StaleUpdateProbe \in BOOLEAN
       /\ RegRespOwnership \in BOOLEAN
       /\ MaxRounds \in Nat /\ MaxSpawns \in Nat

Max(a, b) == IF a >= b THEN a ELSE b

Symm == Permutations(Nodes \ TreeNodes)

(***************************************************************************)
(* The collective tree: binary-heap shape over a fixed linearization of   *)
(* TreeNodes with OwnerSpace at index 1. The tree is UNDIRECTED; a round  *)
(* rooted at owner ow orients it: the parent of n is the next hop from n  *)
(* toward ow and n's children are its remaining neighbors, matching       *)
(* CollectiveMapping::get_children(origin, local).                        *)
(***************************************************************************)
NT == Cardinality(TreeNodes)
TreeSeq == CHOOSE s \in [1..NT -> TreeNodes] :
             /\ s[1] = OwnerSpace
             /\ \A i, j \in 1..NT : (i # j) => (s[i] # s[j])
Idx(n) == CHOOSE i \in 1..NT : TreeSeq[i] = n
AncIdx(j) == {j, j \div 2, j \div 4, j \div 8} \ {0}
Adj(n) == LET i == Idx(n) IN
          {TreeSeq[k] : k \in ({i \div 2, 2 * i, 2 * i + 1} \cap (1..NT))}
NextHop(n, r) ==   \* first node on the tree path from n toward r (n # r)
  IF Idx(n) \in AncIdx(Idx(r))
  THEN TreeSeq[CHOOSE k \in {2 * Idx(n), 2 * Idx(n) + 1} :
                 k \in AncIdx(Idx(r))]
  ELSE TreeSeq[Idx(n) \div 2]
ChildrenRR(r, n) == IF n = r THEN Adj(n) ELSE Adj(n) \ {NextHop(n, r)}
RootOf(ow) == IF ow \in TreeNodes THEN ow ELSE OwnerSpace
TreeKids(d, ow) == IF d \in TreeNodes THEN ChildrenRR(RootOf(ow), d) ELSE {}
\* master's find_nearest for non-members, abstracted: any fixed member
\* other than the requester (OwnerSpace) reproduces the misroute
FarMember == IF NT = 1 THEN OwnerSpace
             ELSE CHOOSE fm \in TreeNodes : fm # OwnerSpace
RespDst(d, requester) ==
  IF SrcRouting \/ (d \in TreeNodes) THEN requester ELSE FarMember

States == {"Absent", "Valid", "PGlobal", "Global", "PLocal", "Local"}
Lvls   == {"V", "G"}

PendOf(l)  == IF l = "V" THEN "PGlobal" ELSE "PLocal"
LiveOf(l)  == IF l = "V" THEN "Valid"   ELSE "Global"
DownTo(l)  == IF l = "V" THEN "Global"  ELSE "Local"
AtLevel(st, l)    == st \in {LiveOf(l), PendOf(l)}
AboveLevel(st, l) == l = "G" /\ st \in {"Valid", "PGlobal"}
BelowLevel(st, l) == \/ (l = "V" /\ st \in {"Global", "PLocal", "Local"})
                     \/ (l = "G" /\ st = "Local")
\* states at which references of level l may legally be held
HoldsAt(st, l) == IF l = "V" THEN st \in {"Valid", "PGlobal"}
                  ELSE st \in {"Valid", "PGlobal", "Global", "PLocal"}
ZeroL == [x \in Lvls |-> 0]
\* the level a creator stamps on new replicas (design review: replicas
\* start in the known state of the object at the creator)
StampOf(st) == IF st \in {"Valid", "PGlobal"} THEN "V" ELSE "G"

NoRound == [act |-> FALSE, lvl |-> "G", rid |-> 0, ow |-> OwnerSpace,
            par |-> OwnerSpace, wait |-> {}, nrdy |-> OwnerSpace,
            tS |-> 0, tR |-> 0]

VARIABLES node, instSet, msgs, budget
vars == <<node, instSet, msgs, budget>>

(***************************************************************************)
(* Messages: uniform record; lvl tags the reference level where relevant. *)
(***************************************************************************)
MsgBase == [t |-> "none", lvl |-> "G", dst |-> OwnerSpace, src |-> OwnerSpace,
            rid |-> 0, ow |-> OwnerSpace, ver |-> 0, nrdy |-> OwnerSpace,
            tS |-> 0, tR |-> 0, cand |-> OwnerSpace, id |-> 0, gen |-> 0,
            lastR |-> 0, hops |-> 0, clk |-> 0, olv |-> "V"]

Mk(type, over) ==
  [f \in DOMAIN MsgBase |->
     IF f = "t" THEN type
     ELSE IF f \in DOMAIN over THEN over[f] ELSE MsgBase[f]]

ReqMsg(d, s, l, rid, ow, v) == Mk("req", [dst |-> d, src |-> s, lvl |-> l,
                                      rid |-> rid, ow |-> ow, ver |-> v])
RespMsg(d, s, l, rid, nr, ts, tr, c) == Mk("resp", [dst |-> d, src |-> s,
                                      lvl |-> l, rid |-> rid, nrdy |-> nr,
                                      tS |-> ts, tR |-> tr, clk |-> c])
SuccMsg(d, s, l, rid)     == Mk("succ", [dst |-> d, src |-> s, lvl |-> l,
                                      rid |-> rid])
UpdMsg(d, s, ol, v, lr, c) == Mk("upd", [dst |-> d, src |-> s, olv |-> ol,
                                      ver |-> v, lastR |-> lr, clk |-> c])
RestartMsg(d, c, v, ck)   == Mk("restart", [dst |-> d, cand |-> c, ver |-> v,
                                      clk |-> ck])
RefMsg(d, s, l, i, ck, ol) == Mk("ref", [dst |-> d, src |-> s, lvl |-> l,
                                      id |-> i, clk |-> ck, olv |-> ol])
SpawnMsg(d, c, l, ow, v, lr, ck, ol) == Mk("spawn", [dst |-> d, cand |-> c,
                                      lvl |-> l, ow |-> ow, ver |-> v,
                                      lastR |-> lr, clk |-> ck, olv |-> ol])
UnpinMsg(c, m, l)         == Mk("unpin", [dst |-> c, src |-> m, lvl |-> l])
RegMsg(s, g)              == Mk("reg",  [dst |-> OwnerSpace, src |-> s,
                                      gen |-> g])
RegRespMsg(d, g, ow, v, lr, ck, ol) == Mk("regresp", [dst |-> d, gen |-> g,
                                      ow |-> ow, ver |-> v, lastR |-> lr,
                                      clk |-> ck, olv |-> ol])
AcqMsg(d, o, l, h)        == Mk("acq",  [dst |-> d, cand |-> o, lvl |-> l,
                                      hops |-> h])
DenyMsg(d, l)             == Mk("deny", [dst |-> d, lvl |-> l])
FailMsg(d, l)             == Mk("fail", [dst |-> d, lvl |-> l])

(***************************************************************************)
(* Initial state: the owner-space replica exists VALID holding one valid  *)
(* reference (e.g. an IndexTreeNode created tree_valid).                   *)
(***************************************************************************)
InitNode(n) ==
  [st    |-> IF n \in TreeNodes THEN "Valid" ELSE "Absent",
   refs  |-> IF n = OwnerSpace THEN [x \in Lvls |-> IF x = "V" THEN 1 ELSE 0]
             ELSE ZeroL,
   sent  |-> ZeroL, recv |-> ZeroL, pins |-> ZeroL,
   reg   |-> (n \in TreeNodes),
   own   |-> OwnerSpace, ver |-> 0,
   lastR |-> 0, voted |-> 0,
   clk   |-> 0, pclk |-> 0, bump |-> FALSE,
   rflag |-> FALSE, gen |-> 0, died |-> FALSE, vdied |-> FALSE,
   rnd   |-> NoRound]

Init == /\ node = [n \in Nodes |-> InitNode(n)]
        /\ instSet = {}
        /\ msgs = {}
        /\ budget = [packs |-> 0, acqs |-> 0, spawns |-> 0]

(***************************************************************************)
(* Helpers                                                                *)
(***************************************************************************)
\* can_downgrade at level l (ValidDistributedCollectable dispatch)
CanDowngradeL(f, n, l) ==
  /\ f[n].refs[l] = 0
  /\ AtLevel(f[n].st, l)
  /\ (RegistrationGate => f[n].reg)


\* F10: under stamp-counting EVERY counted pack is an event at the
\* stamped level, so the clock bump always applies when the flag is set
\* (master's level-conditional bump in pack_global_ref is obsolete)

\* check_for_downgrade at a node that believes it is the downgrade owner.
\* An owner alone chains levels inline (perform_downgrade -> can_delete).
TryRound(n, f, is) ==
  IF /\ f[n].own = n
     /\ f[n].st \in {"Valid", "Global"}
     /\ f[n].refs[StampOf(f[n].st)] = 0
     /\ (RegistrationGate => f[n].reg)
     /\ ~f[n].rnd.act
     /\ Max(f[n].clk, f[n].lastR) < MaxRounds
  THEN LET l     == StampOf(f[n].st)  \* replica state = object level
           parts == IF n \in TreeNodes
                    THEN ChildrenRR(n, n)
                           \cup (IF n = OwnerSpace THEN is ELSE {})
                    ELSE {OwnerSpace}
           rid   == Max(f[n].clk, f[n].lastR) + 1
       IN IF parts = {}
          THEN IF f[n].sent[l] = f[n].recv[l]
               THEN IF l = "G"
                    THEN [f |-> [f EXCEPT ![n].st = "Local",
                                          ![n].died = TRUE], out |-> {}]
                    ELSE \* VALID level commits; chain the GLOBAL level
                         IF /\ f[n].refs["G"] = 0
                            /\ f[n].sent["G"] = f[n].recv["G"]
                         THEN [f |-> [f EXCEPT ![n].st = "Local",
                                               ![n].vdied = TRUE,
                                               ![n].died = TRUE], out |-> {}]
                         ELSE [f |-> [f EXCEPT
                                        ![n].st = "Global",
                                        ![n].vdied = TRUE], out |-> {}]
               ELSE [f |-> f, out |-> {}]
          ELSE [f |-> [f EXCEPT
                         ![n].lastR = rid,
                         ![n].pclk = rid,
                         ![n].bump = FALSE,
                         ![n].rnd = [act |-> TRUE, lvl |-> l, rid |-> rid,
                                     ow |-> n, par |-> n, wait |-> parts,
                                     nrdy |-> n, tS |-> 0, tR |-> 0]],
                out |-> {ReqMsg(p, n, l, rid, n, f[n].ver) : p \in parts}]
  ELSE [f |-> f, out |-> {}]

(***************************************************************************)
(* Spontaneous local actions (budget-bounded)                             *)
(***************************************************************************)

\* pack a reference of level l while holding one at that level
Pack(n, m, l) ==
  /\ n # m
  /\ budget.packs < MaxPacks
  /\ IF l = "V" THEN node[n].st = "Valid"
     ELSE node[n].st \in {"Valid", "Global"}
  /\ node[n].refs[l] > 0
  /\ node[n].reg
  /\ LET c2 == IF node[n].bump THEN node[n].clk + 1 ELSE node[n].clk IN
     /\ budget' = [budget EXCEPT !.packs = @ + 1]
     /\ msgs' = msgs \cup {RefMsg(m, n, l, budget.packs + 1, c2,
                                   StampOf(node[n].st))}
     \* a creation stamped VALID is counted at the valid level too, so a
     \* VALID commit is arithmetically blocked while it is in flight
     /\ LET stmp == StampOf(node[n].st)
            s2 == [x \in Lvls |->
                     node[n].sent[x] + (IF x = l THEN 1 ELSE 0)
                     + (IF x = stmp /\ stmp # l THEN 1 ELSE 0)]
        IN
        IF n = OwnerSpace
        THEN /\ instSet' = instSet \cup ({m} \ TreeNodes)
             /\ node' = [node EXCEPT ![n].sent = s2,
                                     ![n].clk = c2, ![n].bump = FALSE,
                                     ![n].rnd.nrdy =
                                       IF /\ node[n].rnd.act
                                          /\ m \notin instSet
                                          /\ m \notin TreeNodes
                                       THEN m ELSE @]
        ELSE /\ instSet' = instSet
             /\ node' = [node EXCEPT ![n].sent = s2,
                                     ![n].clk = c2, ![n].bump = FALSE]

\* local mint: global refs mint at any global-or-above live state (matches
\* acquire_global's GLOBAL/VALID/PENDING_GLOBAL cases); valid refs mint at
\* VALID (fidelity question Q-VAL-MINT: confirm acquire_valid's local path)
AcquireLocal(n, l) ==
  /\ budget.acqs < MaxAcqs
  /\ IF l = "G" THEN node[n].st \in {"Valid", "PGlobal", "Global"}
     ELSE node[n].st = "Valid"
  /\ node' = [node EXCEPT ![n].refs[l] = @ + 1]
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ UNCHANGED <<instSet, msgs>>

\* acquire from a pending state: must go through the downgrade owner
AcqStart(n, l) ==
  /\ budget.acqs < MaxAcqs
  /\ node[n].st = PendOf(l)
  /\ node[n].own # n
  /\ msgs' = msgs \cup {AcqMsg(node[n].own, n, l, Cardinality(Nodes) + 2)}
  /\ budget' = [budget EXCEPT !.acqs = @ + 1]
  /\ UNCHANGED <<node, instSet>>

\* By-handle send from the owner space at level l (send_node's `valid`
\* flag). Covered: counted + pre-registered (Option A), the covering
\* reference pinned on node c at the SAME level. A pending sender uses
\* the acquire fallback (modeled by requiring the live state).
Spawn(m, c) ==
  /\ m \notin TreeNodes
  /\ node[m].st = "Absent"
  /\ m \notin instSet
  /\ budget.spawns < MaxSpawns
  /\ instSet' = instSet \cup {m}
  /\ budget' = [budget EXCEPT !.spawns = @ + 1]
  /\ LET stmp == StampOf(node[OwnerSpace].st) IN
     IF CoveredByHandle
     THEN /\ node[OwnerSpace].st \in {"Valid", "Global"}
          /\ node[c].refs[stmp] > node[c].pins[stmp]
          /\ LET c2 == IF node[OwnerSpace].bump
                       THEN node[OwnerSpace].clk + 1
                       ELSE node[OwnerSpace].clk
             IN /\ node' = [node EXCEPT
                             ![c].pins[stmp] = @ + 1,
                             ![OwnerSpace].sent[stmp] = @ + 1,
                             ![OwnerSpace].clk = c2,
                             ![OwnerSpace].bump = FALSE,
                             ![OwnerSpace].rnd.nrdy =
                               IF node[OwnerSpace].rnd.act THEN m ELSE @]
                /\ msgs' = msgs \cup
                     {SpawnMsg(m, c, stmp, node[OwnerSpace].own,
                               node[OwnerSpace].ver,
                               node[OwnerSpace].lastR, c2, stmp)}
     ELSE /\ node[OwnerSpace].st \in {"Valid", "PGlobal", "Global", "PLocal"}
          /\ node' = [node EXCEPT ![OwnerSpace].rnd.nrdy =
                        IF node[OwnerSpace].rnd.act THEN m ELSE @]
          /\ msgs' = msgs \cup {SpawnMsg(m, c, stmp, OwnerSpace, 0, 0, 0,
                                          stmp)}

\* remove a reference of level l -> can_delete
DropRef(n, l) ==
  /\ node[n].refs[l] > node[n].pins[l]
  /\ budget' = budget
  /\ IF node[n].refs[l] > 1
     THEN /\ node' = [node EXCEPT ![n].refs[l] = @ - 1]
          /\ UNCHANGED <<instSet, msgs>>
     ELSE LET keepFlag == RestartForwarding /\ node[n].rnd.act
              f0 == [node EXCEPT ![n].refs[l] = 0,
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

Collect(n) ==
  /\ node[n].st = "Local"
  /\ ~node[n].rnd.act
  /\ node' = [node EXCEPT ![n].st = "Absent"]
  /\ UNCHANGED <<instSet, msgs, budget>>

(***************************************************************************)
(* Message handlers                                                       *)
(***************************************************************************)

\* A packed reference of level m.lvl arrives (also acquire grants).
RecvRef(m) ==
  LET d == m.dst
      l == m.lvl
  IN
  IF node[d].st = "Absent"
  THEN LET ng == node[d].gen + 1 IN
       /\ node' = [node EXCEPT ![d].st = LiveOf(m.olv),
                               ![d].refs = [x \in Lvls |->
                                              IF x = l THEN 1 ELSE 0],
                               ![d].recv = [x \in Lvls |->
                                              IF x = l \/ x = m.olv
                                              THEN 1 ELSE 0],
                               ![d].sent = ZeroL, ![d].pins = ZeroL,
                               ![d].reg = (d \in TreeNodes),
                               ![d].gen = ng, ![d].died = FALSE,
                               ![d].vdied = (m.olv = "G"),
                               ![d].own = OwnerSpace, ![d].ver = 0,
                               ![d].voted = 0, ![d].clk = m.clk,
                               ![d].pclk = 0, ![d].bump = FALSE]
       /\ msgs' = (msgs \ {m}) \cup
                    (IF d \in TreeNodes THEN {} ELSE {RegMsg(d, ng)})
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET wasPend == node[d].st = PendOf(l)
           quiet   == node[d].refs[l] = 0
           newSt   == IF wasPend THEN LiveOf(l) ELSE node[d].st
           c2      == Max(node[d].clk, m.clk)
           notify  == node[d].own # d /\ (wasPend \/ quiet)
                      /\ ~node[d].rnd.act
           park    == \/ (node[d].own # d /\ ~(wasPend \/ quiet)
                          /\ ~node[d].rnd.act)
                      \/ (FlagVeto /\ node[d].rnd.act)
       IN /\ node' = [node EXCEPT ![d].st = newSt, ![d].refs[l] = @ + 1,
                                  ![d].recv = [x \in Lvls |->
                                     node[d].recv[x]
                                     + (IF x = l THEN 1 ELSE 0)
                                     + (IF x = m.olv /\ m.olv # l
                                        THEN 1 ELSE 0)],
                                  ![d].clk = c2,
                                  ![d].voted = IF wasPend THEN 0 ELSE @,
                                  ![d].rflag = @ \/ park]
          /\ msgs' = (msgs \ {m}) \cup
                       (IF notify
                        THEN {RestartMsg(node[d].own, d, node[d].ver, c2)}
                        ELSE {})
          /\ UNCHANGED <<instSet, budget>>

\* By-handle response at level m.lvl: counted + pre-registered (Option A).
RecvSpawn(m) ==
  LET d == m.dst
      l == m.lvl
      unpin == IF CoveredByHandle THEN {UnpinMsg(m.cand, d, l)} ELSE {}
  IN
  IF node[d].st = "Absent"
  THEN LET ng == node[d].gen + 1 IN
       IF CoveredByHandle
       THEN /\ node' = [node EXCEPT ![d].st = LiveOf(l), ![d].reg = TRUE,
                                    ![d].gen = ng, ![d].died = FALSE,
                                    ![d].vdied = (m.olv = "G"),
                                    ![d].refs = ZeroL, ![d].sent = ZeroL,
                                    ![d].pins = ZeroL,
                                    ![d].recv = [x \in Lvls |->
                                                   IF x = l THEN 1 ELSE 0],
                                    ![d].own = m.ow, ![d].ver = m.ver,
                                    ![d].lastR = m.lastR,
                                    ![d].voted = 0, ![d].clk = m.clk,
                                    ![d].pclk = 0, ![d].bump = FALSE]
            /\ msgs' = (msgs \ {m}) \cup unpin \cup
                 (IF m.ow # d
                  THEN {RestartMsg(m.ow, d, m.ver, m.clk)} ELSE {})
            /\ UNCHANGED <<instSet, budget>>
       ELSE /\ node' = [node EXCEPT ![d].st = LiveOf(l), ![d].reg = FALSE,
                                    ![d].gen = ng, ![d].died = FALSE,
                                    ![d].vdied = (m.olv = "G"),
                                    ![d].refs = ZeroL, ![d].sent = ZeroL,
                                    ![d].recv = ZeroL, ![d].pins = ZeroL,
                                    ![d].own = OwnerSpace, ![d].ver = 0,
                                    ![d].voted = 0, ![d].clk = 0,
                                    ![d].pclk = 0, ![d].bump = FALSE]
            /\ msgs' = (msgs \ {m}) \cup {RegMsg(d, ng)} \cup unpin
            /\ UNCHANGED <<instSet, budget>>
  ELSE IF CoveredByHandle
       THEN LET quiet == node[d].refs[l] = 0
                notify == node[d].own # d /\ quiet /\ ~node[d].rnd.act
                park == \/ (node[d].own # d /\ ~quiet /\ ~node[d].rnd.act)
                        \/ (FlagVeto /\ node[d].rnd.act)
            IN /\ node' = [node EXCEPT ![d].recv[l] = @ + 1,
                                       ![d].clk = Max(@, m.clk),
                                       ![d].rflag = @ \/ park]
               /\ msgs' = (msgs \ {m}) \cup unpin \cup
                    (IF notify
                     THEN {RestartMsg(node[d].own, d, node[d].ver,
                                      Max(node[d].clk, m.clk))} ELSE {})
               /\ UNCHANGED <<instSet, budget>>
       ELSE /\ msgs' = (msgs \ {m}) \cup unpin
            /\ UNCHANGED <<node, instSet, budget>>

RecvUnpin(m) ==
  /\ node' = [node EXCEPT ![m.dst].pins[m.lvl] =
                IF @ > 0 THEN @ - 1 ELSE @]
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<instSet, budget>>

\* Registration (ref-created replicas only under Option A).
RecvReg(m) ==
  LET s == m.src
      O == OwnerSpace
  IN
  IF node[O].st \in {"Valid", "PGlobal", "Global", "PLocal"}
  THEN LET hairy == /\ instSet = {} /\ node[O].own = O
                    /\ TreeNodes = {OwnerSpace}
                    /\ \E l \in Lvls : node[O].sent[l] # node[O].recv[l]
                    /\ ~node[O].rnd.act
           newVer == IF hairy THEN node[O].ver + 1 ELSE node[O].ver
           newOwn == IF hairy THEN s ELSE node[O].own
           poison == node[O].rnd.act /\ node[O].rnd.wait # {}
           \* F16 fix: if our aggregation is closed the poison cannot
           \* protect an in-flight round rooted elsewhere in the tree;
           \* bump our clock past every round we have seen so the
           \* registrant's future packs poison any such round's votes
           bumpC == IF RegClockBump /\ ~poison
                    THEN Max(node[O].clk, node[O].lastR) + 1
                    ELSE node[O].clk
           is2 == instSet \cup {s}
           f0 == [node EXCEPT
                    ![O].own  = newOwn,
                    ![O].ver  = newVer,
                    ![O].clk  = bumpC,
                    ![O].rnd.nrdy = IF poison THEN s ELSE @,
                    ![s].reg  = IF ~RegHandshake /\ m.gen = node[s].gen
                                THEN TRUE ELSE @]
           nudgeSelf == RegHandshake /\ ~hairy /\ ~poison /\ newOwn = O
           tr == IF nudgeSelf THEN TryRound(O, f0, is2)
                 ELSE [f |-> f0, out |-> {}]
           nudgeOut ==
             IF RegHandshake /\ ~hairy /\ ~poison /\ newOwn # O
             THEN {RestartMsg(newOwn, newOwn, node[O].ver, bumpC)}
             ELSE {}
           handshakeOut ==
             IF RegHandshake
             THEN {RegRespMsg(s, m.gen, newOwn, newVer, node[O].lastR,
                              bumpC, StampOf(node[O].st))}
             ELSE {}
           hairyOut ==
             IF hairy
             THEN {UpdMsg(s, O, StampOf(node[O].st), newVer,
                          node[O].lastR, bumpC)}
             ELSE {}
       IN /\ instSet' = is2
          /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup handshakeOut \cup hairyOut
                       \cup nudgeOut \cup tr.out
          /\ UNCHANGED budget
  ELSE FALSE   \* blocking-find tripwire; HandlesCovered verifies unreachable

RecvRegResp(m) ==
  LET s == m.dst IN
  IF m.gen = node[s].gen /\ node[s].st # "Absent" /\ node[s].st # "Local"
  THEN LET adopt == /\ RegRespOwnership
                    /\ IF OwnershipVersioning
                       THEN m.ver > node[s].ver
                       ELSE TRUE
           f0 == [node EXCEPT
                    ![s].reg   = TRUE,
                    ![s].own   = IF adopt THEN m.ow ELSE @,
                    ![s].ver   = IF adopt THEN m.ver ELSE @,
                    ![s].clk   = Max(node[s].clk, m.clk),
                    ![s].lastR = Max(node[s].lastR, m.lastR)]
           tr == TryRound(s, f0, instSet)
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>
  ELSE /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>

(***************************************************************************)
(* Downgrade request at level m.lvl. Vote semantics by relation of the    *)
(* node's state to the round's level:                                     *)
(*   AT level    -> the Stage-1 rules (counts, clock, flag veto)          *)
(*   ABOVE level -> CatchUp: self-downgrade first (a fresh lower-level    *)
(*     round proves the higher level committed); else not-ready           *)
(*   BELOW level -> impossible in the new design (stamped creations);    *)
(*     answered not-ready defensively                                      *)
(***************************************************************************)
RecvReqBody(m) ==
  LET d == m.dst
      l == m.lvl
      stale == RoundTagging /\ m.rid <= node[d].lastR
  IN
  /\ node[d].st # "Absent"
  /\ ~node[d].rnd.act
  /\ IF node[d].st = "Local" \/ stale
     THEN /\ msgs' = (msgs \ {m}) \cup
                       {RespMsg(RespDst(d, m.src), d, l, m.rid, d, 0, 0, node[d].clk)}
          /\ UNCHANGED <<node, instSet, budget>>
     ELSE LET adopt == IF OwnershipVersioning
                       THEN m.ver > node[d].ver
                       ELSE TRUE
              own2 == IF adopt THEN m.ow ELSE node[d].own
              ver2 == IF adopt THEN m.ver ELSE node[d].ver
              lr2  == Max(node[d].lastR, m.rid)
              \* catch-up: a G-round request is a commit proof for a
              \* PGlobal voter (new mode); old mode: any valid-level state
              \* (master's while-loop -> finding #12). A fully-VALID node
              \* seeing a G-round is impossible in the new design
              \* (StrictLevelOrder); it answers not-ready defensively.
              cu     == /\ AboveLevel(node[d].st, l) /\ CatchUp
                        /\ (ReceiptChecks => node[d].st = "PGlobal")
              stEff  == IF cu THEN "Global" ELSE node[d].st
              can    == /\ stEff \in {LiveOf(l), PendOf(l)}
                        /\ node[d].refs[l] = 0
                        /\ (RegistrationGate => node[d].reg)
                        /\ node[d].clk <= m.rid
                        /\ (FlagVeto => ~node[d].rflag)
          IN
          IF ~can
          THEN /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2,
                                       ![d].st = stEff,
                                       ![d].vdied = @ \/ cu,
                                       ![d].voted = IF cu THEN 0 ELSE @,
                                       ![d].rflag = IF FlagVeto THEN FALSE
                                                    ELSE @]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(RespDst(d, m.src), d, l, m.rid, d, 0, 0,
                                     node[d].clk)}
               /\ UNCHANGED <<instSet, budget>>
          ELSE IF (TreeKids(d, m.ow)
                   \cup (IF d = OwnerSpace THEN instSet ELSE {}))
                  \ {m.ow, m.src} # {}
          THEN \* relay: fan out (relay's own vote happens at aggregation)
               LET kids == (TreeKids(d, m.ow)
                            \cup (IF d = OwnerSpace THEN instSet ELSE {}))
                           \ {m.ow, m.src} IN
               /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2,
                                       ![d].st = stEff,
                                       ![d].vdied = @ \/ cu,
                                       ![d].voted = IF cu THEN 0 ELSE @,
                                       ![d].pclk = m.rid,
                                       ![d].bump = FALSE,
                                       ![d].rnd = [act |-> TRUE, lvl |-> l,
                                                   rid |-> m.rid,
                                                   ow |-> m.ow,
                                                   par |-> m.src,
                                                   wait |-> kids,
                                                   nrdy |-> m.ow,
                                                   tS |-> 0, tR |-> 0]]
               /\ msgs' = (msgs \ {m}) \cup
                            {ReqMsg(k, d, l, m.rid, m.ow, ver2) : k \in kids}
               /\ UNCHANGED <<instSet, budget>>
          ELSE \* vote ready now (leaf, or relay with no children)
               /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2,
                                       ![d].st = PendOf(l),
                                       ![d].vdied = @ \/ cu,
                                       ![d].voted = m.rid,
                                       ![d].clk = m.rid,
                                       ![d].pclk = m.rid,
                                       ![d].bump = TRUE]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(RespDst(d, m.src), d, l, m.rid, m.ow,
                                     node[d].sent[l], node[d].recv[l],
                                     m.rid)}
               /\ UNCHANGED <<instSet, budget>>

(***************************************************************************)
(* Downgrade response: decision + the root's own downgrade are ATOMIC.    *)
(* A committing V-round root chains the G level inline when it is the     *)
(* sole holder (perform_downgrade -> can_delete in the real code).        *)
(***************************************************************************)
RecvResp(m) ==
  LET r == m.dst
      rd == node[r].rnd
  IN
  /\ rd.act
  /\ m.src \in rd.wait
  /\ (RoundTagging => (m.rid = rd.rid /\ m.lvl = rd.lvl))
  /\ LET l      == rd.lvl
         clean  == rd.nrdy = rd.ow
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
     THEN LET causal == c2 <= node[r].pclk
              ready  == /\ nrdy2 = r
                        /\ CanDowngradeL(node, r, l)
                        /\ tS2 + node[r].sent[l] = tR2 + node[r].recv[l]
                        /\ causal
                        /\ (FlagVeto => ~node[r].rflag)
          IN
          IF ready
          THEN LET succTo == (IF r \in TreeNodes
                              THEN ChildrenRR(r, r) ELSE {OwnerSpace})
                             \cup (IF r = OwnerSpace THEN instSet ELSE {})
                   f0 == [node EXCEPT ![r].st = DownTo(l),
                                      ![r].vdied = @ \/ (l = "V"),
                                      ![r].died = @ \/ (l = "G"),
                                      ![r].voted = 0,
                                      ![r].clk = c2,
                                      ![r].rnd = NoRound]
                   \* chain: a V-commit leaves the root GLOBAL; if its gc
                   \* refs are already zero, can_delete runs immediately
                   tr == IF l = "V" THEN TryRound(r, f0, instSet)
                         ELSE [f |-> f0, out |-> {}]
               IN /\ node' = tr.f
                  /\ msgs' = (msgs \ {m}) \cup tr.out \cup
                               {SuccMsg(k, r, l, rd.rid) : k \in succTo}
                  /\ UNCHANGED <<instSet, budget>>
          ELSE IF nrdy2 # r
          THEN LET v2 == node[r].ver + 1
                   restartOut == IF node[r].rflag
                                 THEN {RestartMsg(nrdy2, r, v2, c2)} ELSE {}
               IN /\ node' = [node EXCEPT ![r].own = nrdy2, ![r].ver = v2,
                                          ![r].rflag = FALSE, ![r].clk = c2,
                                          ![r].bump = TRUE,
                                          ![r].rnd = NoRound]
                  /\ msgs' = (msgs \ {m}) \cup
                               {UpdMsg(nrdy2, r, StampOf(node[r].st), v2,
                                       node[r].lastR, c2)}
                               \cup restartOut
                  /\ UNCHANGED <<instSet, budget>>
          ELSE LET f0 == [node EXCEPT ![r].rnd = NoRound, ![r].clk = c2,
                                      ![r].bump = TRUE, ![r].rflag = FALSE]
                   retry == ~causal \/ node[r].rflag
                   tr == IF retry THEN TryRound(r, f0, instSet)
                         ELSE [f |-> f0, out |-> {}]
               IN /\ node' = tr.f
                  /\ msgs' = (msgs \ {m}) \cup tr.out
                  /\ UNCHANGED <<instSet, budget>>
     ELSE \* the relay aggregates and votes with its own counters
          LET rdyAt  == /\ node[r].st \in {LiveOf(l), PendOf(l)}
                        /\ node[r].refs[l] = 0
                        /\ (RegistrationGate => node[r].reg)
                        /\ c2 <= node[r].pclk
                        /\ (FlagVeto => ~node[r].rflag)
          IN
          IF rdyAt
          THEN /\ node' = [node EXCEPT ![r].st = PendOf(l),
                                       ![r].voted = rd.rid,
                                       ![r].clk = Max(c2, node[r].pclk),
                                       ![r].bump = TRUE,
                                       ![r].rnd = NoRound]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(rd.par, r, l, rd.rid, nrdy2,
                                     tS2 + node[r].sent[l],
                                     tR2 + node[r].recv[l],
                                     Max(c2, node[r].pclk))}
               /\ UNCHANGED <<instSet, budget>>
          ELSE /\ node' = [node EXCEPT ![r].clk = c2, ![r].rnd = NoRound,
                                       ![r].rflag = IF FlagVeto THEN FALSE
                                                    ELSE @]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(rd.par, r, l, rd.rid, r, 0, 0, c2)}
               /\ UNCHANGED <<instSet, budget>>

\* Downgrade success at level m.lvl.
RecvSucc(m) ==
  LET d == m.dst
      l == m.lvl
      apply == IF ReceiptChecks
               THEN node[d].voted = m.rid /\ node[d].st = PendOf(l)
               ELSE node[d].st \in {LiveOf(l), PendOf(l)}
      fwd == IF apply
             THEN {SuccMsg(k, d, l, m.rid) :
                     k \in ((IF d \in TreeNodes
                             THEN ChildrenRR(RootOf(node[d].own), d)
                             ELSE {})
                            \cup (IF d = OwnerSpace THEN instSet ELSE {}))
                           \ {m.src, node[d].own}}
             ELSE {}
  IN
  IF node[d].st = "Absent"
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF ~apply
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE /\ node' = [node EXCEPT ![d].st = DownTo(l),
                               ![d].vdied = @ \/ (l = "V"),
                               ![d].died = @ \/ (l = "G"),
                               ![d].voted = 0]
       /\ msgs' = (msgs \ {m}) \cup fwd
       /\ UNCHANGED <<instSet, budget>>

\* Ownership transfer. New mode: version-gated, per-level PENDING rollback,
\* never otherwise touches state. Old mode: unconditional adoption plus
\* the state promotion/catch-up of the current code (resurrection bugs).
RecvUpd(m) ==
  LET d == m.dst IN
  /\ node[d].st # "Absent"   \* blocking find (see Stage-1 F6b note)
  /\ IF OwnershipVersioning /\ m.ver <= node[d].ver
     THEN /\ msgs' = msgs \ {m}
          /\ UNCHANGED <<node, instSet, budget>>
     ELSE LET oldPromote ==
                ~OwnershipVersioning /\
                  \/ (m.olv = "V" /\ node[d].st \in {"Global", "PLocal",
                                                     "Local"})
                  \/ (m.olv = "G" /\ node[d].st = "Local")
              \* rollback only if the failed round was at OUR pending
              \* level; an update stamped G at a PGlobal node is a
              \* V-commit proof (apply, do not roll back)
              rollV == OwnershipVersioning /\ node[d].st = "PGlobal"
                       /\ m.olv = "V"
              commitV == OwnershipVersioning /\ node[d].st = "PGlobal"
                         /\ m.olv = "G"
              rollG == OwnershipVersioning /\ node[d].st = "PLocal"
              newSt == IF oldPromote THEN LiveOf(m.olv)
                       ELSE IF rollV THEN "Valid"
                       ELSE IF commitV \/ rollG THEN "Global"
                       ELSE node[d].st
              f0 == [node EXCEPT ![d].own = d, ![d].ver = m.ver,
                                 ![d].lastR = Max(node[d].lastR, m.lastR),
                                 ![d].clk = Max(node[d].clk, m.clk),
                                 ![d].st = newSt,
                                 ![d].vdied = @ \/ commitV,
                                 ![d].voted = IF rollV \/ commitV \/ rollG
                                              THEN 0 ELSE @,
                                 ![d].rflag = FALSE]
              tr == TryRound(d, f0, instSet)
          IN /\ node' = tr.f
             /\ msgs' = (msgs \ {m}) \cup tr.out
             /\ UNCHANGED <<instSet, budget>>

RecvRestart(m) ==
  LET d == m.dst
      c2 == Max(node[d].clk, m.clk)
  IN
  IF node[d].st \in {"Absent", "Local"}
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF node[d].own # d
  THEN IF ~RestartForwarding
       THEN /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<node, instSet, budget>>
       ELSE IF node[d].ver > m.ver
       THEN /\ node' = [node EXCEPT ![d].clk = c2]
            /\ msgs' = (msgs \ {m}) \cup
                         {RestartMsg(node[d].own, m.cand, node[d].ver, c2)}
            /\ UNCHANGED <<instSet, budget>>
       ELSE /\ node' = [node EXCEPT ![d].rflag = TRUE, ![d].clk = c2]
            /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<instSet, budget>>
  ELSE IF node[d].rnd.act
  THEN IF RestartForwarding
       THEN /\ node' = [node EXCEPT ![d].rflag = TRUE, ![d].clk = c2]
            /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<instSet, budget>>
       ELSE /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF ~(node[d].st \in {"Valid", "Global"} /\
            node[d].refs[StampOf(node[d].st)] = 0 /\
            (RegistrationGate => node[d].reg))
  THEN /\ node' = [node EXCEPT ![d].clk = c2]
       /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<instSet, budget>>
  ELSE IF m.cand # d
  THEN LET v2 == node[d].ver + 1 IN
       /\ node' = [node EXCEPT ![d].own = m.cand, ![d].ver = v2,
                               ![d].clk = c2]
       /\ msgs' = (msgs \ {m}) \cup
                    {UpdMsg(m.cand, d, StampOf(node[d].st), v2,
                            node[d].lastR, c2)}
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET f0 == [node EXCEPT ![d].clk = c2]
           tr == TryRound(d, f0, instSet)
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>

\* Remote acquire chase at level m.lvl (grant = counted RefMsg).
\* Common to all modes: a live self-believing owner grants; a replica
\* genuinely below the requested level denies (the object committed
\* there); a live non-owner forwards. The modes differ at the dead ends:
\* an Absent target answers deny ("deny"/"requester") or parks the
\* message until the replica registers ("park"; F20: forever if it never
\* does). Model-bound truncations (pack budget, hop budget -- the impl
\* chase is unbounded): in "requester" mode they answer an ADVISORY deny
\* (harmless -- the requester retries), so no chase is silently lost; in
\* the final-deny modes they retire the message with NO answer, since a
\* final deny minted by a bound artifact would poison the contract check.
RecvAcq(m) ==
  LET d == m.dst
      l == m.lvl
      orig == m.cand
      grantable == IF l = "V" THEN node[d].st = "Valid"
                   ELSE node[d].st \in {"Valid", "PGlobal", "Global"}
      deny == /\ msgs' = (msgs \ {m}) \cup {DenyMsg(orig, l)}
              /\ UNCHANGED <<node, instSet, budget>>
      retire == /\ msgs' = msgs \ {m}
                /\ UNCHANGED <<node, instSet, budget>>
  IN
  IF node[d].st = "Absent"
  THEN IF \/ AcquireMode = "park"
          \/ (AcquireMode \in {"hybrid", "final"}
              /\ \E m2 \in msgs : m2.t = "upd" /\ m2.dst = d)
       THEN FALSE   \* parked: replays when the replica registers
       ELSE deny
  ELSE IF grantable /\ node[d].own = d
  THEN IF budget.packs < MaxPacks
       THEN LET c2 == IF node[d].bump THEN node[d].clk + 1
                      ELSE node[d].clk IN
            /\ node' = [node EXCEPT ![d].sent[l] = @ + 1, ![d].clk = c2,
                                    ![d].bump = FALSE]
            /\ budget' = [budget EXCEPT !.packs = @ + 1]
            /\ msgs' = (msgs \ {m}) \cup
                         {RefMsg(orig, d, l, budget.packs + 1, c2,
                                 StampOf(node[d].st))}
            /\ UNCHANGED instSet
       ELSE IF AcquireMode \in {"requester", "hybrid"} THEN deny
            ELSE retire
  ELSE IF BelowLevel(node[d].st, l)
  THEN deny   \* this replica committed the level
  ELSE IF node[d].own # d
  THEN IF m.hops > 0
       THEN /\ msgs' = (msgs \ {m}) \cup
                         {AcqMsg(node[d].own, orig, l, m.hops - 1)}
            /\ UNCHANGED <<node, instSet, budget>>
       ELSE IF AcquireMode \in {"requester", "hybrid"} THEN deny
            ELSE retire
  ELSE \* self-owner still pending at the level
       IF AcquireMode \in {"requester", "hybrid"} THEN deny ELSE FALSE

\* Deny handling. In the final-deny modes ("deny", "park") the requester
\* just consumes it -- the deny IS the answer, and AcquireContract holds
\* it to the commit-only standard. In "requester" mode the deny is
\* advisory and the requester's OWN replica is the authority: a replica
\* at-or-below commit for the level carries the proof (final failure,
\* recorded as a fail message for the invariant); a locally-live level
\* means the local fast path wins (retire); still-pending re-chases with
\* the refreshed owner belief while budget lasts, and then parks the
\* retry on the replica's next protocol event (the held deny message
\* becomes deliverable again when the replica's state resolves).
RecvDeny(m) ==
  IF AcquireMode \notin {"requester", "hybrid"}
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE LET r == m.dst
           l == m.lvl
           locsat == IF l = "V" THEN node[r].st = "Valid"
                     ELSE node[r].st \in {"Valid", "PGlobal", "Global"}
       IN
       IF node[r].st = "Absent" \/ BelowLevel(node[r].st, l)
       THEN /\ msgs' = (msgs \ {m}) \cup {FailMsg(r, l)}
            /\ UNCHANGED <<node, instSet, budget>>
       ELSE IF locsat
       THEN /\ msgs' = msgs \ {m}
            /\ UNCHANGED <<node, instSet, budget>>
       ELSE IF budget.acqs < MaxAcqs
       THEN /\ msgs' = (msgs \ {m}) \cup
                         {AcqMsg(node[r].own, r, l, Cardinality(Nodes) + 2)}
            /\ budget' = [budget EXCEPT !.acqs = @ + 1]
            /\ UNCHANGED <<node, instSet>>
       ELSE FALSE   \* retry parked on the replica's next protocol event

\* The requester's final acquire failure (requester mode only); exists
\* on the wire solely so AcquireContract can audit its justification.
RecvFail(m) ==
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<node, instSet, budget>>

\* Hybrid mode: the requester abandons an outstanding chase once its own
\* replica carries the commit proof (in the impl: the local protocol
\* event triggers the acquire block's ready exactly once; a parked chase
\* message may leak at a dead DID -- finding F21 -- but the REQUESTER
\* resolves, and a racing late grant is returned by the handler).
AbandonAcq(m) ==
  /\ AcquireMode = "hybrid"
  /\ m.t = "acq"
  /\ \/ node[m.cand].st = "Absent"
     \/ BelowLevel(node[m.cand].st, m.lvl)
  /\ msgs' = msgs \ {m}
  /\ UNCHANGED <<node, instSet, budget>>

\* F21 fix: the owner-space liveness probe reclaims a stale ownership
\* transfer parked at a dead DID. An update stuck at an Absent target is
\* the parked pending_collectables entry; the probe's DEAD verdict
\* (owner-space replica beyond GLOBAL -- the last replica to die, per
\* DeadOwnerClean) erases it. The impl defers the verdict at a PENDING
\* owner-space replica until the round resolves; enabling on the stable
\* owner-space state models that.
DropStaleUpd(m) ==
  /\ StaleUpdateProbe
  /\ m.t = "upd"
  /\ node[m.dst].st = "Absent"
  /\ node[OwnerSpace].st \in {"Local", "Absent"}
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
    [] m.t = "fail"    -> RecvFail(m)
    [] m.t = "unpin"   -> RecvUnpin(m)
    [] OTHER           -> FALSE

ProgressNext ==
  \/ \E m \in msgs : Recv(m)
  \/ \E m \in msgs : AbandonAcq(m)
  \/ \E m \in msgs : DropStaleUpd(m)
  \/ \E n \in Nodes, l \in Lvls : DropRef(n, l)
  \/ \E n \in Nodes : Collect(n)

CreateNext ==
  \/ \E n \in Nodes, m \in Nodes, l \in Lvls : Pack(n, m, l)
  \/ \E n \in Nodes, l \in Lvls : AcquireLocal(n, l) \/ AcqStart(n, l)
  \/ \E m \in Nodes, c \in Nodes : Spawn(m, c)

Next == ProgressNext \/ CreateNext

Spec == Init /\ [][Next]_vars /\ WF_vars(ProgressNext)

(***************************************************************************)
(* Invariants                                                             *)
(***************************************************************************)
CountBound == MaxPacks + MaxAcqs + MaxSpawns + 2
LFun == [Lvls -> 0..CountBound]

TypeOK ==
  /\ node \in [Nodes ->
       [st: States, refs: LFun, sent: LFun, recv: LFun, pins: LFun,
        reg: BOOLEAN, own: Nodes, ver: Nat, lastR: Nat, voted: Nat,
        clk: Nat, pclk: Nat, bump: BOOLEAN,
        rflag: BOOLEAN, gen: Nat, died: BOOLEAN, vdied: BOOLEAN,
        rnd: [act: BOOLEAN, lvl: Lvls, rid: Nat, ow: Nodes, par: Nodes,
              wait: SUBSET Nodes, nrdy: Nodes, tS: Nat, tR: Nat]]]
  /\ instSet \subseteq (Nodes \ TreeNodes)
  /\ budget \in [packs: 0..MaxPacks, acqs: 0..MaxAcqs, spawns: 0..MaxSpawns]

(* References of each level are only held at states that permit them.     *)
SafeRefs ==
  \A n \in Nodes, l \in Lvls :
    node[n].refs[l] > 0 => HoldsAt(node[n].st, l)

(* One-way transitions per lifetime: full death and validity death.       *)
DiedStaysDead ==
  \A n \in Nodes :
    /\ (node[n].died => node[n].st \in {"Local", "Absent"})
    /\ (node[n].vdied => node[n].st \notin {"Valid", "PGlobal"})

(* A committed GLOBAL-level collection was globally justified.            *)
DeadOwnerClean ==
  node[OwnerSpace].st \in {"Local", "Absent"} =>
    /\ \A n \in Nodes, l \in Lvls : node[n].refs[l] = 0
    /\ \A n \in Nodes : node[n].st \in {"PLocal", "Local", "Absent"}
    /\ ~\E m \in msgs : m.t = "ref"

(* A committed VALID-level downgrade was globally justified: once the     *)
(* owner space has left the valid level, no valid references or valid     *)
(* reference messages exist, and nobody is still fully VALID (PGlobal is  *)
(* permitted while its success is in flight).                             *)
ValidDeadClean ==
  node[OwnerSpace].st \in {"Global", "PLocal", "Local", "Absent"} =>
    /\ \A n \in Nodes : node[n].refs["V"] = 0
    /\ \A n \in Nodes : node[n].st # "Valid"
    /\ ~\E m \in msgs : m.t = "ref" /\ m.lvl = "V"

(* The ruled invariant: GLOBAL-level rounds exist only after the VALID   *)
(* level committed everywhere. PGlobal is allowed (success propagation    *)
(* overlaps; agreement completed at the decision), as are in-flight       *)
(* successes. Nothing fully VALID, no valid refs, no VALID-stamped or     *)
(* valid-payload messages may coexist with any GLOBAL-level round.        *)
StrictLevelOrder ==
  (\/ \E m \in msgs : m.t \in {"req", "succ"} /\ m.lvl = "G"
   \/ \E n \in Nodes : node[n].rnd.act /\ node[n].rnd.lvl = "G")
  => /\ \A k \in Nodes : node[k].st # "Valid" /\ node[k].refs["V"] = 0
     /\ ~\E m2 \in msgs : \/ (m2.t = "ref" /\ (m2.lvl = "V" \/ m2.olv = "V"))
                            \/ (m2.t = "spawn" /\ m2.olv = "V")

OwnerUnique ==
  Cardinality({n \in Nodes : node[n].own = n /\ node[n].st # "Absent"}) <= 1

HandlesCovered ==
  (\E m \in msgs : m.t \in {"spawn", "reg"}) =>
    node[OwnerSpace].st \in {"Valid", "PGlobal", "Global", "PLocal"}

(* Probes for the impl's tripwire asserts (2026-08-24 fuzzer triage):    *)
(* if these HOLD at tree scope the asserts are certified; a violation     *)
(* trace is the legitimate scenario the impl must instead defer.          *)
NoRequestAtBusyNode ==
  \A m \in msgs : (m.t = "req") => ~node[m.dst].rnd.act

RootRetainsOwnership ==
  \A n \in Nodes :
    (node[n].rnd.act /\ node[n].rnd.ow = n) => (node[n].own = n)

(* F19 acquire contract (ruling 2026-08-31): an acquire may fail only if *)
(* the object actually committed its downgrade at the requested level.   *)
(* Stated on the wire: while a deny for level l is in flight, the        *)
(* l-level death must be globally justified (the stable post-commit      *)
(* facts of ValidDeadClean / DeadOwnerClean). A violation trace is a     *)
(* spurious deny -- the class of failure behind the expression.cc:269    *)
(* aborts.                                                               *)
FinalFail(m) == \/ m.t = "fail"
                \/ (m.t = "deny" /\ AcquireMode \in {"deny", "park", "final"})
AcquireContract ==
  \A m \in msgs : FinalFail(m) =>
    IF m.lvl = "V"
    THEN /\ \A n \in Nodes : node[n].st # "Valid" /\ node[n].refs["V"] = 0
         /\ ~\E m2 \in msgs : m2.t = "ref" /\ (m2.lvl = "V" \/ m2.olv = "V")
    ELSE /\ \A n \in Nodes : node[n].st \in {"PLocal", "Local", "Absent"}
         /\ \A n \in Nodes, l \in Lvls : node[n].refs[l] = 0
         /\ ~\E m2 \in msgs : m2.t = "ref"

(* Every acquire chase eventually resolves (grant, deny, or a documented *)
(* model-bound truncation). A chase parked forever is a requester thread *)
(* waiting forever on its ready event in the implementation -- a hang    *)
(* that EventualCollection alone cannot see, because its all-Absent      *)
(* disjunct does not require the network to drain.                       *)
AcqResolved == <>[](~\E m \in msgs : m.t \in {"acq", "deny", "fail"})

(* F21: parked ownership transfers must not leak. An update message     *)
(* stuck forever at an Absent destination IS the leaked                 *)
(* pending_collectables entry (and, before the probe fix, a booby trap  *)
(* for any later blocking find or find_or_request on that DID).         *)
UpdsDrain == <>[](~\E m \in msgs : m.t = "upd")

(* Bounded-liveness cap excuse: quiescent states where the ONLY missing
   step is a retry the round budget forbids. Requires: empty network, no
   open aggregation, and a LIVE SELF-BELIEVING OWNER that is round-capped
   (it would retry with more budget). Deliberately does NOT excuse: stuck
   messages (F13 wedges), a parked uncapped owner, or lost ownership.   *)
BudgetQuiesced ==
  /\ msgs = {}
  /\ \A n \in Nodes : ~node[n].rnd.act
  /\ \E n \in Nodes : /\ node[n].own = n
                      /\ node[n].st \in {"Valid", "Global"}
                      /\ Max(node[n].clk, node[n].lastR) >= MaxRounds

EventualCollection ==
  <>[]((\A n \in Nodes : node[n].st = "Absent")
       \/ (BoundedLiveness /\ BudgetQuiesced))

=============================================================================
