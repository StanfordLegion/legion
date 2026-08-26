---------------------------- MODULE Downgrade ----------------------------
(***************************************************************************)
(* Stage-2 model of the Legion DistributedCollectable downgrade protocol: *)
(* TWO reference levels (VALID -> GLOBAL -> LOCAL), flat topology with    *)
(* the owner-space relay. Stage-1 (single level) is archived in           *)
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
(*   - MixedLevelReady: a node BELOW a VALID round's level votes ready    *)
(*     trivially without state change (proposed), vs the current code's   *)
(*     base-can_downgrade vote that freezes it at PENDING_LOCAL (old)     *)
(*   - ReceiptChecks re-adjudication: stale VALID successes now coexist   *)
(*     with GLOBAL rounds                                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  Nodes,               \* address spaces
  OwnerSpace,          \* the object's owner space (in Nodes)
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
  CatchUp,             \* above-level node self-downgrades on a fresh request
  MixedLevelReady      \* below-level node votes trivially ready (proposed)

ASSUME /\ OwnerSpace \in Nodes
       /\ MaxPacks \in Nat /\ MaxAcqs \in Nat
       /\ MaxRounds \in Nat /\ MaxSpawns \in Nat

Max(a, b) == IF a >= b THEN a ELSE b

Symm == Permutations(Nodes \ {OwnerSpace})

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
AdoptLvl(cur, new) == IF cur = "G" \/ new = "G" THEN "G" ELSE "V"

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
DenyMsg(d)                == Mk("deny", [dst |-> d])

(***************************************************************************)
(* Initial state: the owner-space replica exists VALID holding one valid  *)
(* reference (e.g. an IndexTreeNode created tree_valid).                   *)
(***************************************************************************)
InitNode(n) ==
  [st    |-> IF n = OwnerSpace THEN "Valid" ELSE "Absent",
   refs  |-> IF n = OwnerSpace THEN [x \in Lvls |-> IF x = "V" THEN 1 ELSE 0]
             ELSE ZeroL,
   sent  |-> ZeroL, recv |-> ZeroL, pins |-> ZeroL,
   reg   |-> (n = OwnerSpace),
   own   |-> OwnerSpace, ver |-> 0,
   lastR |-> 0, voted |-> 0,
   clk   |-> 0, pclk |-> 0, bump |-> FALSE,
   rflag |-> FALSE, gen |-> 0, died |-> FALSE, vdied |-> FALSE,
   olv   |-> "V",
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


\* clock-bump applicability: valid packs always honor the flag; global
\* packs only bump outside valid-level states (matching pack_global_ref)
BumpApplies(st, l) == l = "V" \/ st \notin {"Valid", "PGlobal"}

\* check_for_downgrade at a node that believes it is the downgrade owner.
\* An owner alone chains levels inline (perform_downgrade -> can_delete).
TryRound(n, f, is) ==
  IF /\ f[n].own = n
     /\ f[n].st \in {"Valid", "Global"}
     /\ f[n].refs[f[n].olv] = 0
     /\ (RegistrationGate => f[n].reg)
     /\ ~f[n].rnd.act
     /\ Max(f[n].clk, f[n].lastR) < MaxRounds
  THEN LET l     == f[n].olv   \* STRICT ORDER: rounds run at object level
           parts == IF n = OwnerSpace THEN is ELSE {OwnerSpace}
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
                                               ![n].olv = "G",
                                               ![n].died = TRUE], out |-> {}]
                         ELSE [f |-> [f EXCEPT
                                        ![n].st = IF f[n].st = "Valid"
                                                  THEN "Global" ELSE f[n].st,
                                        ![n].vdied = @ \/ (f[n].st = "Valid"),
                                        ![n].olv = "G"], out |-> {}]
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
  /\ LET c2 == IF node[n].bump /\ BumpApplies(node[n].st, l)
               THEN node[n].clk + 1 ELSE node[n].clk IN
     /\ budget' = [budget EXCEPT !.packs = @ + 1]
     /\ msgs' = msgs \cup {RefMsg(m, n, l, budget.packs + 1, c2,
                                   node[n].olv)}
     /\ IF n = OwnerSpace
        THEN /\ instSet' = instSet \cup {m}
             /\ node' = [node EXCEPT ![n].sent[l] = @ + 1,
                                     ![n].clk = c2, ![n].bump = FALSE,
                                     ![n].rnd.nrdy =
                                       IF node[n].rnd.act /\ m \notin instSet
                                       THEN m ELSE @]
        ELSE /\ instSet' = instSet
             /\ node' = [node EXCEPT ![n].sent[l] = @ + 1,
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
Spawn(m, c, l) ==
  /\ m # OwnerSpace
  /\ node[m].st = "Absent"
  /\ m \notin instSet
  /\ budget.spawns < MaxSpawns
  /\ instSet' = instSet \cup {m}
  /\ budget' = [budget EXCEPT !.spawns = @ + 1]
  /\ IF CoveredByHandle
     THEN /\ IF l = "V" THEN node[OwnerSpace].st = "Valid"
             ELSE node[OwnerSpace].st \in {"Valid", "Global"}
          /\ node[c].refs[l] > node[c].pins[l]
          /\ LET c2 == IF node[OwnerSpace].bump /\
                          BumpApplies(node[OwnerSpace].st, l)
                       THEN node[OwnerSpace].clk + 1
                       ELSE node[OwnerSpace].clk
             IN /\ node' = [node EXCEPT
                             ![c].pins[l] = @ + 1,
                             ![OwnerSpace].sent[l] = @ + 1,
                             ![OwnerSpace].clk = c2,
                             ![OwnerSpace].bump = FALSE,
                             ![OwnerSpace].rnd.nrdy =
                               IF node[OwnerSpace].rnd.act THEN m ELSE @]
                /\ msgs' = msgs \cup
                     {SpawnMsg(m, c, l, node[OwnerSpace].own,
                               node[OwnerSpace].ver,
                               node[OwnerSpace].lastR, c2,
                               node[OwnerSpace].olv)}
     ELSE /\ node[OwnerSpace].st \in {"Valid", "PGlobal", "Global", "PLocal"}
          /\ node' = [node EXCEPT ![OwnerSpace].rnd.nrdy =
                        IF node[OwnerSpace].rnd.act THEN m ELSE @]
          /\ msgs' = msgs \cup {SpawnMsg(m, c, l, OwnerSpace, 0, 0, 0,
                                          node[OwnerSpace].olv)}

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
       /\ node' = [node EXCEPT ![d].st = LiveOf(l),
                               ![d].refs = [x \in Lvls |->
                                              IF x = l THEN 1 ELSE 0],
                               ![d].recv = [x \in Lvls |->
                                              IF x = l THEN 1 ELSE 0],
                               ![d].sent = ZeroL, ![d].pins = ZeroL,
                               ![d].reg = FALSE,
                               ![d].gen = ng, ![d].died = FALSE,
                               ![d].vdied = (m.olv = "G"),
                               ![d].olv = m.olv,
                               ![d].own = OwnerSpace, ![d].ver = 0,
                               ![d].voted = 0, ![d].clk = m.clk,
                               ![d].pclk = 0, ![d].bump = FALSE]
       /\ msgs' = (msgs \ {m}) \cup
                    (IF d = OwnerSpace THEN {} ELSE {RegMsg(d, ng)})
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET wasPend == node[d].st = PendOf(l)
           quiet   == node[d].refs[l] = 0
           \* Q-D: a valid ref reaching a never-valid GLOBAL replica of a
           \* still-valid object promotes it (object-monotone)
           promoteV == l = "V" /\ node[d].st = "Global" /\ ~node[d].vdied
           newSt   == IF wasPend THEN LiveOf(l)
                      ELSE IF promoteV THEN "Valid" ELSE node[d].st
           c2      == Max(node[d].clk, m.clk)
           notify  == node[d].own # d /\ (wasPend \/ quiet)
                      /\ ~node[d].rnd.act
           park    == \/ (node[d].own # d /\ ~(wasPend \/ quiet)
                          /\ ~node[d].rnd.act)
                      \/ (FlagVeto /\ node[d].rnd.act)
       IN /\ node' = [node EXCEPT ![d].st = newSt, ![d].refs[l] = @ + 1,
                                  ![d].recv[l] = @ + 1, ![d].clk = c2,
                                  ![d].olv = AdoptLvl(@, m.olv),
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
                                    ![d].olv = m.olv,
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
                                    ![d].olv = m.olv,
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
                                       ![d].olv = AdoptLvl(@, m.olv),
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
                    /\ \E l \in Lvls : node[O].sent[l] # node[O].recv[l]
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
                              node[O].clk, node[O].olv)}
             ELSE {}
           hairyOut ==
             IF hairy
             THEN {UpdMsg(s, O, node[O].olv, newVer,
                          node[O].lastR, node[O].clk)}
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
  THEN LET adopt == IF OwnershipVersioning
                    THEN m.ver > node[s].ver
                    ELSE TRUE
           f0 == [node EXCEPT
                    ![s].reg   = TRUE,
                    ![s].own   = IF adopt THEN m.ow ELSE @,
                    ![s].ver   = IF adopt THEN m.ver ELSE @,
                    ![s].clk   = Max(node[s].clk, m.clk),
                    ![s].olv   = AdoptLvl(node[s].olv, m.olv),
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
(*   BELOW level -> MixedLevelReady: trivially ready with this level's    *)
(*     counters, state untouched (proposed); old: base-can_downgrade on   *)
(*     gc refs and a PENDING_LOCAL freeze (the current code's behavior)   *)
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
                       {RespMsg(m.src, d, l, m.rid, d, 0, 0, node[d].clk)}
          /\ UNCHANGED <<node, instSet, budget>>
     ELSE LET adopt == IF OwnershipVersioning
                       THEN m.ver > node[d].ver
                       ELSE TRUE
              own2 == IF adopt THEN m.ow ELSE node[d].own
              ver2 == IF adopt THEN m.ver ELSE node[d].ver
              lr2  == Max(node[d].lastR, m.rid)
              olv2 == AdoptLvl(node[d].olv, m.lvl)
              \* catch-up on a G-round request (a commit proof under the
              \* strict-order design): new mode only from PGlobal (the
              \* node's vote receipt justifies it); old mode from any
              \* valid-level state (master's while-loop -> finding #12)
              cu     == /\ AboveLevel(node[d].st, l) /\ CatchUp
                        /\ (MixedLevelReady => node[d].st = "PGlobal")
              stEff  == IF cu THEN "Global" ELSE node[d].st
              below  == BelowLevel(stEff, l)
              canAt  == /\ stEff \in {LiveOf(l), PendOf(l)}
                        /\ node[d].refs[l] = 0
                        /\ (RegistrationGate => node[d].reg)
                        /\ node[d].clk <= m.rid
                        /\ (FlagVeto => ~node[d].rflag)
              \* old code's below-level vote: base can_downgrade on gc refs
              canBelowOld == /\ below /\ ~MixedLevelReady
                             /\ node[d].refs["G"] = 0
                             /\ node[d].clk <= m.rid
                             /\ (FlagVeto => ~node[d].rflag)
              canBelowNew == below /\ MixedLevelReady
              can == \/ canAt
                     \/ canBelowOld
                     \/ canBelowNew
                     \/ (AboveLevel(node[d].st, l) /\ ~CatchUp /\ FALSE)
          IN
          IF ~can
          THEN /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2, ![d].olv = olv2,
                                       ![d].st = stEff,
                                       ![d].vdied = @ \/ cu,
                                       ![d].voted = IF cu THEN 0 ELSE @,
                                       ![d].rflag = IF FlagVeto THEN FALSE
                                                    ELSE @]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(m.src, d, l, m.rid, d, 0, 0,
                                     node[d].clk)}
               /\ UNCHANGED <<instSet, budget>>
          ELSE IF d = OwnerSpace /\ instSet \ {m.ow} # {}
          THEN \* relay: fan out (relay's own vote happens at aggregation)
               LET kids == instSet \ {m.ow} IN
               /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2, ![d].olv = olv2,
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
               LET newSt == IF canBelowNew THEN stEff ELSE PendOf(l)
                   newVoted == IF canBelowNew THEN node[d].voted ELSE m.rid
               IN
               /\ node' = [node EXCEPT ![d].own = own2, ![d].ver = ver2,
                                       ![d].lastR = lr2, ![d].olv = olv2,
                                       ![d].st = newSt,
                                       ![d].vdied = @ \/ cu,
                                       ![d].voted = newVoted,
                                       ![d].clk = m.rid,
                                       ![d].pclk = m.rid,
                                       ![d].bump = TRUE]
               /\ msgs' = (msgs \ {m}) \cup
                            {RespMsg(m.src, d, l, m.rid, m.ow,
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
              \* a below-level root (e.g. a GLOBAL replica owning a
              \* still-VALID object) contributes a trivially-ready vote
              selfOk == \/ CanDowngradeL(node, r, l)
                        \/ (BelowLevel(node[r].st, l) /\ MixedLevelReady)
              ready  == /\ nrdy2 = r
                        /\ selfOk
                        /\ tS2 + node[r].sent[l] = tR2 + node[r].recv[l]
                        /\ causal
                        /\ (FlagVeto => ~node[r].rflag)
          IN
          IF ready
          THEN LET succTo == IF r = OwnerSpace THEN instSet
                             ELSE {OwnerSpace}
                   f0 == [node EXCEPT ![r].st = DownTo(l),
                                      ![r].vdied = @ \/ (l = "V"),
                                      ![r].died = @ \/ (l = "G"),
                                      ![r].olv = IF l = "V" THEN "G" ELSE @,
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
                               {UpdMsg(nrdy2, r, node[r].olv, v2,
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
          LET below  == BelowLevel(node[r].st, l)
              atLvl  == node[r].st \in {LiveOf(l), PendOf(l)}
              rdyAt  == /\ atLvl
                        /\ node[r].refs[l] = 0
                        /\ (RegistrationGate => node[r].reg)
                        /\ c2 <= node[r].pclk
                        /\ (FlagVeto => ~node[r].rflag)
              rdyBelowNew == below /\ MixedLevelReady
              rdyBelowOld == /\ below /\ ~MixedLevelReady
                             /\ node[r].refs["G"] = 0
                             /\ c2 <= node[r].pclk
                             /\ (FlagVeto => ~node[r].rflag)
          IN
          IF rdyAt \/ rdyBelowOld
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
          ELSE IF rdyBelowNew
          THEN /\ node' = [node EXCEPT ![r].clk = Max(c2, node[r].pclk),
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
      fwd == IF d = OwnerSpace /\ apply
             THEN {SuccMsg(k, d, l, m.rid) :
                     k \in instSet \ {m.src, node[d].own}}
             ELSE {}
  IN
  IF node[d].st = "Absent"
  THEN /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<node, instSet, budget>>
  ELSE IF ~apply
  THEN /\ node' = [node EXCEPT ![d].olv =
                      IF l = "V" THEN "G" ELSE @]  \* commit proof
       /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<instSet, budget>>
  ELSE /\ node' = [node EXCEPT ![d].st = DownTo(l),
                               ![d].vdied = @ \/ (l = "V"),
                               ![d].died = @ \/ (l = "G"),
                               ![d].olv = IF l = "V" THEN "G" ELSE @,
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
              olv2 == AdoptLvl(node[d].olv, m.olv)
              \* rollback only if the failed round was at OUR pending
              \* level; olv=G at a PGlobal node is a V-commit proof
              rollV == OwnershipVersioning /\ node[d].st = "PGlobal"
                       /\ olv2 = "V"
              commitV == OwnershipVersioning /\ node[d].st = "PGlobal"
                         /\ olv2 = "G"
              rollG == OwnershipVersioning /\ node[d].st = "PLocal"
              newSt == IF oldPromote THEN LiveOf(m.olv)
                       ELSE IF rollV THEN "Valid"
                       ELSE IF commitV \/ rollG THEN "Global"
                       ELSE node[d].st
              f0 == [node EXCEPT ![d].own = d, ![d].ver = m.ver,
                                 ![d].lastR = Max(node[d].lastR, m.lastR),
                                 ![d].clk = Max(node[d].clk, m.clk),
                                 ![d].olv = IF OwnershipVersioning
                                            THEN olv2 ELSE @,
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
            node[d].refs[node[d].olv] = 0 /\
            (RegistrationGate => node[d].reg))
  THEN /\ node' = [node EXCEPT ![d].clk = c2]
       /\ msgs' = msgs \ {m}
       /\ UNCHANGED <<instSet, budget>>
  ELSE IF m.cand # d
  THEN LET v2 == node[d].ver + 1 IN
       /\ node' = [node EXCEPT ![d].own = m.cand, ![d].ver = v2,
                               ![d].clk = c2]
       /\ msgs' = (msgs \ {m}) \cup
                    {UpdMsg(m.cand, d, node[d].olv, v2,
                            node[d].lastR, c2)}
       /\ UNCHANGED <<instSet, budget>>
  ELSE LET f0 == [node EXCEPT ![d].clk = c2]
           tr == TryRound(d, f0, instSet)
       IN /\ node' = tr.f
          /\ msgs' = (msgs \ {m}) \cup tr.out
          /\ UNCHANGED <<instSet, budget>>

\* Remote acquire chase at level m.lvl (grant = counted RefMsg).
RecvAcq(m) ==
  LET d == m.dst
      l == m.lvl
      orig == m.cand
      grantable == IF l = "V" THEN node[d].st = "Valid"
                   ELSE node[d].st \in {"Valid", "PGlobal", "Global"}
  IN
  IF /\ grantable /\ node[d].own = d
     /\ budget.packs < MaxPacks
  THEN LET c2 == IF node[d].bump /\ BumpApplies(node[d].st, l)
                 THEN node[d].clk + 1 ELSE node[d].clk IN
       /\ node' = [node EXCEPT ![d].sent[l] = @ + 1, ![d].clk = c2,
                               ![d].bump = FALSE]
       /\ budget' = [budget EXCEPT !.packs = @ + 1]
       /\ msgs' = (msgs \ {m}) \cup
                    {RefMsg(orig, d, l, budget.packs + 1, c2, node[d].olv)}
       /\ UNCHANGED instSet
  ELSE IF /\ node[d].st # "Absent" /\ node[d].own # d
          /\ m.hops > 0
  THEN /\ msgs' = (msgs \ {m}) \cup
                    {AcqMsg(node[d].own, orig, l, m.hops - 1)}
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
  \/ \E n \in Nodes, l \in Lvls : DropRef(n, l)
  \/ \E n \in Nodes : Collect(n)

CreateNext ==
  \/ \E n \in Nodes, m \in Nodes, l \in Lvls : Pack(n, m, l)
  \/ \E n \in Nodes, l \in Lvls : AcquireLocal(n, l) \/ AcqStart(n, l)
  \/ \E m \in Nodes, c \in Nodes, l \in Lvls : Spawn(m, c, l)

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
        olv: Lvls,
        rnd: [act: BOOLEAN, lvl: Lvls, rid: Nat, ow: Nodes, par: Nodes,
              wait: SUBSET Nodes, nrdy: Nodes, tS: Nat, tR: Nat]]]
  /\ instSet \subseteq (Nodes \ {OwnerSpace})
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

(* The ruled invariant, checkable soundness half: no replica ever        *)
(* believes the valid level is dead while it is not.                      *)
ObjLvlSound ==
  (\E n \in Nodes : node[n].st # "Absent" /\ node[n].olv = "G") =>
    /\ \A k \in Nodes : node[k].st # "Valid" /\ node[k].refs["V"] = 0
    /\ ~\E m \in msgs : m.t = "ref" /\ m.lvl = "V"

(* The ruled invariant, strict half: GLOBAL-level rounds exist only after *)
(* the VALID level committed everywhere (PGlobal allowed: success         *)
(* propagation overlaps by design; agreement completed at the decision).  *)
StrictLevelOrder ==
  (\/ \E m \in msgs : m.t \in {"req", "succ"} /\ m.lvl = "G"
   \/ \E n \in Nodes : node[n].rnd.act /\ node[n].rnd.lvl = "G")
  => /\ \A k \in Nodes : node[k].st # "Valid" /\ node[k].refs["V"] = 0
     /\ ~\E m2 \in msgs : m2.t = "ref" /\ m2.lvl = "V"

OwnerUnique ==
  Cardinality({n \in Nodes : node[n].own = n /\ node[n].st # "Absent"}) <= 1

HandlesCovered ==
  (\E m \in msgs : m.t \in {"spawn", "reg"}) =>
    node[OwnerSpace].st \in {"Valid", "PGlobal", "Global", "PLocal"}

EventualCollection == <>[](\A n \in Nodes : node[n].st = "Absent")

=============================================================================
