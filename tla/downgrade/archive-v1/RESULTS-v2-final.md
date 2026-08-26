# Downgrade protocol v2 matrix — Mon Aug 17 22:09:42 PDT 2026

## live-base  (22:10:14)
config: DowngradeNewLive.cfg — expectation: pass expected: full design liveness
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 49.
Progress(19) at 2026-08-17 22:09:46: 202,573 states generated (202,573 s/min), 62,406 distinct states found (62,406 ds/min), 12,192 states left on queue.
Finished in 31s at (2026-08-17 22:10:14)

## bisectlive-RestartForwarding  (22:10:18)
config: BisectLiveRestartForwarding.cfg — expectation: violation: dropped restarts leak
**VIOLATION**: Error: Temporal properties were violated. (trace length 15)
211080 states generated, 61984 distinct states found, 9297 states left on queue.
Finished in 03s at (2026-08-17 22:10:18)

## bisectlive-RegHandshake  (22:10:20)
config: BisectLiveRegHandshake.cfg — expectation: violation: registration nudge missing
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
79878 states generated, 29817 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 22:10:20)

## bisectlive-OwnershipVersioning  (22:10:24)
config: BisectLiveOwnershipVersioning.cfg — expectation: violation: stale forwarded-restart transfers
**VIOLATION**: Error: Temporal properties were violated. (trace length 18)
205524 states generated, 64771 distinct states found, 13718 states left on queue.
Finished in 03s at (2026-08-17 22:10:24)

## bisectlive-RegistrationGate  (22:12:18)
config: BisectLiveRegistrationGate.cfg — expectation: unknown: gate may be redundant
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 43.
Progress(17) at 2026-08-17 22:10:28: 188,878 states generated (188,878 s/min), 62,795 distinct states found (62,795 ds/min), 17,078 states left on queue.
Finished in 01min 52s at (2026-08-17 22:12:17)

## old-liveness  (22:12:20)
config: DowngradeOldLive.cfg — expectation: violation expected: current protocol leaks
**VIOLATION**: Error: Temporal properties were violated. (trace length 17)
65449 states generated, 23089 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 22:12:19)

## bisect-FlagVeto  (00:18:26)
config: BisectFlagVeto.cfg — expectation: violation: F3 count cancellation
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 22)
The depth of the complete state graph search is 23.
24958170 states generated, 5837433 distinct states found, 1691171 states left on queue.
Finished in 57s at (2026-08-18 00:18:26)

## bisect-CoveredByHandle  (00:18:30)
config: BisectCoveredByHandle.cfg — expectation: violation: uncovered by-handle zombie
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 15)
The depth of the complete state graph search is 15.
1348392 states generated, 276396 distinct states found, 108675 states left on queue.
Finished in 03s at (2026-08-18 00:18:30)

## bisect-OwnershipVersioning  (00:18:32)
config: BisectOwnershipVersioning.cfg — expectation: violation: update resurrection
**VIOLATION**: Error: Invariant DiedStaysDead is violated. (trace length 12)
The depth of the complete state graph search is 12.
148119 states generated, 41326 distinct states found, 20987 states left on queue.
Finished in 01s at (2026-08-18 00:18:32)

## safety-exhaustive  (01:45:32)
config: DowngradeNew.cfg — expectation: pass expected: full design safety
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 59.
Progress(48) at 2026-08-18 01:39:26: 1,229,999,365 states generated (14,844,606 s/min), 225,470,917 distinct states found (2,065,146 ds/min), 4,605,520 states left on queue.
Finished in 01h 16min at (2026-08-18 01:45:31)

## safety-4node  (07:40:20)
config: Downgrade4.cfg — expectation: pass expected: relay aggregation with >=2 children
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 66.
Progress(49) at 2026-08-18 04:43:54: 300,610,000 states generated (24,075,431 s/min), 59,114,604 distinct states found (4,172,776 ds/min), 1,896,392 states left on queue.
Finished in 03h 19min at (2026-08-18 07:40:20)

## bisect-ReceiptChecks  (13:45:29)
config: BisectReceiptChecks.cfg — expectation: open: receipts vs gated late-cleanup in v2
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 59.
Progress(48) at 2026-08-18 13:39:53: 1,192,548,233 states generated (14,893,234 s/min), 261,450,250 distinct states found (1,991,497 ds/min), 4,209,863 states left on queue.
Finished in 01h 32min at (2026-08-18 13:45:28)

## bisect-RegistrationGate  (terminated by decision 2026-08-19 00:30)
config: BisectRegistrationGate.cfg — expectation: open: gate necessity in v2
**BOUNDED-CLEAN** (not exhaustive): 5.07B states generated, 971M distinct,
depth 42, no violation after ~10.5h. The gate-off state space is ~5x the
base config and exceeds this machine's disk; run terminated. Combined with
the EXHAUSTIVE liveness pass with the gate off, this is strong evidence the
registration gate is redundant in the v2 design at Stage-1 scope. Since the
gate is one condition in can_downgrade, keeping it defensively costs
nothing; deleting it is defensible on this evidence. Re-adjudicate at
Stage 2.

## safety-bigbudget  (terminated by decision 2026-08-19 09:27)
config: DowngradeBig.cfg — expectation: pass expected: larger budgets
**BOUNDED-CLEAN** (not exhaustive): terminated at 3G free disk before queue
exhaustion could complete. Last progress: Progress(30) at 2026-08-19 09:26:53: 4,455,773,404 states generated (5,560,057 s/min), 938,783,669 distinct states found (1,009,527 ds/min),
No violation found. Robustness sweep, not a design-deciding run; the
exhaustive verdicts stand at base budgets (3-node and 4-node configs).

## coverq  (09:27:15)
config: DowngradeCoverQ.cfg — expectation: exhibit: registration can legally reach a dead owner (open question)
**VIOLATION**: Error: Invariant HandlesCovered is violated. (trace length 18)
The depth of the complete state graph search is 18.
2501659 states generated, 584352 distinct states found, 218066 states left on queue.
Finished in 07s at (2026-08-19 09:27:14)

---
matrix pass finished at Wed Aug 19 09:27:15 PDT 2026 (relaunch to resume anything interrupted)
