# Downgrade protocol v2 matrix — Wed Aug 19 13:24:13 PDT 2026

## live-base  (13:24:24)
config: DowngradeNewLive.cfg — expectation: pass expected: full design liveness
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 41.
Progress(21) at 2026-08-19 13:24:17: 181,913 states generated (181,913 s/min), 60,407 distinct states found (60,407 ds/min), 8,816 states left on queue.
Finished in 10s at (2026-08-19 13:24:23)

## bisectlive-RestartForwarding  (13:24:26)
config: BisectLiveRestartForwarding.cfg — expectation: violation: dropped restarts leak
**VIOLATION**: Error: Temporal properties were violated. (trace length 14)
115036 states generated, 34025 distinct states found, 0 states left on queue.
Finished in 02s at (2026-08-19 13:24:26)

## bisectlive-RegHandshake  (13:24:29)
config: BisectLiveRegHandshake.cfg — expectation: violation: registration nudge missing
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
107694 states generated, 38593 distinct states found, 0 states left on queue.
Finished in 02s at (2026-08-19 13:24:29)

## bisectlive-OwnershipVersioning  (13:24:33)
config: BisectLiveOwnershipVersioning.cfg — expectation: violation: stale forwarded-restart transfers
**VIOLATION**: Error: Temporal properties were violated. (trace length 19)
174324 states generated, 61195 distinct states found, 12029 states left on queue.
Finished in 03s at (2026-08-19 13:24:33)

## bisectlive-RegistrationGate  (13:25:05)
config: BisectLiveRegistrationGate.cfg — expectation: unknown: gate may be redundant
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 42.
Progress(19) at 2026-08-19 13:24:37: 182,607 states generated (182,607 s/min), 64,018 distinct states found (64,018 ds/min), 14,210 states left on queue.
Finished in 31s at (2026-08-19 13:25:04)

## old-liveness  (13:25:07)
config: DowngradeOldLive.cfg — expectation: violation expected: current protocol leaks
**VIOLATION**: Error: Temporal properties were violated. (trace length 17)
61633 states generated, 22849 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-19 13:25:06)

## safety-exhaustive  (14:11:59)
config: DowngradeNew.cfg — expectation: pass expected: full design safety
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 56.
Progress(45) at 2026-08-19 14:07:15: 600,120,843 states generated (14,870,631 s/min), 118,421,891 distinct states found (2,276,743 ds/min), 3,754,505 states left on queue.
Finished in 46min 52s at (2026-08-19 14:11:59)

## bisect-FlagVeto  (14:12:25)
config: BisectFlagVeto.cfg — expectation: violation: F3 count cancellation
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 20)
The depth of the complete state graph search is 21.
9427905 states generated, 2372926 distinct states found, 651941 states left on queue.
Finished in 24s at (2026-08-19 14:12:24)

## bisect-CoveredByHandle  (14:12:28)
config: BisectCoveredByHandle.cfg — expectation: violation: uncovered by-handle zombie
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 14)
The depth of the complete state graph search is 14.
592397 states generated, 161054 distinct states found, 68884 states left on queue.
Finished in 02s at (2026-08-19 14:12:28)

## bisect-OwnershipVersioning  (14:12:30)
config: BisectOwnershipVersioning.cfg — expectation: violation: update resurrection
**VIOLATION**: Error: Invariant DiedStaysDead is violated. (trace length 12)
The depth of the complete state graph search is 12.
129970 states generated, 42832 distinct states found, 22252 states left on queue.
Finished in 01s at (2026-08-19 14:12:30)

## bisect-ReceiptChecks  (14:38:12)
config: BisectReceiptChecks.cfg — expectation: open: receipts necessity under v3 counted creations
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 56.
Progress(40) at 2026-08-19 14:32:36: 483,081,148 states generated (31,123,312 s/min), 99,365,694 distinct states found (5,446,441 ds/min), 6,685,333 states left on queue.
Finished in 25min 41s at (2026-08-19 14:38:11)

## bisect-RegistrationGate  (18:17:43)
config: BisectRegistrationGate.cfg — expectation: open: gate necessity in v2
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 57.
Progress(48) at 2026-08-19 18:12:40: 2,624,228,209 states generated (13,957,434 s/min), 481,068,344 distinct states found (1,598,428 ds/min), 5,130,188 states left on queue.
Finished in 03h 39min at (2026-08-19 18:17:42)

## safety-4node  (18:24:26)
config: Downgrade4.cfg — expectation: pass expected: relay aggregation with >=2 children
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 62.
Progress(30) at 2026-08-19 18:19:47: 41,418,126 states generated (21,533,523 s/min), 9,716,347 distinct states found (4,645,435 ds/min), 1,514,436 states left on queue.
Finished in 06min 41s at (2026-08-19 18:24:25)

## safety-bigbudget  (terminated by decision 2026-08-20 01:45)
config: DowngradeBig.cfg — expectation: pass expected: larger budgets
**BOUNDED-CLEAN** (not exhaustive): 3.62B states generated, 793M distinct,
depth 30, no violation after ~7.3h; terminated ahead of disk exhaustion.
(~25% smaller than the v2 space at equal depth; robustness sweep only —
all design-deciding runs completed exhaustively.)

---
matrix pass finished (bigbudget closed by decision) 2026-08-20 01:45
---
matrix pass finished at Thu Aug 20 01:43:47 PDT 2026 (relaunch to resume anything interrupted)
