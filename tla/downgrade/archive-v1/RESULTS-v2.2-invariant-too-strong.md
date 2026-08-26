## bisectlive-RestartForwarding  (21:26:53)
config: BisectLiveRestartForwarding.cfg — expectation: violation: dropped restarts leak (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 15)
235155 states generated, 65851 distinct states found, 7565 states left on queue.
Finished in 03s at (2026-08-17 21:26:53)

## bisectlive-RegHandshake  (21:26:56)
config: BisectLiveRegHandshake.cfg — expectation: violation: registration wedge / no nudge (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
79878 states generated, 29817 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 21:26:55)

## bisectlive-OwnershipVersioning  (21:26:59)
config: BisectLiveOwnershipVersioning.cfg — expectation: violation: ownership clobber leak (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 13)
205241 states generated, 64005 distinct states found, 13805 states left on queue.
Finished in 03s at (2026-08-17 21:26:59)

## bisectlive-RegistrationGate  (21:28:54)
config: BisectLiveRegistrationGate.cfg — expectation: unknown
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 44.
Progress(17) at 2026-08-17 21:27:03: 193,427 states generated (193,427 s/min), 64,624 distinct states found (64,624 ds/min), 17,923 states left on queue.
Finished in 01min 54s at (2026-08-17 21:28:54)

## old-liveness  (21:28:56)
config: DowngradeOldLive.cfg — expectation: violation expected: current protocol leaks
**VIOLATION**: Error: Temporal properties were violated. (trace length 11)
65449 states generated, 23089 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 21:28:56)

## bisect-ReceiptChecks  (21:29:04)
config: BisectReceiptChecks.cfg — expectation: expected redundant in Stage 1 (single level => at most one commit)
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 17)
The depth of the complete state graph search is 17.
2491232 states generated, 569771 distinct states found, 209136 states left on queue.
Finished in 06s at (2026-08-17 21:29:03)

## bisect-RoundTagging  (21:29:12)
config: BisectRoundTagging.cfg — expectation: expected redundant in Stage 1 (subsumed by clocks)
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 18)
The depth of the complete state graph search is 18.
3015400 states generated, 705687 distinct states found, 263159 states left on queue.
Finished in 08s at (2026-08-17 21:29:12)

## bisect-RegistrationGate  (21:29:21)
config: BisectRegistrationGate.cfg — expectation: unknown: gate may be redundant given nudge+records
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 16)
The depth of the complete state graph search is 17.
2620128 states generated, 676065 distinct states found, 296588 states left on queue.
Finished in 07s at (2026-08-17 21:29:20)

## safety-4node  (21:29:27)
config: Downgrade4.cfg — expectation: pass expected: relay aggregation with >=2 children
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 18)
The depth of the complete state graph search is 18.
1606741 states generated, 385140 distinct states found, 123154 states left on queue.
Finished in 05s at (2026-08-17 21:29:27)

## safety-bigbudget  (21:30:03)
config: DowngradeBig.cfg — expectation: pass expected: larger reference/round budgets
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 17)
The depth of the complete state graph search is 17.
16129552 states generated, 3545783 distinct states found, 1561052 states left on queue.
Finished in 35s at (2026-08-17 21:30:02)

---
matrix pass finished at Mon Aug 17 21:30:03 PDT 2026 (relaunch to resume anything interrupted)
