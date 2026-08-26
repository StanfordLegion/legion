# Downgrade protocol model-checking matrix — Mon Aug 17 04:20:46 PDT 2026

## safety-exhaustive (DowngradeNew.cfg, 3 nodes, symmetric)
**PASS** (completed by the in-flight run)
The depth of the complete state graph search is 58.
Progress(45) at 2026-08-17 04:46:23: 952,540,108 states generated (17,567,907 s/min), 176,706,757 distinct states found (2,524,600 ds/min), 6,644,142 states left on queue.

## bisect-FlagVeto  (04:53:48)
config: BisectFlagVeto.cfg — expectation: violation: F3 count cancellation (DeadOwnerClean)
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 19)
The depth of the complete state graph search is 21.
9036092 states generated, 2136055 distinct states found, 651513 states left on queue.
Finished in 14s at (2026-08-17 04:54:03)

## bisect-CoveredByHandle  (04:54:03)
config: BisectCoveredByHandle.cfg — expectation: violation: zombie by-handle replica (DeadOwnerClean)
**VIOLATION**: Error: Invariant DeadOwnerClean is violated. (trace length 15)
The depth of the complete state graph search is 15.
429090 states generated, 121249 distinct states found, 53676 states left on queue.
Finished in 01s at (2026-08-17 04:54:05)

## bisect-RegistrationGate  (04:54:05)
config: BisectRegistrationGate.cfg — expectation: unknown: gate may be redundant given nudge+records
**INCONCLUSIVE** (run did not complete cleanly — see log-bisect-RegistrationGate.out)
Progress(15) at 2026-08-17 04:54:08: 1,482,172 states generated (1,482,172 s/min), 422,619 distinct states found (422,619 ds/min), 193,969 states left on queue.

## bisect-OwnershipVersioning  (05:10:20)
config: BisectOwnershipVersioning.cfg — expectation: violation: update resurrection/clobber (DiedStaysDead/OwnerUnique)
**VIOLATION**: Error: Invariant DiedStaysDead is violated. (trace length 12)
The depth of the complete state graph search is 12.
104967 states generated, 33764 distinct states found, 17558 states left on queue.
Finished in 01s at (2026-08-17 05:10:21)

## bisect-RoundTagging  (05:10:21)
config: BisectRoundTagging.cfg — expectation: unknown: rid staleness may be subsumed by clocks in Stage 1
## bisectlive-RestartForwarding  (11:46:50)
config: BisectLiveRestartForwarding.cfg — expectation: violation: dropped restarts leak (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 22)
237967 states generated, 64958 distinct states found, 4851 states left on queue.
Finished in 03s at (2026-08-17 11:46:50)

## bisectlive-RegHandshake  (11:46:52)
config: BisectLiveRegHandshake.cfg — expectation: violation: registration wedge / no nudge (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
80359 states generated, 30175 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 11:46:52)

## bisectlive-OwnershipVersioning  (11:46:56)
config: BisectLiveOwnershipVersioning.cfg — expectation: violation: ownership clobber leak (EventualCollection)
**VIOLATION**: Error: Temporal properties were violated. (trace length 16)
213163 states generated, 65661 distinct states found, 16089 states left on queue.
Finished in 03s at (2026-08-17 11:46:56)

## bisectlive-RegistrationGate  (11:51:03)
config: BisectLiveRegistrationGate.cfg — expectation: unknown
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 47.
Progress(16) at 2026-08-17 11:47:00: 205,138 states generated (205,138 s/min), 68,428 distinct states found (68,428 ds/min), 21,810 states left on queue.
Finished in 04min 06s at (2026-08-17 11:51:03)

## old-liveness  (11:51:05)
config: DowngradeOldLive.cfg — expectation: violation expected: current protocol leaks
**VIOLATION**: Error: Temporal properties were violated. (trace length 15)
52425 states generated, 20105 distinct states found, 0 states left on queue.
Finished in 01s at (2026-08-17 11:51:04)

## bisect-ReceiptChecks  (14:22:19)
config: BisectReceiptChecks.cfg — expectation: expected redundant in Stage 1 (single level => at most one commit)
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 58.
Progress(43) at 2026-08-17 14:16:06: 867,620,158 states generated (32,157,609 s/min), 163,718,881 distinct states found (5,336,137 ds/min), 8,969,973 states left on queue.
Finished in 35min 17s at (2026-08-17 14:22:18)

## bisect-RoundTagging  (17:00:40)
config: BisectRoundTagging.cfg — expectation: expected redundant in Stage 1 (subsumed by clocks)
**PASS** (exhaustive, no violation)
The depth of the complete state graph search is 58.
Progress(45) at 2026-08-17 16:52:08: 1,073,510,555 states generated (16,501,837 s/min), 203,081,445 distinct states found (2,532,108 ds/min), 8,004,825 states left on queue.
Finished in 01h 09min at (2026-08-17 17:00:39)

