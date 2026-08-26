---
name: tlc-model-checking
description: Run the TLC model checker on TLA+ specifications — invocation, config files, safety vs liveness runs, checkpoint/resume, cluster execution, and how to read the output. Use when model-checking a protocol spec with TLA+/TLC.
---

# Model checking TLA+ specifications with TLC

TLC is an explicit-state model checker for TLA+. It enumerates every reachable
state of a finite model of your spec and checks invariants (safety) and
temporal properties (liveness) against all of them. This skill covers running
it effectively; it assumes the spec itself already exists.

## Prerequisites

- Java 11+ (`java -version`).
- `tla2tools.jar` — download from https://github.com/tlaplus/tlaplus/releases
  (one jar contains TLC, SANY the parser, and the PDF pretty-printer).

## Basic invocation

```sh
java -XX:+UseParallelGC -Xmx8g -cp tla2tools.jar tlc2.TLC \
     -deadlock -gzip -workers auto \
     -metadir states-myrun -checkpoint 15 \
     -config MyModel.cfg MySpec.tla > log-myrun.out 2>&1
```

Flag notes (the non-obvious ones):

- `-deadlock` **disables** deadlock checking. You almost always want this for
  protocol models with finite action budgets: once budgets are spent the model
  legitimately reaches terminal states, and TLC would otherwise report every
  one as a spurious "deadlock" error.
- `-workers auto` uses all cores; a number caps it. TLC scales well to ~40
  workers for state generation. Liveness checking has a serial phase and
  benefits less.
- `-gzip` compresses the on-disk state queue (big win; the queue can reach
  tens of GB).
- `-metadir DIR` puts the state queue and checkpoints in DIR instead of a
  generated `states/` path. Always set it: it makes cleanup, disk monitoring,
  and checkpoint recovery predictable. Put it on a large, **local** disk —
  never NFS (the queue is I/O-bound; NFS slows runs by an order of magnitude).
- `-checkpoint N` checkpoints every N minutes (default 30). Cheap insurance
  for any run longer than a few minutes.
- `-difftrace` prints only the variables that changed in each state of a
  counterexample trace. Strongly recommended — full-state traces for records
  with many fields are nearly unreadable.
- `-Xmx`: TLC holds the fingerprint set in heap and spills to disk past that.
  8g is fine locally; use most of RAM on a dedicated cluster node (e.g.
  `-Xmx100g` on a 128GB node).
- The config defaults to `<SpecName>.cfg` if `-config` is omitted.

Verdict extraction from a log (also usable as an sbatch epilogue):

```sh
grep -m1 -E "^Error: (Invariant .* is violated|Temporal properties were violated)" log.out
grep -m1 "No error has been found" log.out
```

Exactly one of these should match a completed run; neither matching means the
run was interrupted (see Checkpointing).

## The config file

```
\* MyModel.cfg
CONSTANTS
  Nodes = {n0, n1, n2}     \* bare identifiers become "model values"
  OwnerSpace = n0
  MaxPacks = 2             \* finite budgets bound the state space
  MaxRounds = 4
  UseNewProtocol = TRUE    \* boolean toggles select design variants
INVARIANTS
  TypeOK SafeRefs OwnerUnique
SYMMETRY Symm              \* safety runs only — see Liveness below
SPECIFICATION Spec         \* or: INIT Init  NEXT Next
```

- **Model values** (`n0`, `n1`, …) are opaque, comparable-only values — ideal
  for node/process identities.
- **Budget constants**: TLA+ `Nat` is unbounded, so every counter and every
  action that can repeat must be bounded by a constant (or a `CONSTRAINT`
  state predicate), or the state space is infinite. Structure specs so budgets
  are config-tunable: small for smoke tests, larger for confidence runs.
- **Boolean toggles** guarding mechanisms of the design let one spec serve
  many configs (full design, old design, single-mechanism bisections).
- `SYMMETRY` names a spec-level definition like
  `Symm == Permutations(Nodes \ {OwnerSpace})` (exclude any distinguished
  node). Symmetry reduction typically cuts states by ~|S|! and is sound for
  invariants, but **unsound for liveness** — TLC will warn; never combine
  `SYMMETRY` with `PROPERTIES`.
- `PROPERTIES` lists temporal properties (e.g. `EventualCollection`) for
  liveness checking.

## Safety vs liveness runs

Run them as **separate configs**. Safety (invariants only) is fast per state
and symmetry-reducible; liveness adds a strongly-connected-components search
over the state graph that is several times slower, partly serial, and
symmetry-incompatible. Consequently, give liveness configs smaller budgets.

Liveness requires fairness in the spec, or every property is vacuously
violated by stuttering:

```tla
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
EventualCollection == <>[](\A n \in Nodes : node[n].st = "Absent")
```

A liveness violation prints a trace ending in `Back to state N` — a lasso: the
prefix reaches a cycle in which the property never becomes true. In an
event-driven protocol model, liveness violations are how **lost wakeups**
manifest (a check that only runs inside a message handler never re-fires), so
model checks at the fidelity of the real code: one action per handler
execution, conditions evaluated only where the implementation evaluates them.

## Reading the output

Progress lines:

```
Progress(24) at ...: 72,710,533 states generated (12,054,864 s/min),
  17,363,429 distinct states found (2,845,112 ds/min), 4,718,161 states left on queue.
```

- `Progress(N)` — current BFS depth. Violations report the depth at which the
  shortest counterexample lives; expect diminishing depth growth over time.
- `distinct states found` — real size of the explored space; `ds/min` is the
  meaningful rate for time estimates.
- `states left on queue` — the frontier. Rising = still expanding; falling =
  draining toward completion. Also your disk-usage proxy.

Terminal verdicts:

- `Model checking completed. No error has been found.` **plus** `0 states left
  on queue` = exhaustive pass. TLC also prints fingerprint-collision
  probability estimates; if they are small (≪1) the verdict is trustworthy.
- `Error: Invariant X is violated.` followed by a numbered trace. Each state
  is labeled `State N: <ActionName line …>` — the action sequence alone often
  identifies the failure mode before you read a single variable.
- `Error: Temporal properties were violated.` — lasso trace as above.
- A killed/interrupted run that got through depth D with no error is only
  **bounded-clean to depth D** — record it as that, never as a pass.

Treat every violation trace as a deliverable: decode it into a narrative
(which actions, which coincidence of message orderings) before deciding
whether it is a spec-fidelity bug or a genuine design bug. Both happen, and
early violations are usually model infidelity — check the spec against the
real code's handler semantics first.

## Checkpointing and resuming

- Checkpoints land in `<metadir>/<timestamp>/`; a usable checkpoint is marked
  by a `<ModuleName>.st.chkpt` file inside it (note: named after the module,
  not `MC`).
- Resume with `-recover <metadir>/<timestamp>` (plus all the original flags).
- A kill mid-checkpoint-write leaves a corrupt checkpoint: on recover TLC
  throws (unexpected exception / missing file / disk-read errors). Detect
  those strings, delete the metadir, and restart fresh.

Self-resuming driver pattern for a matrix of runs (proven robust against
repeated interruptions):

```sh
run_one() {  # <name> <cfg>
  [ -f "done-$1" ] && return           # completed-run marker
  rec=()
  ckpt=$(ls -dt states-$1/*/ 2>/dev/null | head -1 || true)
  [ -n "$ckpt" ] && [ -n "$(ls ${ckpt%/}/*.st.chkpt 2>/dev/null)" ] \
    && rec=(-recover "${ckpt%/}") || rm -rf "states-$1"
  java ... tlc2.TLC ... -metadir "states-$1" "${rec[@]}" -config "$2" Spec.tla \
    > "log-$1.out" 2>&1
  # parse verdict; on verdict: record it, touch "done-$1", rm -rf "states-$1"
  # on corrupt-checkpoint error strings: rm -rf "states-$1" (fresh retry next pass)
}
```

Guard the driver with a pidfile (single instance — two TLCs sharing a metadir
corrupt each other) and a heartbeat line for external monitoring.

## Resource management

- **Disk is the binding constraint on big runs**, not CPU: the state queue for
  multi-billion-state runs reaches 10–20+ GB even gzipped. Monitor free space;
  `No space left` mid-run yields only a bounded-clean verdict.
- Rough sizing: distinct-states rate is roughly constant per config, so
  `target_states / ds_per_min` estimates wall-clock. State counts grow
  explosively with budgets — plan for ~10× per budget increment, and measure
  with a short probe before committing to a long run.
- **Cluster execution**: TLC is single-node/multi-thread — distribute a
  *matrix* of configs across nodes (one TLC per node), not one TLC across
  nodes. Bundle spec + configs + jar + batch script; keep `-metadir` on
  node-local disk (`/tmp/$USER-$JOBID`), give TLC most of the node's RAM, and
  grep the verdict into a small result file at job end. SLURM sketch:

  ```sh
  #SBATCH --exclusive
  SCRATCH=/tmp/$USER-tlc-$SLURM_JOB_ID; mkdir -p $SCRATCH
  java -XX:+UseParallelGC -Xmx100g -Djava.io.tmpdir=$SCRATCH \
       -cp tla2tools.jar tlc2.TLC -deadlock -gzip -workers $SLURM_CPUS_ON_NODE \
       -checkpoint 30 -metadir $SCRATCH/states -config $1.cfg Spec.tla
  ```

- **Sandboxed environments** (e.g. Claude Code's bash sandbox): TLC needs to
  (a) listen on a localhost socket and (b) extract its standard modules to
  `java.io.tmpdir`. Run it with the sandbox disabled; if you must stay
  sandboxed, at minimum set `-Djava.io.tmpdir=$TMPDIR`, but the socket bind
  will still fail — expect to need the escape hatch.
- Long runs: launch detached (`nohup`/sbatch) with output to a log file, and
  poll the log; don't hold a foreground shell open for hours.

## Verification methodology

- **Smoke first, exhaust later.** Tiny budgets (minutes) surface most spec
  bugs and many design bugs. Run exhaustive/large-budget configs only when the
  design has no open questions — an expensive matrix on an unsettled design is
  wasted compute, because any design change invalidates every verdict.
- **Bisection configs prove necessity.** For each mechanism toggle, run a
  config with only that toggle off and everything else on. Expected outcome is
  a *violation* — the trace is machine-checked documentation of why the
  mechanism exists. A bisection that *passes* marks the mechanism as a
  deletion candidate (redundant at that scope; re-test at larger scope before
  deleting).
- **Label every config with its expectation** (pass / violation / open) in the
  run matrix, so an unexpected result is immediately visible. An
  expected-pass job that violates is a finding, top priority to decode.
- **Larger budgets are load-bearing, not just margin.** Minimal
  counterexamples for subtle bugs can need one more message/acquire than the
  base config allows. Keep one big-budget config in the matrix even when the
  base config is exhaustive-clean.
- Any spec change, however small, invalidates in-flight and completed runs of
  the old spec: stop them, re-run the matrix, and re-record. Keep a results
  log (config, expectation, verdict, depth, state count, wall-clock) and a
  findings log (each violation decoded, with its fix) as first-class
  artifacts.

## Common TLA+/TLC gotchas

- **Junction lists are indentation-sensitive.** A multi-line expression (e.g.
  `IF/THEN/ELSE`) continuing under a `/\` bullet must sit strictly to the
  right of the `/\`, or the parser ends the conjunct early — often silently
  changing meaning.
- `/\` binds tighter than `\/`. Parenthesize any mixed conjunction/
  disjunction; never rely on precedence across bullets.
- Every action must determine every variable (use `UNCHANGED <<x, y>>`), or
  TLC reports "successor state is not completely specified".
- `Nat`/`Int` in a variable's type without a budget or `CONSTRAINT` = infinite
  state space; TLC just runs forever (watch for depth growing without the
  queue ever draining).
- Functions vs operators: only functions (`[x \in S |-> e]`) can be values of
  variables; operators cannot.
- TLC evaluates left-to-right and shallowly: guard partial expressions (e.g.
  `x # {} /\ f[CHOOSE ...]`) so the guard runs first.
- `Permutations` (for `SYMMETRY`) needs `EXTENDS TLC`.
