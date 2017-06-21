# PhaseBarrier

When using SIMULTANEOUS coherence it is up to the application to properly synchronize access to a LogicalRegion.
While applications are free to construct their own synchronization primitives, Legion also provides two useful
synchronization primitives: reservations and PhaseBarriers.

## Reservations provide an atomic synchronization primitive similar to locks,
but capable of operating in a deferred execution environment.

## PhaseBarriers provide a producer-consumer synchronization mechanism that allow a set of producer
operations to notify a set of consumer operations when data is ready.

While both of these operations can be used directly, the common convention in Legion programs is to specify
on launcher objects which reservations should be acquired/released and which PhaseBarriers need to be
waited on or triggered before and after an operation is executed.

First, it is very important to realize that PhaseBarriers are in no way related to traditional barriers in
SPMD programming models such as MPI.
Instead, PhaseBarriers are a very light-weight producer-consumer synchronization mechanism.
In some ways they are similar to phasers in X10 and named barriers in GPU computing.
PhaseBarriers allow a dynamic number of consumers (possibly from different tasks) to be registered.
Once all of these producers have finished running the generation of the barrier will be advanced.
Consumers of the PhaseBarrier wait on a particular generation.
Only once the generation has been reached will the consumers be allowed to execute.

When a PhaseBarrier is created, it must be told how many possible tasks will be registering producers
and/or consumers with it, but the exact number of producers and producers can be dynamically determined.
The number of tasks which may be registering producers or consumers is called the participants count.
When it is executing, each participant task can launch as sub-operations which either arrive or wait
on a specific PhaseBarrier as it wants.
Once it is done launching sub-operations that use a specific generation of the PhaseBarrier, it then calls
advance_phase_barrier to get the name of the PhaseBarrier corresponding to the next generation.
PhaseBarriers remain valid indefinitely (or until they exhaust the maximum number of generations,
usually 2^32) unless they are explicitly deleted by the application.
