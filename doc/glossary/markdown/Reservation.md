# Reservation

When using SIMULTANEOUS coherence it is up to the application to properly synchronize access to a LogicalRegion.
While applications are free to construct their own synchronization primitives, Legion also provides two useful
synchronization primitives: reservations and PhaseBarriers.

## Reservations provide an atomic synchronization primitive similar to locks,
but capable of operating in a deferred execution environment.

## PhaseBarriers provide a producer-consumer synchronization mechanism that allow a set of producer
operations to notify a set of consumer operations when data is ready.
