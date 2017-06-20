# Relaxed Coherence

When a task issues a stream of sub-task launches to Legion, Legion goes about analyzing these sub-tasks for data
dependences based on their region requirements in program order (e.g. the order they were issued to the Legion runtime).
Normally RegionRequirements are annotated with EXCLUSIVE coherence, which tells Legion that if there is a
dependence between two tasks, it must be obeyed in keeping with program order execution.

However, there are often cases where this is too restrictive of a constraint.
In some applications, tasks might have a data dependence, but only need serializability and not
explicit program order execution.
In others, the application might not want Legion to enforce any ordering, and instead will handle
its own synchronization to data in a common logical region.
To support these cases, Legion provides two relaxed coherence modes:
ATOMIC and SIMULTANEOUS.

## ATOMIC coherence allows Legion to re-order tasks as long as access to a particular LogicalRegion
is guaranteed to be serializable.

## SIMULTANEOUS coherence instructs Legion to ignore any data dependences on LogicalRegions with the
guarantee that the application will manage access to the shared logical regions using its own synchronization primitives.
The use of SIMULTANEOUS coherence allows tasks to run in parallel despite data dependences.
