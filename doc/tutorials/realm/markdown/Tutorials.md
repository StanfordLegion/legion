# Draft: Realm Tutorials
This document represents a set of incremental tutorials for Realm.

## Basic Realm Application
TODO

## Data Model
In data model we discuss RegionInstance and explore what can be done
with it. We discuss AOS (array of structures), SOA (structure of
arrays) layouts. We discuss accessors such as GenericAccessor,
AffineAccessor. Portable vs unportable. 

TODO(apryakhin): Explain IndexSpace<N, T>, sparse data contruction (perhaps as more
advanced tutorial)

## Machine Model
Discuss processors, query interface.
Query various processor types, setup core reservation (not
Reservations), use affinity. Consider moving up before Data Model.

## Execution Model
Objective is to show how to work with tasks/events. To showcase a deferred execution
model. To explain how to register tasks, create events and use them as
preconditions, postconditions..etc.

What example should we do?

### Deferred allocations
It's important to show case something more
complex that a unit test? Perhaps some CUDA-graphics interop use
case.

### Copies..dense, sparse copies, gather, scatter
TODO
### Fill operation
TODO
### Reductions
TODO

## Asynchronous syncrhonization
### Reservations
A.k.a. mutexes, give an example of child sub-tasks updating shared data
structure with some analysis. Present fork-join parallelism?
###Barriers
Ghost-cell exchange example (we have barrier_reduce.cc test FYI)

## Profiling
We want to show how to profile Realm.

## Dynamic Code Generation
Task registration

## Writing Realm Modules (Extensibility)
TODO

## Case Studies (or more complex examples)
1. How Omniverse uses Realm? (internal-only)
2. What are the other use cases for Realm?
