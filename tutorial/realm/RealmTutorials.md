# Draft: Realm Tutorials
This document represents a set of incremental tutorials for Realm.

## Hello World
TODO

## Machine Model
Discuss `Processor` and query interface. Query various processor types, 
setup core reservation (not Reservations), use affinity.

## Region Instances
Discuss `RegionInstance` basics and explore its interface. Understand
AOS (array of structures), SOA (structure of arrays) layouts,
`GenericAccessor`, `AffineAccessor`.
TODO(apryakhin): Perhaps accessors should be placed as a different
tutorial

## Index Spaces
Discuss `IndexSpace` and it's interface, dense and sparse data
constructions.

### Partitioning
Discuss image/preimage.

## Execution Model
Discuss how to work with tasks and events (preconditions,
postconditions..etc)

TODO(apryakhin): Discuss what should be included.

### Deferred allocations
TODO(apryakhin): Determine part of which section it should be.

### Subraphs
TODO(apryakhin): Understand whether that's in fact relevant.

### Copies
Discuss dense, sparse and indirect copies.

### Fill operation
TODO
### Reductions
TODO

## Asynchronous syncrhonization
### Reservations
Provide an example of child sub-tasks updating shared data
structure.
###Barriers
Ghost-cell exchange example (we have barrier reduce.cc test FYI)

## Profiling
Discuss how to profile Realm

## Realm Interop
### CUDA Interop

## Dynamic Code Generation
TODO

## Writing Realm Modules (Extensibility)
TODO

## Case Studies (or more complex examples)
TODO(apryakhin): Determine what those examples are?
