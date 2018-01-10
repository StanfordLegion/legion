# Task Dataflow Graph

As the top-level task executes, Legion computes a Task Dataflow Graph (TDG) which captures data dependences
between operations based on the LogicalRegions, fields, and privileges that operations request.
Dependences in legion are computed based on the program order in which operations are issued to the runtime.
Legion therefore maintains sequential execution semantics for all operations, even though operations may
ultimately execute in parallel.
By maintaining sequential execution semantics, Legion significantly simplifies reasoning about operations within a task.
