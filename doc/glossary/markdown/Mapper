# Mapper

Legion makes no implicit decisions concerning how applications are mapped onto target hardware.
Instead mapping decisions regarding how tasks are assigned to processors and how physical instances of LogicalRegions
are assigned to memories are made entirely by mappers.
Mappers are part of application code and implement a mapping interface.
Mappers are queried by the Legion runtime whenever any mapping decision needs to be made.
Legion guarantees that mapping decisions only impact performance and are orthogonal to correctness which
simplifies tuning of Legion applications and enables easy porting to different architectures.

## Most mapper calls have three arguments
### Reference to the operation (task, copy, inline mapping, etc)
### Input argument struct
### Output argument struct
