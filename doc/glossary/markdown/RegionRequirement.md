# RegionRequirement

Region requirements are the objects used to name the LogicalRegions that are used by tasks, copies,
and inline mapping operations.
RegionRequirements can name either LogicalRegions, or LogicalPartitions for index space task launches.
In addition to placing logical upper bounds on the privileges required for an operation,
RegionRequirements also specify the privileges and coherence modes associated with the needed LogicalRegion/Partition.
RegionRequirements have a series of constructors for different scenarios.
All fields in RegionRequirements are publicly visible so applications can mutate them freely including configuring
RegionRequirements in ways not supported with the default set of constructors.
