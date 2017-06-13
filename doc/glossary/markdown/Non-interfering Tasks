# Non-interfering Tasks

Determining that two sub-tasks have non-interfering RegionRequirement objects is how Legion
implicitly extracts parallelism from applications.
There are three forms of non-interference:

## Region non-interference: two RegionRequirement objects are non-interfering on regions
if they access LogicaRegions from different region trees, or disjoint LogicalRegions in the same tree.
## Field-level non-interference: two RegionRequirement objects are non-interfering on fields if they access
disjoint sets of fields within the same LogicalRegion.
## Privileges non-interference: two RegionRequirement objects are non-interfering on privileges if they 
both request READ_ONLY privileges, or they both request REDUCE privileges with the same reduction operator.
