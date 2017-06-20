# Region Tree

LogicalRegions in Legion are created by taking the cross product of a FieldSpace with an IndexSpace.
Each invocation of this cross product generates a new LogicalRegion.
LogicalRegions created in this way can also be partitioned into logical sub-regions.
We refer to a LogicalRegion and all of its logical sub-regions as a region tree.
Since each cross-product of a FieldSpace with an IndexSpace results in a new LogicalRegion,
we assign to each region tree a tree ID.
Therefore, each LogicalRegion can be uniquely identified by a 3-tuple consisting of an IndexSpace,
a FieldSpace, and a tree ID.
