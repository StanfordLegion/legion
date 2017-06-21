# LogicalPartition

The act of partitioning in Legion breaks a set of points represented by an IndexSpace into subsets of points,
each of which will become index sub-spaces.
A LogicalPartition is constructed by applying an IndexSpacePartition to a LogicalRegion.
