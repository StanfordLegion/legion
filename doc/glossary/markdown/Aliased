# Aliased

A partition is aliased if it contains elements that exist in multiple LogicalRegions or subregions.

For example consider a stencil computation with ghost cells.
To be able to compute the stencil computation in parallel, we need to create two partitions:
one disjoint partition with each logical sub-region describing the points that each tasks will write,
and a second partition with each logical sub-region describing the points that each task will need to
read from to perform the stencil computation.
While the first partition will be disjoint, the second partition will be aliased because each sub-region will
additionally require two ghost cells on each side of the set of elements in each sub-region in the disjoint partition.
The need for these ghost cells means that some cells will exist in multiple sub-regions and therefore the partition
will be aliased.
