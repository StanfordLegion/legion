# IndexAllocator

Unstructured index spaces have no points allocated, but can dynamically allocate and free points
using IndexAllocator objects.
IndexAllocator objects must be obtained from the runtime using the create_index_allocator method.
Allocators can be used to allocate or free elements.
They return ptr_t elements which are Legion’s untyped pointer type.
Pointers in a Legion are opaque and have no data associated with them.
Instead they are used for naming the rows in LogicalRegions which are created using the corresponding IndexSpace.
