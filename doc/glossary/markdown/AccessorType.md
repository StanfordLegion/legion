# AccessorType

While the AccessorType::Generic region accessor is general purpose and works for all physical instances,
it is very slow.
Most Legion applications are performance sensitive and therefore need fast access to the data contained within
physical instances (by fast we mean as fast as C pointer dereferences).
To achieve this, we provide specialized accessors for specific data layouts.
These accessors are templated so that significant amounts of constant folding occurs at compile-time,
resulting in memory accesses that are effectively C pointer dereferences.

## SOA - struct-of-arrays accessors
The accessor for struct-of-arrays physical instances is AccessorType::SOA.
This type is templated on the size of the field being accessed in bytes.
The template value can also be instantiated with 0, but this will cause the accessor to fall back
to using a dynamically computed field size which will not be as fast C pointer dereferences (but will always be correct).
For each of our accessors, we first call the can_convert method on the generic accessor to confirm that we can
convert to new accessor type.
If any of them fails, then we return false.
If they all succeed, then we can invoke convert method to get specialized SOA accessors.

In addition to SOA accessors, there are several other specialized accessors:

## AOS - array-of-struct accessors
## HybridSOA - for handling layouts that interleave multiple elements for different fields (still in progress, inspired by the ISPC compiler
## ReductionFold - for reduction instances with an explicit fold operation
## ReductionList - for reduction instances without an explicit fold operation

