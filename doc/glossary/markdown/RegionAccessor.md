# RegionAccessor

To access data within a PhysicalRegion, an application must create RegionAccessor objects.
Physical instances can be laid out in many different ways including array-of-struct (AOS), struct-of-array (SOA),
and hybrid formats depending on decisions made as part of the process of mapping a Legion application.
RegionAccessor objects provide the necessary level of indirection to make application code independent of the
selected mapping and therefore correct under all possible mapping decisions.
RegionAccessor objects have their own namespace that must be explicitly included.

RegionAccessor objects are tied directly to the PhysicalRegion for which they are created.
Once the PhysicalRegion is invalidated, either because it is reclaimed or it is explicitly unmapped by the application,
then all accessors for the physical instance are also invalidated and any attempt to re-use them will result in
undefined behavior.
Each RegionAccessor is also associated with a specific field of the physical instance and can be obtained
by invoking the get_field_accessor method on a PhysicalRegion and passing the corresponding FieldID for the desired field.
To aid programmers in writing correct Legion applications, we provide a typeify method to convert from an untyped
RegionAccessor to a typed one.
This allows the C++ compiler to enforce standard typing rules on RegionAccessor operations.

The AccessorType::Generic template argument on the RegionAccessor type specifies the kind of accessor.
In this example we create a specific kind of accessor called a generic accessor.
Generic accessors are the simplest kind of accessors and have the ability to verify many important correctness
properties of region tasks (e.g. abiding by their specified privileges), but they also have the worst performance.
In practice, we often write two variants of every Legion task, one using generic accessors which we use to
validate the application, and second using high-performance accessors.
Generic accessors should NEVER be used in production code.

The generic RegionAccessor provides the read and write methods for accessing data within the region.
These methods are overloaded to either work with ptr_t pointers for LogicalRegions created with
unstructured index spaces, or with non-dimensionalized DomainPoint objects for LogicalRegions
associated with structured index spaces.

Legion pointers do not directly reference data, but instead name an entry in an IndexSpace.
They are used when accessing data within accessors for LogicalRegions.
The accessor is specifically associated with the field being accessed and the pointer names the row entry.
Since pointers are associated with index spaces they can be used with an accessor for physical instance.
In this way Legion pointers are not tied to memory address spaces or physical instances, but instead can be
used to access data for any physical instance of a LogicalRegion created with an index space to which the pointer belongs.

