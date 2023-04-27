# Region Instances

## Introduction
This tutorial will cover how to manage application data in
Realm. Specifically, we will show how to allocate a blob of data in
system memory and create writer and reader tasks to access this data.

Here is a list of covered topics:

* [Data Layouts](#data-layouts)
* [Creating Instances](#creating-instances)
* [Accessing Instances](#accessing-intances)
* [Best Practices](#best-practices)
* [References](#references)

## Data Layouts
Before diving into the specifics of using Realm for data management,
let us review some commonly used definitions in high-performance
computing applications:

1. `Affine data layout`: This type of layout involves mapping
multi-dimensional data structures to a linear memory space while
preserving the locality of reference. In an affine layout, the memory
address of an element is calculated as a linear function of its
multi-dimensional index.

2. Array-of-structures (AOS): This data layout consists of an array
composed of a structure that contains multiple fields. Elements of
the array are stored sequentially in memory, with all the fields for
an element stored together. AOS is often used when the fields of the
data structure have varying sizes and types.

3. Structure-of-arrays (SOA): This data layout stores each field of a
data structure in a separate array. The elements of the array are
stored in a column-wise fashion, with all the elements for a
particular field stored together. SOA is often used when the fields
of the data structure have fixed sizes and types, and when the data
access pattern is column-wise.

## Creating Instances
Realm uses `RegionInstances` to store persistent application data.
Region instances offer a comprehensive interface that allows defining
various data layouts and to accessing them efficiently. Region instances
are instantiated using the static member function
`create_instance`:

```c++
static Event create_instance(RegionInstance& inst, Memory memory,
                             InstanceLayoutGeneric* ilg,
                             const ProfilingRequestSet& prs,
                             Event wait_on = Event::NO_EVENT);
```

, which takes as input a `Memory` and an `InstanceLayoutGeneric` object
containing information necessary to create a data layout. The operation
of instance creation, including underlying allocation, is asynchronous
and returns a runtime event, which can serve as a precondition for any
subsequent operation.

It should be noted that region instances are always associated with a
particular memory and cannot be moved; thus, data migration should only
be accomplished through data copy operations in Realm.

The InstanceLayoutGeneric object provides a powerful interface to
create any data layout commonly used in HPC applications, including
AOS, SOA, hybrid layouts, compact layouts, and more complex layouts
that involve interleaving fields with dimensions or non-trivial tiling.

In this example, an instance is created using the `create_instance` function
with a standard AOS layout:

```c++
  Event create_event = RegionInstance::create_instance(
      task_args.inst, *memories.begin(), task_args.bounds, field_sizes,
      /*AOS=*/1, ProfilingRequestSet());
```

This instance encompasses two logical layouts such as `InstanceLogicalLayout1`
and `InstanceLogicalLayout2`. These layouts are attributed by the `FieldID`
and supplied to the instance interface as part of the `field_sizes` map:

```c++
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID1] = sizeof(InstanceLogicalLayout1);
  field_sizes[FID2] = sizeof(InstanceLogicalLayout2);
```

## Accessing Instances
It may be necessary for a task to access data stored in a
region instance directly. Realm offers a set of accessors such as
`AffineAccessor`, `MultiAffineAccessor` and `GenericAccessor` that
allow users to access individual elements. GenericAccessor can handle
any data layout irrespectively whether the data is stored on the local or
remote node. AffineAccessor is designed to work for local data
with an affine layout only that is scoped to a single layout piece.
MultiAffineAccessor allows handling affine layouts with multiple
pieces.

In this example, the `main_task` creates a region instance with an
affine layout and stores it in the system memory (`SYSTEM_MEM`)
accessible by the executing processor `p`. The reader and writer tasks 
are launched on the same processor as the `main_task` which
allowing them to use the `AffineAccessor` to access individual
elements of an instance provided within `TaskArguments`:

```c++
void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  verify<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  verify<InstanceLogicalLayout2, long long, double>(task_args.inst,
                                                    task_args.bounds, FID2,
                                                    /*add=*/2);
}

void writer_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  update<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  update<InstanceLogicalLayout2, long long, double>(
      task_args.inst, task_args.bounds, FID2, /*add=*/2);
}
```

The example demonstrates the control dependency between the writer and
reader tasks, where the writer task waits for the instance
completion event (`create_event`), and the reader task waits for the 
writer to complete:

```c++
  Event writer_event =
      p.spawn(WRITER_TASK, &task_args, sizeof(TaskArguments), create_event);
  Event reader_event =
      p.spawn(READER_TASK, &task_args, sizeof(TaskArguments), writer_event);
```

All operations, such as creating an instance and writing and reading data, are
also non-blocking which is achieved through realm's asynchronous execution
model.

## Best Practices
The affine data layout is the most commonly used in high-performance
computing applications and is recommended for Realm applications as
well. It allows for the efficient traversal of data in both row-major and
column-major order.

SOA is generally more efficient when accessing a specific
field of data for a large number of objects, as each field is
stored contiguously in memory. This can lead to better data locality
and cache performance, especially when processing large amounts of 
data.

AOS, on the other hand, is more convenient when accessing
multiple fields of a single object, as all fields for each object are
stored together in memory. It can simplify the code needed to access
the data and can also be more memory efficient for small datasets.

GenericAccessors.  While useful, these accessors are generally more
expensive and should be used with caution. A read/write operation via
this accessor could potentially result in a network transaction,
assuming the data is stored on a remote node.

Accessors use a k-d tree to perform piece lookups that are
logarithmic in the number of pieces. If we build an accessor for just
a subset of an instance, it will pre-prune the tree to just the subtree
that covers the specified subset (assuming this would be a single
piece, lookups are now constant time again).

## References

1. [Instance public header file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/instance.h)
