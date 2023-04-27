# Realm Copies

* [Introduction](#introduction)
* [DMA Sytem](#dma-system)
* [Structured And Unstructured
  Copies](#structured-and-unstructured-copies)
* [Filling Region Instances](#filling-region-instances)
* [Copying Region Instances](#copying-region-instances)
* [References](#references)

## Introduction
In this example, we will discuss how to perform operations such
as filling and copying data stored in region instances while
exploring the features of index spaces.

## DMA System
In Realm, data is stored inside region instances. The region instances
cannot be moved; therefore, the only way to migrate data is to
perform a copy operation. Data copies are executed by the `DMA` system
in Realm, which also manages the execution of reductions and fill
operations. The DMA channels provided by modules handle the actual
movement of data between storage locations in the memory hierarchy.
When a new copy request is made, the Realm process retrieves
information about the instances involved from the nodes on which they
were created. These requests are performed concurrently while waiting
for the transfer's precondition to be met.

## Structured And Unstructured Copies
Realm allows users to copy structured and
unstructured data, which can also be referred to as structured and
unstructured copies.

A structured copy refers to an operation over data organized
in a predictable, uniform pattern. Most of the time, the copy domain can
be defined by a dense index space with a single bounding rectangle. For
example, `src_index_space` and `dst_index_space` in this example have
the following structure:
```
----------------------------------------------------------------------
| P0:V0, P1:V1, P2:V2, P3:V3, P4:V4, P5:V5, P6:V6, P7:V7...PN-1:VN-1 |
----------------------------------------------------------------------
```

On the other hand, an unstructured copy refers to an operation over
data organized in a more flexible manner, with each
element potentially located at a different address in memory. An example
would be an array of indirect indices used
to copy elements from source to destination or vice versa.
In Realm, an unstructured copy operation is also referred to as an indirect
copy or gather/scatter, which will be covered in detail in the next
tutorial.

In the following example, we create two dense one-dimensional(1D) index spaces
and use them to create two region instances: `inst1` and `inst2`.
Both instances have an affine SOA layout:

```c++
RegionInstance inst1;
Event ev1 = RegionInstance::create_instance(
    inst1, memories[mem_idx++ % memories.size()], src_index_space, field_sizes,
    /*block_size=*/0, ProfilingRequestSet(), user_event);

RegionInstance inst2;
Event ev2 = RegionInstance::create_instance(
    inst2, memories[mem_idx++ % memories.size()], dst_index_space,
    field_sizes, /*block_size=*/0, ProfilingRequestSet(), user_event);
```

## Filling Region Instances
Index spaces provide a way to populate region instances with specific
values. We fill both `inst1` and `inst2` as shown below:

```c++
std::vector<CopySrcDstField> dsts(1);
dsts[0].set_field(inst, FID_BASE, sizeof(int));
return is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value),
               wait_on);
```
This code sets up a vector of `CopySrcDstField` objects, specifying
the destination region instance (`inst1` or `inst2`), the field
ID (`FID_BASE`), and the size of the field data (`sizeof(int)`).
Like most operations in Realm, the fill operation is asynchronous
and returns an event that can be waited upon.

## Copying Region Instances
The index spaces that describe the source and destination instances
do not necessarily have to match in size, so we need to compute an
intersection between them:

```c++
  IndexSpace<1> isect;
  Event isect_event = IndexSpace<1>::compute_intersection(
      src_index_space, dst_index_space, isect, ProfilingRequestSet(),
      fill_event);
```

This function is one of the set operations provided by
the interface discussed in the previous tutorial. Once we have
computed the intersection `isect`, we use it to perform the copy
itself. To specify the source and destination instances
and their corresponding fields, we use the `CopySrcDstField` class:

```c++
std::vector<CopySrcDstField> srcs(1), dsts(1);
srcs[0].set_field(inst1, FID_BASE, sizeof(int));
dsts[0].set_field(inst2, FID_BASE, sizeof(int));
Event ev3 = index_space.copy(srcs, dsts, ProfilingRequestSet(), fill_event);
```
Realm provides a set of iterators to access the underlying rectangles 
and points in index spaces programmatically. It is helpful for
fine-grained point access to the data stored in a region instance.

In the following code, we iterate over the intersection `isect`, get
a set of points, use them to read and verify actual values from `inst2`.
Even though the instance contains just a simple one-piece affine
layout, we use `GenericAccessor` to pull the values out of the
instance for `FIB_BASE` in case the memory is allocated on a remote
node:

```c++
GenericAccessor<int, 1, int> acc(inst2, FID_BASE);
for (IndexSpaceIterator<1, int> it(isect); it.valid; it.step()) {
  for (PointInRectIterator<1, int> it2(it.rect); it2.valid; it2.step()) {}
}
```

## References
1. [Indexspace header
file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/indexspace.h)
