# Index Spaces

## Introduction
Index space is a core data structure offered by Realm to applications
as part of it's public interface. This tutorial introduces the basic 
knowledge necessary to work with index spaces and discusses how to 
manage index spaces in the application code.

Here is a list of covered topics:

* [Creating Index Spaces](#creating-index-spaces)
* [Set Operations](#set-operations)
* [Iterating Over Index Spaces](#iterating-over-index-spaces)
* [Managing Memory In Index Spaces](#managing-memory-in-index-spaces)

## Index Space Basics
An IndexSpace (index space) is a Plain Old Data (POD) structure in
Realm that defines an N-dimensional space with a set of indices
(points). It is templated on two parameters: `IndexSpace<N, T>`, where `N`
is the number of dimensions and `T` is the type of index values.
Index spaces provide an interface to data copies in Realm and offer a
number of set operations, such as union, intersection, difference,
partitioning, and more.

In Realm, two related classes, `Point` and `Rect`, represent single
points and ranges of points, respectively, in an index space. An index
space contains a bounding rectangle and an optional `SparsityMap`
(sparsity map). A sparsity map is the actual dynamically allocated
object that exists on each node that is interested in accessing its
data. A sparsity map is used to indicate which indices are present and
which  ones are missing or "sparse".

## Creating Index Spaces
An index space can be constructed from either a list of points or
rectangles. In this example, all index spaces are constructed from a
single dense rectangle:

```c++
  IndexSpace<2> center_is(Rect<2>(Point<2>(size / 2 - 1, size / 2 - 1),
                                  Point<2>(size / 2 + 1, size / 2 + 1)));
```

## Set operations
An application can perform various set operations on index spaces. In
this example, we demonstrate Realm's set operations that include
`compute_union`, `compute_difference`, `compute_intersection`,
and `compute_coverings`. Most of
these operations utilize Realm's deferred execution model, returning
an event that can be used to query the operation's status and use it
as a pre- or post-condition for subsequent calls.

The tutorial begins by creating a list of disjoint dense index spaces
in two-dimensional space:
```c++
  std::vector<IndexSpace<2>> subspaces;
  for (size_t y = 0; y <= size; y++) {
    subspaces.push_back(
        IndexSpace<2>(Rect<2>(Point<2>(0, y), Point<2>(size, y))));
  }
```

with the following shape:

```
-----------------------------------
(0, 0), (0, 1), (0, 2)...(0, 15)
(1, 0), (1, 1), (1, 2)...(1, 15)
(2, 0), (2, 1), (2, 2)...(2, 15)
...
(15, 0), (15, 1), (15, 2)...(15, 15)
------------------------------------
```

Each index space describes a 1D line along
the x-axis. These index spaces are then passed to `compute_union`,
which constructs an index space with bounds defined by a single
rectangle. This resulting index space is assigned to the variable
`union_is`:

```c++
  IndexSpace<2> union_is;
  Event event1 =
      IndexSpace<2>::compute_union(subspaces, union_is, ProfilingRequestSet());
```

It may be slightly surprising that `union_is` has a sparsity map
initially because Realm defers the computation of the union
(and the grouping into a single rectangle). We can call `tighten`
to make the sparsity map disappear, so the result is just a
dense rectangle:

```
/**
 * Return the tightest description possible of the index space.
 * @param precise false is the sparsity map may be preserved even
 * for dense spaces.
 * @return the tightest index space possible.
 */
IndexSpace<N,T> tighten(bool precise = true) const;
```

Next, we continue by cutting out a small index space from the center 
of `union_is` to obtain a sparse index space containing four disjoint
rectangles. This is done using `compute_difference`, and the resulting
index space is assigned to the variable `diffs_is`:

```c++
  IndexSpace<2> diffs_is;
  Event event2 = IndexSpace<2>::compute_difference(
      union_is, center_is, diffs_is, ProfilingRequestSet(), event1);
```

We then compute the intersection of `diffs_is` and `union_is`,
although this is purely for demonstrative purposes since the
resulting index space, `isect_is`, will be the same as `diffs_is`.

## Iterating Over Index Spaces

Since the result of an intersection is a sparse index space, the
rectangles are stored inside the internal sparsity map.
There are several ways to access the sparsity, such as via an
`IndexSpaceIterator` or using the `compute_covering` interface. The
latter is better suited for building an approximate covering of the
sparsity map when making region instances.

In this example, we use `IndexSpaceIterator` to extract the disjoint
rectangles from the sparsity maps of both `isect_is` and `diffs_is`.
Since both `isect_is` and `diffs_is` should be the same, this iterator
validates that this is indeed the case:

```c++
  IndexSpaceIterator<2> diffs_it(diffs_is), isect_it(isect_is);
  while (diffs_it.valid && isect_it.valid) {
    if (diffs_it.rect != isect_it.rect) {
      log_app.error() << "rects don't match: " << diffs_it.rect
                      << " != " << isect_it.rect;
    }
    diffs_it.step();
    isect_it.step();
  }
  if (diffs_it.valid || isect_it.valid) {
    log_app.error() << "At least one iterator is invalid";
  }
```

## Managing Memory In Index Spaces
As discussed earlier, Realm dynamically allocates an underlying sparsity 
map on the owner node when creating sparse index spaces.
The allocated sparsity maps are being held during the whole application
lifetime. This will likely be fixed in one of the upcoming Realm
releases but users should be aware of this fact.

## References
1. [Index spaces header file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/indexspace.h)
