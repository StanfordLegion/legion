---
layout: page
permalink: /tutorial/realm/reductions.html
title: Realm Reductions
---

## Introduction
Reductions are an important concept in parallel programming. 
They combine multiple values into a single value,
typically to summarize or aggregate data.

In this tutorial, we will walk through an example of using reductions in
Realm. We will begin by discussing the basics of reductions, then move
on to the example code, which demonstrates how to perform reductions
through the copy interface provided by index spaces.

Here is a list of covered topics:

* [Basics of Reductions](#basics-of-reductions)
* [Registering Reduction Operators](#registering-reduction-operators)
* [Performing Reduction Copies](#performing-reduction-copies)
* [What is Next](#what-is-next)

## Basics of Reductions
Reductions are used to combine multiple values into a single value.
For example, if we have an array of numbers, we might want to compute
their sum. We could do this by summing the elements of the array in
parallel, then combining the results using a reduction operation.

Reductions can be performed on various data structures,
including arrays, matrices, and trees.
In each case, the reduction operation combines multiple values into a 
single value. The operation must be associative, which means that the
order in which the values are combined does not matter. For example,
addition is associative, but subtraction is not.

## Registering Reduction Operators

Realm provides reductions through operators such as `fold` and
`apply`. The runtime requires that every reduction operation follows
this convection:

```c++
#ifdef NOT_REALLY_CODE
    class MyReductionOp {
    public:
      typedef int LHS;
      typedef int RHS;

      void apply(LHS& lhs, RHS rhs) const;

      // both of these are optional
      static const RHS identity;
      void fold(RHS& rhs1, RHS rhs2) const;
    };
#endif
```
Reductions have two types associated with them `left-hand-side (LHS)`
and `right-hand-side (RHS)` where `apply` combines lhs and rhs into a
new lhs type and `fold` combines two rhs types into a new rhs type.
In general, for sum reductions, lhs and rhs can be the same
type, but it is not always the case.

In Realm, reductions are performed via the copy interface provided
by index spaces, and each reduction operator needs to be registered with
the runtime first:

```c++
rt.register_reduction<SumReduction>(REDOP_SUM);
```

The `REDOP_SUM` id represents an exclusive sum reduction:
(TODO: Consider adding non-exclusive)

```c++
class SumReduction {
 public:
  using LHS = ComplexType;
  using RHS = int;

  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs) {
    assert(EXCLUSIVE);
    lhs.value += rhs;
  }

  static const RHS identity;
  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE)
      rhs1 += rhs2;
    else {
    __sync_fetch_and_add(&rhs1, rhs2);
    }
  }
};
```

## Performing Reduction Copies

In the example given, the program performs four reduction copies.
Specifically, there are two non-exclusive fold reductions over the 
same instance with an integer type, and two exclusive apply reductions
over the same instance with a left-hand-side `StructType`.

The sequence looks as following:

```c++
1. fold<NON-EXCL>(int, int) && fold<NON-EXCL>(int, int)
2. apply<EXCL>(StructType, int) && apply<EXCL>(StructType, int)
```

To execute the reduction copies, the program prepares and issues a
realm reduction copy by setting the corresponding `redop` field and
calling `domain.copy(...)`:

```c++
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src_inst, src_fid, src_fsize);
  dsts[0].set_field(dst_inst, dst_fid, dst_fsize);
  dsts[0].set_redop(REDOP_SUM, fold, exclusive);
  return domain.copy(srcs, dsts, ProfilingRequestSet(), wait_on);
```

Non-exclusive reductions can be
handled with atomics inside the reduction operator. In contrast,
exclusive reductions do not require atomic operations since the
application explicitly tells Realm that no race situation will appear.
However, because the program is issuing several reduce-copies over
the same instance/field, it still needs to ensure some atomicity.

One option for ensuring atomicity is to pre-define the execution order
of reduce-copies, but this may not be optimal as the order of
completion may be uncertain. Another option is to use a
non-exclusive reduction, which handles synchronization with atomics
inside the reduction operator, but this may be too fine-grained and
inefficient. Instead, the program uses exclusive reduction with
reservations to guarantee the atomicity of reduce-copies at a coarse
level, allowing them to run in either order.

```c++
  // Run exclusive applies with reservations.
  Reservation resrv = Reservation::create_reservation();

  Event resrv_event1 = resrv.acquire(0, true, fold_event);
  Event apply_event1 =
      reduce<2, T, StructType>(dst_inst0, dst_inst1, domain,
                               /*src_fid=*/FID_INT,
                               /*dst_fid=*/FID_COMPLEX_TYPE,
                               /*src_fsize=*/sizeof(int),
                               /*dst_fsize=*/sizeof(StructType),
                               /*exclusive=*/true,
                               /*fold=*/false, resrv_event1);
  resrv.release(apply_event1);

  Event resrv_event2 = resrv.acquire(0, true, fold_event);
  Event apply_event2 =
      reduce<2, T, StructType>(dst_inst0, dst_inst1, domain,
                               /*src_fid=*/FID_INT,
                               /*dst_fid=*/FID_COMPLEX_TYPE,
                               /*src_fsize=*/sizeof(int),
                               /*dst_fsize=*/sizeof(StructType),
                               /*exclusive=*/true,
                               /*fold=*/false, resrv_event2);
  resrv.release(apply_event2);

```

For details on reservations, please refer to a dedicated tutorial.

## What is Next

The sum reduction is just one example of what can be done with
reductions, but many other options are also available.
In addition to sum, reductions can also be performed using operations
such as and, or, xor, prod, diff, min, and max. Furthermore, Realm
allows performing reductions through barriers which is covered in the
next tutorial.
