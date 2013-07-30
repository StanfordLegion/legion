
% A Tutorial for the Legion Language

<!--
Copyright 2013 Stanford University and Los Alamos National Security, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Legion Overview

## Premise

Writing high-performance code on modern hardware is hard.

With the emergence of constraints on scaling within a single core,
hardware designers have turned to parallelism as a solution. However,
while current programming models provide many ways to describe
parallelism in an application, most are blind to the structure of
data. Programmers are forced to shepherd data as it moves around the
machine, resulting in over-specified and error-prone code. These codes
incur substantial maintenance costs as they age, as the old data paths
may not be optimal for new generations of hardware.

Previous approaches to this problem (e.g., Sequoia, DPJ) have required
programmers to provide a single, static decomposition of the
data. This does not work for application domains where the shape of
the data is not known until runtime. For example, if an application
operates on an irregular graph, the optimal partitioning of the data
might depend on the input to the program, and might even change while
the program is running.

## Programming Model

Legion is a new parallel programming model that addresses these
concerns. Specifically:

  * Legion captures parallelism which is difficult to describe in
    traditional programming models.

      * The Legion model encourages programmers to expose all levels
        of parallelism in an application. Legion can exploit different
        types of parallelism in different ways on different types of
        hardware.

  * Legion understands the structure and movement of data.

      * Legion is able to reason about the movement of data through
        the machine, can detect cases where a remote computation needs
        data, and can copy that data automatically.

  * Legion runs on a wide variety of machines: multicore CPUs, GPUs,
    accelerators (including Xeon Phi), and clusters of machines (where
    each machine may possibly have a different configuration).

      * Legion programs are guaranteed to run correctly on new
        hardware with zero effort. Tuning an application for optimal
        performance on new hardware typically requires programmer
        effort.

  * Legion is *not magic*.

      * In general, tuning an application for performance is both
        application and machine-specific. As such, performance tuning
        is (and must be) the job of the programmer, not the job of
        Legion. It is expected that performance tuning will require
        programmer effort on each new machine.

      * Legion provides a *default mapper* with heuristics for
        reasonable performance on many machines. The default mapper is
        intended to be a zero-effort solution for the initial
        development phase of an application, when the focus is on
        correctness rather than performance.

      * Once a correct program is available, Legion provides a
        *mapping interface* for the programmer to specify, in as much
        detail as necessary, how to map an application onto a specific
        machine. These settings *preserve correctness*, so a program
        that is correct in one configuration will always run correctly
        everywhere, no matter what choices the mapper makes.

## Language versus Runtime

Legion is implemented in two parts: a runtime library written in C++,
and a programming language which is compiled into C++ source code.

The C++ runtime provides the lowest level of control to Legion
programmers, with all the knobs exposed to provide maximum performance
potential. However, programming to the runtime interface also tends to
be error prone, as C++ is unable to verify that the client code is
using the Legion runtime correctly. In addition, code targeting the
Legion runtime tends to be verbose, making it difficult to read and
write code in this manner. As such, programming directly to the
runtime is recommended primarily for expert users with intimate
knowledge of Legion semantics.

The Legion language is an alternative which provides expressiveness
and safety not available in code targeting the Legion runtime. Legion
programs are typically more concise than the corresponding C++
programs, and are anecdotally easier to read and write. The Legion
compiler checks many safety properties of Legion programs, resulting
in fewer errors slipping through to runtime. This aspect of Legion is
particularly nice, because debugging parallel distributed programs
tends to be a difficult and frustrating process. The Legion compiler
then translates Legion programs into C++ source code, which can be
turned into a standalone executable, or linked into a pre-existing
application.

The remainder of this document introduces the Legion language. For
information on the Legion runtime, refer the runtime documentation
(forthcoming).

# Quickstart Guide

The instructions below describe how to install and run the compiler on
Ubuntu and Mac OS X. For more detailed instructions or for other
operating systems, refer to `legion/compiler/README.md`.

## Instructions for Ubuntu

Open a terminal and run the following commands to install a C++
compiler, Git, and PLY:

```bash
sudo apt-get install build-essential git python-pip
sudo pip install ply
```

Checkout a copy of Legion:

```bash
git clone https://github.com/StanfordLegion/legion.git
cd legion/compiler
```

Create a file named `hello.lg` with the following contents:

``` {.legion #section1}
task main()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 0;
  *p = *p + 1;
  assert *p == 1;
}
```

Now run the compiler:

```bash
export LG_RT_DIR="$PWD/../runtime"
./lcomp hello.lg -o hello
./hello
```

With these instructions, the compiler will be unable to use C/C++ code
from Legion. See `legion/compiler/README.md` for installing the Clang
bindings for Python.

## Instructions for Mac OS X

Install Xcode from the Mac OS X App Store. Once the installation is
complete, open Xcode and go to the menu Xcode > Preferences >
Downloads to install the Command Line Tools.

Open a terminal and run the following command to install PLY:

```bash
sudo easy_install pip
sudo pip install ply
```

Checkout a copy of Legion:

```bash
git clone https://github.com/StanfordLegion/legion.git
cd legion/compiler
```

Create a file named `hello.lg` with the following contents:

``` {.legion #section2}
task main()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 0;
  *p = *p + 1;
  assert *p == 1;
}
```

Now run the compiler:

```bash
export LG_RT_DIR="$PWD/../runtime"
./lcomp hello.lg -o hello
./hello
```

# Language Tutorial

In this tutorial, we introduce the Legion language from the
perspective of a user. Experience with at least one compiled
programming language is recommended (e.g., C, C++, Java, ML, Haskell,
etc.). We focus on the non-standard features of Legion (e.g.,
regions).

## Tasks

Legion programs are composed of *tasks*.

```legion
task add1(x: int): int
{
  return x + 1;
}

task add2(x: int): int
{
  return add1(add1(x));
}
```

Tasks in Legion resemble functions in most C-family languages. For
now, think of tasks as functions. The differences between tasks and
functions are explained in Section 2.7, below.

Parameters to tasks are passed by-value and are lexically
scoped. There is no type inference for task declarations, although
there is a limited form of `auto`-style type inference available for
local variables (see below).

Task calls work like you'd expect. Parentheses are *not* optional.

## Variables

Local variables are introduced with the `let` and `var` keywords.

```legion
task local_variables()
{
  let x: int = 5;
  var y = 8;      // y is type int
  y = x + 7;
}
```

Variables created with `var` are mutable. Those created with `let` are
immutable, similar to `final` variables in Java.

Limited type inference is available, so you can leave out the type in
the variable declaration. The initializer for the variable is *not*
optional.

## Regions

Whereas local variables hold data that is local to a task, *regions*
hold data that is shared between tasks. Regions are similar to arrays
in other languages in that both hold many elements of the same
type. However, regions are different from arrays, because the elements
of region have no specific layout (i.e., no array indexing) and no
specific location in the memory of the machine (i.e., because regions
are not placed in memories until runtime, and may even more between
memories dynamically).

Regions are declared with the type of data they hold and the maximum
number of elements that will be allocated.

```legion
task a_region()
{
  let r = region<int>(1);
  let p = new<int@r>();
}
```

Regions are initially created empty. The `new` operator allocates an
element in the region and returns a pointer to that element. Pointers
are described in more detail below.

## Pointers

Legion pointers have some differences from C pointers. Here is an
example:

```legion
task a_pointer()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 0;
  *p = *p + 111;
  assert *p == 111;
}
```

A pointer in Legion includes the region that the pointer points to as
part of its type. That is, `int@r` is a pointer that points to an
`int` inside the region `r`. This property gives the compiler a handle
on the *aliasing problem*. Suppose we know two regions are
disjoint. Then we also know that any pointers into those two regions
are also disjoint (i.e., they do not alias).

There is no pointer arithmetic in Legion. You can't increment the
pointer to get the next value, nor can you obtain a pointer to the
interior of a struct (but more on that later).

Legion does have null pointers. Here's an example.

```legion
task a_null_pointer()
{
  let r = region<int>(1);
  let p = null<int@r>();
  assert isnull(p);
}
```

Note that even null pointers explicitly point into a region.

Dereferencing a null pointer is a runtime error, but this error is not
checked for and will likely result in a program crash.

## Privileges

When dereferencing a pointer in Legion, *privileges* are required to
use the dereferenced value (i.e., to *read*, *write*, or *reduce* to
it). The privileges in the examples above were implicit, because a
task that creates a region always has full privileges for it. When
passing a region to another task, the called task makes explicit what
privileges on the region it requires.

```legion
task takes_a_region(r: region<int>, p: int@r): int
  , reads(r)
{
  return *p;
}

task passes_a_region()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 42;
  let x = takes_a_region(r, p);
  assert x == 42;
}
```

The `reads(r)` line after `task takes_a_region` declares that
`takes_a_region` uses the region `r` with read-only privileges.

Privileges are checked when dereferencing pointers. The example above
would fail to compile if you removed the `reads(r)` privilege on
`takes_a_region`, because the value of `*p` is read to produce the
return value of `takes_a_region`.

Calls to other tasks must always take a subset of the privileges held
by the caller. For example, in the following code, it is ok for
`read_write_region` to call `read_only_region`, but not the other way
around.

```legion
task read_only_region(r: region<int>, p: int@r): int
  , reads(r)
{
  return *p;
}

task read_write_region(r: region<int>, p: int@r): int
  , reads(r)
  , writes(r)
{
  let x = read_only_region(r, p);
  *p = *p + x;
  return x;
}
```

## Partitions

Regions can be *partitioned* into *disjoint subregions*. Partitions
allow the programmer to specify that two tasks operate on independent
data and therefore should run in parallel. More details on parallelism
in Legion are provided in Section 2.7, below.

Partitions can be constructed in two ways:

  * *partition-then-allocate*: the partition can be created first,
    empty, and data can be allocated inside the subregions.

  * *allocate-then-partition*: the data can be allocated first, inside
    the parent region, and the partition can be created afterwards.

```legion
task a_subtask(r: region<int>, p: int@r), reads(r), writes(r) {}

task partition_then_allocate()
{
  let r = region<int>(2);
  let p = partition<r, disjoint>(coloring<r>());
  let r0 = p[0];
  let r1 = p[1];
  let x = new<int@r0>();
  let y = new<int@r1>();
  a_subtask(r0, x);
  a_subtask(r1, y);
}

task allocate_then_partition()
{
  let r = region<int>(2);
  let x = new<int@r>();
  let y = new<int@r>();
  let p = partition<r, disjoint>(color(color(coloring<r>(), x, 0), y, 1));
  let r0 = p[0];
  let r1 = p[1];
  a_subtask(r0, downregion<r0>(x));
  a_subtask(r1, downregion<r1>(y));
}
```

With the allocate-then-partition method, a non-trivial *coloring* must
be provided which maps pointers to subregions in the partition. The
built-in `coloring` operator creates an empty coloring, and `color`
maps a pointer to a color in the returned coloring.

In addition, work is required to prove to the compiler that pointers
`p` and `q` point into subregions `r1` and `r2` respectively. This is
done with the built-in `downregion` operator, which casts a pointer
into a region into a pointer into one of its subregions (similar to a
type downcast in an object-oriented language like C++ or Java). Legion
inserts dynamic tests to verify that the downcast region is valid at
runtime.

## Implicit Parallelism

Consider the following naive implementation of a Fibonacci number
calculation.

```legion
task fib(x: int): int
{
  if (x < 2) {
    return 1;
  } else {
    return fib(x - 1) + fib(x - 2);
  }
}
```

While the code above is wildly inefficient, it does have one
interesting property: the two recursive calls to `fib` can execute in
parallel.

Legion programs define an explicit *serial execution order*. This is
how the program would run if you executed it by hand, starting at
`main`, from top to bottom, line by line. In other words, Legion
programs can be thought as of executing as if they were written in a
more traditional serial programming language, like C.

Behind the scenes, the Legion runtime may decide that it is possible
to execute parts of the program in parallel. Such code is said to be
*implicitly parallel*. However, even in the presence of parallelism,
Legion will always guarantee that the results produced by the program
are identical to the serial execution order. Parallelism *cannot*
impact correctness in Legion programs.

The units of (potential) parallel execution in Legion are tasks. Any
task can potentially run in parallel with other tasks, as long as
those tasks don't interfere with each other.

This is a key difference between Legion tasks and functions in other
languages. Functions in most languages do not run in parallel, at
least not automatically. In Legion, parallelism with tasks is
automatic, implicit, and guaranteed to be safe.

Legion uses regions and privileges to determine when two tasks can run
in parallel. Two tasks that use no regions can always run in parallel,
because parameters and variables cannot be shared between tasks. Two
tasks that use the same region can run in parallel only if their
privileges are compatible.

Consider the following program. Can `expensive1a` and `expensive1b`
run in parallel?

```legion
task expensive1a(r: region<int>, p: int@r): int, reads(r) {/*...*/}

task expensive1b(r: region<int>, p: int@r): int, reads(r) {/*...*/}

task am_i_parallel1(r: region<int>, p: int@r): int
  , reads(r), writes(r)
{
  let x = expensive1a(r, p);
  let y = expensive1b(r, p);
  return x + y;
}
```

In this program, `expensive1a` and `expensive1b` *can* run in
parallel, because both use region `r` with read-only privileges.

What about the following program?

```legion
task expensive2a(r: region<int>, p: int@r): int, reads(r) {/*...*/}

task expensive2b(r: region<int>, p: int@r), writes(r) {/*...*/}

task am_i_parallel2(r: region<int>, p: int@r): int
  , reads(r), writes(r)
{
  let x = expensive2a(r, p);
  expensive2b(r, p);
  return x + *p;
}
```

Again, `expensive2a` and `expensive2b` *can* run in parallel, although
this case is trickier. Parallelizing the two tasks isn't trivial,
because if both try to access the same physical memory at the same
time, then `expensive1` could potentially read invalid results.

In order to parallelize this code, Legion has to make a copy of region
`r`. Then `expensive2a` and `expensive2b` can run in parallel, and
once both finish, Legion declares `expensive2b`'s copy of `r` to be
the official one going forward (so that `am_i_parallel2` reads the
correct value once `expensive2b` returns).

This might or might not be a performance win, depending on the
application and machine. The decision to make this copy can be made by
the user via the mapping interface (not described in this document).

Ok, what about this one?

```legion
task expensive3a(r: region<int>, p: int@r), reads(r), writes(r) {/*...*/}

task expensive3b(r: region<int>, p: int@r), reads(r), writes(r) {/*...*/}

task am_i_parallel3(r: region<int>): int
  , reads(r), writes(r)
{
  let p = new<int@r>();
  let q = new<int@r>();
  expensive3a(r, p);
  expensive3b(r, q);
  return *p + *q;
}
```

Even though `p` and `q` are different pointers, `expensive3a` and
`expensive3b` declared their read-write privileges at the level of
region `r`, and therefore *cannot* run in parallel. That said, we'd
really like this example to run in parallel, because we know the two
pointers are different, even if Legion doesn't.

To make this work, we need to partition the region `r` into two
disjoint subregions `r1` and `r2`. The disjoint partition proves to
the compiler that the pieces do not overlap, and therefore
`expensive4a` and `expensive4b` can run in parallel.

```legion
task expensive4a(r: region<int>, x: int@r), reads(r), writes(r) {/*...*/}

task expensive4b(r: region<int>, x: int@r), reads(r), writes(r) {/*...*/}

task am_i_parallel4(r: region<int>): int
  , reads(r), writes(r)
{
  let p = partition<r, disjoint>(coloring<r>());
  let r0 = p[0];
  let r1 = p[1];
  let x = new<int@r0>();
  let y = new<int@r1>();
  expensive4a(r0, x);
  expensive4b(r1, y);
  return *x + *y;
}
```

This version *can* run in parallel.

## Data Types

### Boolean

Legion has a boolean data type `bool` with two values `true` and `false`.

### Integers

Legion has many integer types.

Legion   C           size          signed?
-------- ----------- ------------- --------
`int`    `intptr_t`  pointer-sized signed
`uint`   `uintptr_t` pointer-sized unsigned
`int8`   `int8_t`    8 bits        signed
`int16`  `int16_t`   16 bits       signed
`int32`  `int32_t`   32 bits       signed
`int64`  `int64_t`   64 bits       signed
`uint8`  `uint8_t`   8 bits        unsigned
`uint16` `uint16_t`  16 bits       unsigned
`uint32` `uint32_t`  32 bits       unsigned
`uint64` `uint64_t`  64 bits       unsigned

You'll notice that Legion is missing equivalents of C's `char`,
`short`, `int`, `long`, and `long long`. If you need these for
interoperability with your C code, please contact the authors and
we'll be happy to add them for you.

### Floating-Point

Legion supports single and double precision floating-point.

Legion   C        reference
-------- -------- -------------------------
`float`  `float`  IEEE 754 single-precision
`double` `double` IEEE 754 double-precision

### Casts

Legion does no implicit coercions between numeric types. To cast
between numeric types, use the name of the type as if it were a
task.

```legion
task casting_explicit_ints(x: int): int
{
  let x8 = int8(x);
  let x16 = int16(x);
  let x32 = int32(x);
  let x64 = int64(x);
  return int(x8) + int(x16) + int(x32) + int(x64);
}
```

## Arrays

Arrays in Legion are a special kind of region with support for
indexing. Arrays are created from an *index space*, abbreviated
*ispace*, which specifies the set of indices for which the array
contains data.

```legion
task an_array()
{
  let is = ispace<int>(2);
  let a = array<is, double>();
  a[0] = 3.14;
  a[1] = 6.28;
}
```

The code above creates an index space `is`, with indices are of type
`int` and 2 elements. The array is created from this index space and
contains elements of type `double`.

Index spaces can be used inside for loops, which provide convenient
iteration over arrays.

```legion
task sum_array(is: ispace<int>, a: array<is, double>): double
  , reads(a)
{
  var sum = 0.0;
  for i in is {
    sum += a[i];
  }
  return sum;
}
```

Because arrays are regions, they require the same privilege
declarations as regions.

## Structs

Legion supports structs that are very similar to C's structs, although
the syntax differs somewhat.

```legion
struct point
{
  x: int,
  y: int,
  z: int,
}

task make_point(a: int, b: int, c: int): point
{
  return {x: a, y: b, z: c};
}

task project_z(p: point): int
{
  return p.z;
}

task increment_z(p: point, i: int): point
{
  return p{z: p.z + i}; // does not modify x and y
}
```

Struct values can be created with the syntax shown in `make_point`,
above.

A very similar syntax can be used to update the values of a struct, as
in `increment_z`. Note that this operation returns a new value, and
does not modify the existing struct.

For simple structs, Legion mimics the way C does padding on structs,
so you can pass structs by value to and from C. Note that this only
holds when you pass the *whole* struct. If you start doing struct
slicing (see below), then you cannot know what form the struct will
take.

Structs are passed by-value, as is everything in Legion.

Structs can, of course, contain other structs.

```legion
struct inner
{
  a: int,
  b: int,
}

struct outer
{
  c: inner,
  d: inner,
}
```

### Struct Slicing

In some cases you don't need all of the fields in a struct to perform
a computation. Struct slicing is a technique for describing exactly
which fields in a struct you need to access.

Suppose you have a region of `point`'s (from the previous example),
but only need access to the `x` field. You can declare a task that
only takes field `x`:

```legion
task takes_point_x(r: region<point>, p: point@r)
  , reads(r.x), writes(r.x) { /*...*/ }
```

If you need access to two or more fields in the same struct, you can
put them in `{}`.

```legion
task takes_point_yz(r: region<point>, p: point@r)
  , reads(r.{y, z}), writes(r.{y, z}) { /*...*/ }
```

Now, `takes_point_x` and `takes_point_yz` can run in parallel, even on
the same region, because they take different fields of the struct.

### Structs with Regions

When structs contains pointers, they must be parameterized on the
regions those pointers point into.

```legion
struct list<r: region<list<r>>>
{
  data: int,
  next: list<r>@r,
}
```

This list is parameterized on the region that it lives in. The fields
of the struct can refer to the region parameter, like the `next`
field, that points into `r`.

You could create an instance of this list with the following code:

```legion
task make_list(r: region<list<r>>, x: int): list<r>@r
  , writes(r)
{
  if (x < 0) {
    return null<list<r>@r>();
  } else {
    let head = new<list<r>@r>();
    let tail = make_list(r, x - 1);

    head->data = x;
    head->next = tail;

    return head;
  }
}
```

However, the above code has a problem: it will be impossible to
partition the list so that multiple tasks can operate on it in
parallel. This isn't such a big deal for a linked list, because
accessing an element requires a sequential walk of the list anyway,
and but for other data structures, like trees, this is a problem.

In order to work in parallel on a binary tree, we'll need to describe
the decomposition of the tree in a way that Legion
understands. Specifically, at each node, we'd like to say that the
children will live in disjoint subregions. This requires us to store
regions themselves in the nodes of the tree.

Here's the definition for the struct:

```legion
struct tree<top: region<tree<?>>>
           [left: region<tree<?>>,
            right: region<tree<?>>]
  , left <= top
  , right <= top
  , left * right
{
  data: int,
  lhs: tree<left>@left,
  rhs: tree<right>@right,
}
```

This time, `tree` takes two types of parameters, one surrounded in
`<>` and one in `[]`. The `<>` are *template* parameters, similar in
many ways to C++. Specifically, template parameters must be available
statically at compile time. The `[]`, called *existential* parameters,
represent parameters passed dynamically, at runtime. The existential
parameters are stored inside the struct, so `tree` has five fields:
`left`, `right`, `data`, `lhs`, and `rhs`.

In the tree, existential parameters are needed to describe the two
branches because the structure of the tree may not exist until
runtime, and might even change during the execution of the
program. Template parameters are fixed for the lifetime of the struct,
but the existential parameters can potentially change dynamically.

The other thing we need for `tree` to be useful to us is to know that
`left` and `right` form the left and right branches of the tree. We do
this with constraints. The constraints `left <= top` and `right <=
top` tell Legion that `left` and `right` are subregions of `top`. The
constraint `left * right` tells Legion that `left` and `right` are
disjoint.

Because the Legion compiler relies on its knowledge of the structure
of data in memory, at least at the granularity of regions, the
compiler needs to check that stores into data structures preserve that
structure. Thus, when a struct value is stored into a region the
compiler must check that it has the necessary constraints to match the
type of the region's elements. The `pack` operator takes a value to be
stored into a pointer and an associated set of constraints and
performs the necessary checks. The `unpack` operator is the inverse of
`pack` and makes a value and set of constraints from a dereferenced
pointer available for use. Because of the checking done by `pack`,
Legion already knows at the point of the `unpack` that the necessary
constraints are guaranteed to hold.

Here is code to turn a list into a tree.

```legion
task make_tree(rl: region<list<rl>>, rt: region<tree<?>>,
               l: list<rl>@rl, min: int, max: int)
  : tree<rt>@rt
  , reads(rl, rt)
  , writes(rt)
{
  if (isnull(l)) {
    return null<tree<rt>@rt>();
  } else {
    if (l->data < min || l->data > max) {
      return make_tree(rl, rt, l->next, min, max);
    } else {
      let root = new<tree<rt>@rt>();

      let p = partition<rt, disjoint>(coloring<rt>());
      let rtl = p[0];
      let rtr = p[1];
      let lhs = make_tree(rl, rtl, l->next, min, l->data);
      let rhs = make_tree(rl, rtr, l->next, l->data, max);

      *root = pack {data: l->data, lhs: lhs, rhs: rhs}
                   as tree<rt>[rtl, rtr];
      return root;
    }
  }
}
```
