**Important:** Regent static control replication (SCR, called SPMD
below) is deprecated and will be removed in a future release.

SPMD is an advanced programming feature of Regent which enables scalability
out to large numbers of nodes. SPMD-style programs can be generated
automatically by the Regent compiler. This is an advanced optimization that is
under active research. As a result, the optimization has a number of
limitations which it currently imposes on the source program.

The hope is to lift these limitations over time, but please understand that
this is research and thus some uncertainty as to the final outcomeis expected.

This optimization requires [RDIR](https://github.com/StanfordLegion/rdir), a
suite of dataflow-style optimizations for Regent. When installing Regent, make
sure RDIR is enabled.

Automatic SPMD execution can be applied to any arbitrary nested control flow,
with the following limitations.

  * The *leaves* (the innermost control flow constructs that
    themselves contain no nested control flow) must be `for` loops
  * All leaves must have the same loop bounds
  * The bodies of leaves each contain a single (?) task launch
      * There must be no loop-carried dependencies between iterations
      * Region arguments must be of the form `p[i]` where `i` is the loop index
          * Note: You can simulate `p[f(i)]` by baking the function `f` into
            an aliased partition `q`
      * Partition arguments must be of the form `p[i]`
  * Tasks called outside of leaves must be pure functions (no region or
    partition arguments)
  * Any assignments to scalar variables inside leaves use a reduction operator
  * No direct region accesses are allowed

There are a number of known issues that are non-fundamental. Most of
these have known fixes, but are waiting for a concrete use case before
being fixed. Test cases are always welcome!

  * RDIR only understands a fixed set of Regent AST nodes; any construct
    outside this set causes SPMD to fail
  * There are some known issues with interactions between nested control flow
    and required synchronization for data movement
  * The error messages when something goes wrong are currently poor
