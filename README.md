# Legion 


[Legion](http://legion.stanford.edu) is a parallel programming model
for distributed, heterogeneous machines.

## Branches

The Legion team uses this repository for active development, so please make sure you're
using the right branch for your needs:

  * stable [![Build Status](https://travis-ci.org/StanfordLegion/legion.svg?branch=stable)](https://travis-ci.org/StanfordLegion/legion/branches) - This is the default branch if you clone the
repository.  It is generally about a month behind the master branch, allowing us to get some
mileage on larger changes before foisting them on everybody.  Most users of Legion should use this
branch, although you should be prepared to try the master branch if you run into issues.
Updates are moved to the stable branch roughly monthly, although important bug fixes may be
applied directly when needed.  Each batch of updates is given a "version" number, and
[CHANGES.txt](https://github.com/StanfordLegion/legion/blob/stable/CHANGES.txt) lists the
major changes.
  * master [![Build Status](https://travis-ci.org/StanfordLegion/legion.svg?branch=master)](https://travis-ci.org/StanfordLegion/legion/branches) - This is the "mainline" used by the Legion team,
and contains changes and bug fixes that may not have made it into the stable branch yet.  If you
are a user of "bleeding-edge" Legion functionality, you will probably need to be using this branch.
  * lots of other feature branches - These exist as necessary for larger changes, and users will
generally want to steer clear of them.  :)

## Overview

Legion is a programming model and runtime system designed for decoupling the specification
of parallel algorithms from their mapping onto distributed heterogeneous architectures.  Since
running on the target class of machines requires distributing not just computation but data
as well, Legion presents the abstraction of logical regions for describing the structure of
program data in a machine independent way.  Programmers specify the partitioning of logical
regions into subregions, which provides a mechanism for communicating both the independence 
and locality of program data to the programming system.  Since the programming system
has knowledge of both the structure of tasks and data within the program, it can aid the
programmer in host of problems that are commonly the burden of the programmer:

  * Discovering/verifying correctness of parallel execution: determining when two tasks
    can be run in parallel without a data race is often difficult.  Legion provides mechanisms
    for creating both implicit and explicit parallel task launches.  For implicit constructs 
    Legion will automatically discover parallelism.  For explicit constructs, Legion will
    notify the programmer if there are potential data races between tasks intended to be
    run in parallel.
  * Managing communication: when Legion determines that there are data dependencies between
    two tasks run in different locations, Legion will automatically insert the necessary
    copies and apply the necessary constraints so the second task will not run until
    its data is available.  We describe how tasks and data are placed in the next paragraph
    on mapping Legion programs.

The Legion programming model is designed to abstract computations in a way that makes
them portable across many different potential architectures.  The challenge then is to make
it easy to map the abstracted computation of the program onto actual architectures.  At
a high level, mapping a Legion program entails making two kinds of decisions:
  
  1. For each task: select a processor on which to run the task.
  2. For each logical region a task needs: select a memory in which to create
     a physical instance of the logical region for the task to use.

To facilitate this process Legion introduces a novel runtime 'mapping' interface.  One of the
NON-goals of the Legion project was to design a programming system that was magically capable 
of making intelligent mapping decisions.  Instead the mapping interface provides a declarative
mechanism for the programmer to communicate mapping decisions to the runtime system
without having to actually write any code to perform the mapping (e.g. actually writing
the code to perform a copy or synchronization).  Furthermore, by making the mapping interface
dynamic, it allows the programmer to make mapping decisions based on information that
may only be available at runtime.  This includes decisions based on:

  * Program data: some computations are dependent on data (e.g. is our irregular graph
    sparse or dense in the number of edges).
  * System data: which processors or nodes are currently up or down, or which are running
    fast or slow to conserve power.
  * Execution data: profiling data that is fed back to the mapper about how a certain
    mapping performed previously.  Alternatively which processors are currently over-
    or under- loaded.

All of this information is made available to the mapper via various mapper calls, some
of which query the mapping interface while others simply are communicating information
to the mapper.

One very important property of the mapping interface is that no mapping decisions are
capable of impacting the correctness of the program.  Consequently, all mapping decisions
made are only performance decisions.  Programmers can then easily tune a Legion application
by modifying the mapping interface implementation without needing to be concerned
with how their decisions impact correctness.  Ultimately, this makes it possible in Legion
to explore whole spaces of mapping choices (which tasks run on CPUs or GPUs, or where data 
gets placed in the memory hierarchy) simply by enumerating all the possible mapping
decisions and trying them.

To make it easy to get a working program, Legion provides a default mapper implementation
that uses heuristics to make mapping decisions.  In general these decision are good, but
they are certain to be sub-optimal across all applications and architectures.  All calls
in the mapping interface are C++ virtual functions that can be overridden, so programmers
can extend the default mapper and only override the mapping calls that are impacting performance.
Alternatively a program can implement the mapping interface entirely from scratch.

For more details on the Legion programming model and its current implementation
we refer to you to our Supercomputing paper.

http://theory.stanford.edu/~aiken/publications/papers/sc12.pdf

## Contents

This repository includes the following contents:

  * `tutorial`: Source code for the [tutorials](http://legion.stanford.edu/tutorial/).
  * `examples`: Larger examples for advanced programming techniques.
  * `apps`: Several complete Legion applications.
  * `language`: The [Regent programming language](http://regent-lang.org/) compiler and examples.
  * `runtime`: The core runtime components:
      * `legion`: The Legion runtime itself (see `legion.h`).
      * `realm`: The Realm low-level runtime (see `realm.h`).
      * `mappers`: Several mappers, including the default mapper (see `default_mapper.h`).
  * `tools`: Miscellaneous tools:
      * `legion_spy.py`: A [visualization tool](http://legion.stanford.edu/debugging/#legion-spy) for task dependencies.
      * `legion_prof.py`: A task-level [profiler](http://legion.stanford.edu/profiling/#legion-prof).

## Dependencies

To get started with Legion, you'll need:

  * Linux, macOS, or another Unix
  * A C++ 98 (or newer) compiler (GCC, Clang, Intel, or PGI) and GNU Make
  * *Optional*: Python 2.7 (used for profiling/debugging tools)
  * *Optional*: CUDA 5.0 or newer (for NVIDIA GPUs)
  * *Optional*: [GASNet](https://gasnet.lbl.gov/) (for networking, see
     [installation instructions](http://legion.stanford.edu/gasnet/))
  * *Optional*: LLVM 3.5-3.9 (for dynamic code generation)
  * *Optional*: HDF5 (for file I/O)

## Installing

Legion is currently compiled with each application. To try a Legion
application, just call `make` in the directory in question. The
`LG_RT_DIR` variable is used to locate the Legion `runtime`
directory. For example:

```bash
git clone https://github.com/StanfordLegion/legion.git
export LG_RT_DIR="$PWD/legion/runtime"
cd legion/examples/full_circuit
make
./ckt_sim
```

## Makefile Variables

The Legion Makefile includes several variables which influence the
build. These may either be set in the environment (e.g. `DEBUG=0
make`) or at the top of each application's Makefile.

  * `DEBUG=<0,1>`: controls optimization level and enables various
    dynamic checks which are too expensive for release builds.
  * `OUTPUT_LEVEL=<level_name>`: controls the compile-time [logging
    level](http://legion.stanford.edu/debugging/#logging-infrastructure).
  * `USE_CUDA=<0,1>`: enables CUDA support.
  * `USE_GASNET=<0,1>`: enables GASNet support (see [installation instructions](http://legion.stanford.edu/gasnet/)).
  * `USE_LLVM=<0,1>`: enables LLVM support.
  * `USE_HDF=<0,1>`: enables HDF5 support.

## Build Flags

In addition to Makefile variables, compilation is influenced by a
number of build flags. These flags may be added to the environment
variable `CC_FLAGS` (or again set inside the Makefile).

  * `CC_FLAGS=-DLEGION_SPY`: enables [Legion Spy](http://legion.stanford.edu/debugging/#legion-spy).
  * `CC_FLAGS=-DPRIVILEGE_CHECKS`: enables [extra privilege checks](http://legion.stanford.edu/debugging/#privilege-checks).
  * `CC_FLAGS=-DBOUNDS_CHECKS`: enables [dynamic bounds checks](http://legion.stanford.edu/debugging/#bounds-checks).

## Command-Line Flags

Legion and Realm accept command-line arguments for various runtime
parameters. Below are some of the more commonly used flags:

  * `-level <category>=<int>`:
    sets [logging level](http://legion.stanford.edu/debugging/#logging-infrastructure) for `category`
  * `-logfile <filename>`:
    directs [logging output](http://legion.stanford.edu/debugging/#logging-infrastructure) to `filename`
  * `-ll:cpu <int>`: CPU processors to create per process
  * `-ll:gpu <int>`: GPU processors to create per process
  * `-ll:cpu <int>`: utility processors to create per process
  * `-ll:csize <int>`: size of CPU DRAM memory per process (in MB)
  * `-ll:gsize <int>`: size of GASNET global memory available per process (in MB)
  * `-ll:rsize <int>`: size of GASNET registered RDMA memory available per process (in MB)
  * `-ll:fsize <int>`: size of framebuffer memory for each GPU (in MB)
  * `-ll:zsize <int>`: size of zero-copy memory for each GPU (in MB)
  * `-lg:window <int>`: maximum number of tasks that can be created in a parent task window
  * `-lg:sched <int>`: minimum number of tasks to try to schedule for each invocation of the scheduler

The default mapper also has several flags for controlling the default mapping.
See `default_mapper.cc` for more details.

## Developing Programs

To start a new Legion application, make a new directory and copy
`apps/Makefile.template` into your directory under the name
`Makefile`. Fill in the appropriate fields at the top of the Makefile
with the filenames needed for your application.

Most Legion APIs are described in `legion.h`; a smaller number are
described in the various header files in the `runtime/realm`
directory. The default mapper is available in `default_mapper.h`.

## Debugging

Legion has a number of tools to aid in debugging programs.

### Extended Correctness Checks

Compile with `DEBUG=1 CC_FLAGS="-DPRIVILEGE_CHECKS -DBOUNDS_CHECKS"
make` and rerun the application. This enables dynamic checks for
privilege and out-of-bounds errors in the application. (These checks
are not enabled by default because they are relatively expensive.) If
the application runs without terminating with an error, then continue
on to Legion Spy.

### Legion Spy

Legion provides a task-level visualization tool called Legion
Spy. This captures the logical and physical dependence graphs. These
may help, for example, as a sanity check to ensure that the correct
sequence of tasks is being launched (and the tasks have the correct
dependencies). Legion Spy also has a self-checking mode which can
validate the correctness of the runtime's logical and physical
dependence algorithms.

To capture a trace, invoke the application with `-lg:spy -logfile
spy_%.log`. (No special compile-time flags are required.) This will
produce a log file per node. Call the post-processing script to render
PDF files of the dependence graphs:

```bash
./app -lg:spy -logfile spy_%.log
$LG_RT_DIR/../tools/legion_spy.py -dez spy_*.log
```

To run Legion Spy's self-checking mode, Legion must be built with the
flag `-DLEGION_SPY`. Following this, the application can be run again,
and the script used to validate (or render) the trace.

```bash
DEBUG=1 CC_FLAGS="-DLEGION_SPY" make
./app -lg:spy -logfile spy_%.log
$LG_RT_DIR/../tools/legion_spy.py -lpa spy_*.log
$LG_RT_DIR/../tools/legion_spy.py -dez spy_*.log
```

## Profiling

Legion contains a task-level profiler. No special compile-time flags
are required. However, it is recommended to build with `DEBUG=0 make`
to avoid any undesired performance issues.

To profile an application, run with `-lg:prof <N>` where `N` is the
number of nodes to be profiled. (`N` can be less than the total number
of nodes---this profiles a subset of the nodes.) Use the
`-lg:prof_logfile <logfile>` flag to save the output from each node to
a separate file. The argument to the `-lg:prof_logfile` flag follows
the same format as for `-logfile`, except that a `%` (to be replaced
by the node number) is mandatory. Finally, pass the resulting log
files to `legion_prof.py`.

```bash
DEBUG=0 make
./app -lg:prof <N> -lg:prof_logfile prof_%.gz
$LG_RT_DIR/../tools/legion_prof.py prof_*.gz
```

This will generate a subdirectory called `legion_prof` under the
current directory, including a file named `index.html`. Open this file
in a browser.

## Other Features

- Inorder Execution: Users can force the high-level runtime to execute
all tasks in program order by passing `-lg:inorder` flag on the
command-line.

- Dynamic Independence Tests: Users can request the high-level runtime
perform dynamic independence tests between regions and partitions by
passing `-lg:dynamic` flag on the command-line.
