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

The Legion repository contains the following directories:

  * `tutorial`: Source code for the [tutorials](http://legion.stanford.edu/tutorial/).
  * `examples`: Larger examples for advanced programming techniques.
  * `apps`: Several complete Legion applications.
  * `language`: The [Regent programming language](http://regent-lang.org/) compiler and examples.
  * `runtime`: The core runtime components:
    * `legion`: The Legion runtime itself (see `legion.h`).
    * `realm`: The Realm low-level runtime (see `realm.h`).
    * `mappers`: Several mappers, including the default mapper (see `default_mapper.h`).
  * `tools`: Miscellaneous tools:
    * `legion_spy.py`: A debugging tool renders task dependencies.
    * `legion_prof.py`: A task-level profiler.

## Dependencies

To get started with Legion, you'll need:

  * Linux, OS X, or another Unix
  * A C++ 98 or newer compiler (GCC, Clang, Intel, or PGI) and GNU Make
  * *Optional*: Python 2.7 (used for profiling/debugging tools)
  * *Optional*: CUDA 5.0 or newer (for NVIDIA GPUs)
  * *Optional*: [GASNet](https://gasnet.lbl.gov/) (for networking)
  * *Optional*: LLVM 3.5 (for dynamic code generation)
  * *Optional*: HDF5 (for file I/O)

## Installing

Legion is currently compiled with each application. To try a Legion
application, just call `make` in the directory in question. The
`LG_RT_DIR` variable is used to locate the Legion `runtime`
directory. For example:

```
git clone https://github.com/StanfordLegion/legion.git
export LG_RT_DIR="$PWD/legion/runtime"
cd legion/examples/full_circuit
make
./ckt_sim
```

## Build Flags

The Legion Makefile template includes several variables which
influence the build. These may either be set on the command-line
(e.g. `DEBUG=0 make` or at the top of each application's Makefile).

  * `DEBUG=<0,1>`: controls optimization level and enables various
    dynamic checks which are too expensive for release builds.
  * `OUTPUT_LEVEL=<level_name>`: controls the compile-time logging
    level (see `runtime/realm/logging.h` for a list of logging level
    names).
  * `USE_CUDA=<0,1>`: enables CUDA support.
  * `USE_GASNET=<0,1>`: enables GASNET support.
  * `USE_LLVM=<0,1>`: enables LLVM support.
  * `USE_HDF=<0,1>`: enables HDF5 support.

## Command-Line Flags

Legion and Realm accept command-line arguments for various runtime
parameters. Below are some of the more commonly used flags:

  * `-level <logger_name>=<int>`:
    dynamic logging level for a given logger name (see `runtime/realm/logging.h` for
    the list of logging levels)
  * `-logfile <filename>`:
    directs logging output to `filename`
  * `-ll:cpu <int>`: CPU processors to create per process
  * `-ll:gpu <int>`: GPU processors to create per process
  * `-ll:cpu <int>`: utility processors to create per process
  * `-ll:csize <int>`: size of CPU DRAM memory per process (in MB)
  * `-ll:gsize <int>`: size of GASNET global memory available per process (in MB)
  * `-ll:rsize <int>`: size of GASNET registered RDMA memory available per process (in MB)
  * `-ll:fsize <int>`: size of framebuffer memory for each GPU (in MB)
  * `-ll:zsize <int>`: size of zero-copy memory for each GPU (in MB)
  * `-hl:window <int>`: maximum number of tasks that can be created in a parent task window
  * `-hl:sched <int>`: minimum number of tasks to try to schedule for each invocation of the scheduler

The default mapper also has several flags for controlling the default mapping.
See default_mapper.cc for more details.

## Developing Programs

To develop a new legion application, begin by creating a new directory in the
applications directory.  Make a copy of the 'Makefile.template' file in the
'apps' directory to use as the Makefile.  Fill in the appropriate fields
at the top of the Makefile so that they contain the file names for each of
the different files needed for your application.

To begin writing Legion applications you should only need to include the 
'legion.h' header file.  The Makefile guarantees this file will be in the
include path for you application when compiling.  More documentation of
the 'legion.h' header file is currently in progress.

To extend the default mapper, you will also need to include 'default_mapper.h'
into whatever file has the declaration for your custom mapper.

## Debugging Programs

Legion currently has two primary tools for doing debugging.  The first is the 
'legion_spy' tool contained in the 'tools' directory.  To use legion spy, first
add the '-DLEGION_SPY' flag to 'CC_FLAGS' in the Makefile of your application
and recompile in DEBUG mode.  The run your application with the following flags
'-cat legion_spy -level 1' and dump the results of standard error to a file.  
Then run legion spy as follows:

python legion_spy -l -p <file>

The legion spy tool will parse the results of the log file.  The '-l' flag will
check the results of the logical region dependence analysis.  If there are any 
errors they should be reported to the Legion developers.  The '-p' file will
dump event graphs corresponding to all of the low-level runtime event dependencies
between any tasks, copies, reductions, or inline mapping operations.  These graphs
are useful for illustrating the actual dependencies computed in the physical states
of the region trees.

The other tool available in Legion for debugging is the log files capturing the
physical state of all region trees on every instance of the high-level runtime.
For applications compiled in DEBUG mode, simply pass the '-hl:tree' flag as input
to dump the files.

## Other Features

- Bounds Checks: Users can enable dynamic pointer checks of all physical region
accesses by compiling with the '-DBOUNDS_CHECKS' flag.

- Inorder Execution: Users can force the high-level runtime to execute all tasks
in program order by compiling with the '-DINORDER_EXECUTION' flag and then passing
'-hl:inorder' flag as an input to the application.

- Dynamic Independence Tests: Users can request the high-level runtime perform 
dynamic independence tests between regions and partitions by compiling with
the '-DDYNAMIC_TESTS' flag and then passing '-hl:dynamic' flag as input.
