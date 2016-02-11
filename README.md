Legion [![Build Status](https://travis-ci.org/StanfordLegion/legion.svg?branch=master)](https://travis-ci.org/StanfordLegion/legion)
==================================================================================

The Legion homepage is now at [legion.stanford.edu](http://legion.stanford.edu).

Publicly visible repository for the Legion parallel programming project at Stanford University.

Overview
==================================================================================
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

The Legion repository is separated into directories described below:

runtime: The runtime directory contains all of the source code for the Legion runtime
system.  The primary files that most users will be interested in are 'legion.h' and
'lowlevel.h' which represent the interfaces to the high-level and low-level runtime
systems for Legion respectively.

apps: The applications directory currently contains two applications: saxpy and circuit.
Saxpy is an unoptimized implementation of the saxpy kernel for illustrating
different features of Legion.  The circuit directory contains the source code for the
circuit simulation used in our paper on Legion.  We plan to release additional
application examples once we figure out the necessary licensing constraints.

tools: The tools directory contains the source code for the 'legion_spy' debugging
tool that we use for doing correctness and performance debugging in Legion.  We also
have a 'legion_prof' tool that does performance profiling of Legion application
runs and can be used for generating both statistics and execution diagrams.

Dependencies
==================================================================================
We have only tested Legion running on Linux based systems.  An implementation of
the POSIX threads library is required for running all Legion applications.  For
running applications on clusters and GPUs, we require at least CUDA 4.2 and
and an installation of GASNET.  Verify that the correct locations of these installations
are set in 'runtime/runtime.mk'.  At least Python 2.4 is required to run the
'legion_spy' debugging tool.

Running Programs
==================================================================================
When running applications users must set the 'LG_RT_DIR' environment variable to 
point to the 'runtime' directory for the repository.  Makefiles will report an error
if the environment variable is not set.

Each application has a Makefile in its directory that is used to control the 
compilation of the application.  At the top of the Makefile there are several
different variables that can be used to control how the application is built.
By default, applications are compiled in debug mode.  This can be changed by
commenting out the 'DEBUG' variable.  Users can statically control the minimum
level of logging information printed by the runtime by setting the 'OUTPUT_LEVEL'
variable.  Choices for 'OUTPUT_LEVEL' can be seen at the top of 'runtime/utilities.h'.
The 'SHARED_LOWLEVEL' runtime variable controls whether the application is compiled
to run on the shared-low-level runtime, or if the variable is not set, the application
will be targeted at the GASNET-GPU generic low-level runtime.

After compilation, programs are launched differently depending on their target
platform.  Applications targeted at the shared-low-level runtime can be run
as a regular process, while applications targeted at the general low-level runtime
must be launched using the 'gasnetrun' command (see GASNET documentation).

Both the low-level and high-level runtime have flags for controlling execution.
Below are some of the more commonly used flags:

-cat <logger_name>[,logger_name]*  restricts logging to the comma separated list of loggers

-level <int>    dynamically restrict the logging level to all statements at
                   the given level number and above.  See 'runtime/utilities.h' for
                   how the numbers associated with each level.

-ll:cpu <int>   the number of CPU processors to create per process

-ll:gpu <int>   number of GPU Processors to create per process

-ll:csize <int>   size of DRAM Memory per process in MB

-ll:gsize <int>    size of GASNET registered RDMA memory available per process in MB

-ll:fsize <int>    size of framebuffer memory for each GPU in MB

-ll:zsize <int>    size of zero-copy memory for each GPU in MB

-ll:util <int>     specify the number of utility processors created per process

-hl:window <int>   specify the maximum number of tasks that can be created in a parent task window

-hl:sched <int>    minimum number of tasks to try to schedule for each invocation of the scheduler

The default mapper also has several flags for controlling the default mapping.
See default_mapper.cc for more details.

Developing Programs
==================================================================================

Using Makefile:

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

Using CMake:

To develop an new legion application with CMake, begin by building legion:
cd legion_src_dir
mkdir build
cd build
ccmake ..

modify CMAKE options like : CMAKE_INSTALL_PREFIX -  where you would like to install legion
                            BUILD_EXMPLES - do you want to build legion with examples
                             ENABLE _CUDA, ENABLE_GASNET etc.
push "c" to configure, then "g" to create makefile

make
make install

After you build legion, specify PATH to the legion install directory in your project's CMakeLists.txt and add legion libraries and include path to CMAKE_EXE_LINKER_FLAGS.



Debugging Programs
==================================================================================
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

Other Features
==================================================================================
- Bounds Checks: Users can enable dynamic pointer checks of all physical region
accesses by compiling with the '-DBOUNDS_CHECKS' flag.

- Inorder Execution: Users can force the high-level runtime to execute all tasks
in program order by compiling with the '-DINORDER_EXECUTION' flag and then passing
'-hl:inorder' flag as an input to the application.

- Dynamic Independence Tests: Users can request the high-level runtime perform 
dynamic independence tests between regions and partitions by compiling with
the '-DDYNAMIC_TESTS' flag and then passing '-hl:dynamic' flag as input.

