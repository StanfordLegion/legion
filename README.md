legion
======

Publicly visible repository for the Legion parallel programming project at Stanford University

Overview
==================================================================================
Legion is a programming model and runtime system designed for capturing properties
of program data to enable higher productivity and efficiency when running on
distributed heterogeneous hardware.  More details of the Legion programming model
can be found in our Supercomputing paper.

http://theory.stanford.edu/~aiken/publications/papers/sc12.pdf

The Legion repository is seperated into directories described below:

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
tool that we use for doing correctness and performance debugging in Legion.

Dependences
==================================================================================
We have only tested Legion running on Linux based systems.  An implementation of
the POSIX threads library is required for running all Legion applications.  For
running applications on clusters and GPUs, we require at least CUDA 4.2 and
and installation of GASNET.  Verify that the correct locations of these installations
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

-cat <logger_name>[,logger_name]*  restricts logging to the comma seperated list of loggers

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

Debugging Programs
==================================================================================
Legion currently has two primary tools for doing debugging.  The first is the 
'legion_spy' tool contained in the 'tools' directory.  To use legion spy, first
add the '-DLEGION_SPY' flag to 'CC_FLAGS' in the Makefile of your application
and recompile in DEBUG mode.  The run your application with the follwing flags
'-cat legion_spy -level 1' and dump the results of standard error to a file.  
Then run legion spy as follows:

python legion_spy -l -p <file>

The legion spy tool will parse the results of the log file.  The '-l' flag will
check the results of the logical region dependence analysis.  If there are any 
errors they should be reported to the Legion developers.  The '-p' file will
dump event graphs corresponding to all of the low-level runtime event dependences
between any tasks, copies, reductions, or inline mapping operations.  These graphs
are useful for illustrating the actual dependences computed in the physical states
of the region trees.

The other tool available in Legion for debugging is the log files capturing the
physical state of all region trees on every instance of the high-level runtime.
For applications compiled in DEBUG mode, simply pass the '-hl:tree' flag as input
to dump the files.

Other Features
==================================================================================
- Pointer Checks: Users can enable dynamic pointer checks of all physical region
accesses by compiling with the '-DPOINTER_CHECKS' flag.

- Inorder Execution: Users can force the high-level runtime to execute all tasks
in program order by compiling with the '-DINORDER_EXECUTION' flag and then passing
'-hl:inorder' flag as an input to the application.

- Dynamic Independence Tests: Users can request the high-level runtime perform 
dynamic independence tests between regions and partitions by compiling with
the '-DDYNAMIC_TESTS' flag and then passing '-hl:dynamic' flag as input.

