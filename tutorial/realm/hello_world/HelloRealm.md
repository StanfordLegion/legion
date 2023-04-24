# Realm Hello World

The tutorial begins with a simple "hello world" example that showcases the basics. 
You can access the source code, the Makefile and CMakeList.txt for building 
and running the application, in the `tutorial/realm` directory of the repository. 
By going through these tutorial programs in detail, we will demonstrate how to 
effectively use the Realm C++ runtime API.

Here is a list of covered topics:

* [Realm Namespaces](#realm-namespaces)
* [Realm Runtime Start-Up](#realm-runtime-start-up)
* [Registering Realm Tasks](#registering-realm-tasks)
* [Launching Tasks](#launching-tasks)
* [Shuting Down Runtime](#shuting-down-runtime)

## Realm Namespaces

Each Realm class has its own C++ header file. All classes are 
aggregated in `realm.h` and can be included in an application for
convenience. Each class definition is placed in a `Realm` namespace to
avoid naming conflicts.

## Realm Runtime Starting-Up

The following code illustrates how to initializes a singleton `Runtime` object.
```c++
Runtime rt;
rt.init(&argc, &argv);
```
The initialization must be performed by every application process. After
initialization is complete, the runtime remains mostly idle (except system
status checks) and waits for the task launches.

## Registering Realm Tasks

A Realm task is an asynchronous operation, which can be defined by a task ID 
and one or more task bodies representing the implementations of the task.
This tutorial contains two types of tasks: main task and hello world task. 
Therefore, we use an enumeration of two members to store the IDs for each task with which the Realm runtime will associate.
The first task should always start from or larger than `Processor::TASK_ID_FIRST_AVAILABLE`, because Realm reserve 
some numbers for internal tasks.
```c++
enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};

```
A task can have an implementation on every kind of processor, e.g., CPU, GPU, etc. 
For example, the `main_task` is the CPU implementation of the main task, while
the `hello_cpu_task`, `hello_gpu_task` and `hello_omp_task` are
the CPU, GPU and OpenMP implementations of the hello world task, respectively.
It is worth noting that in Realm, a CPU processor typically refers to a physical CPU core, 
and therefore, the implementation of a CPU task generally is single-threaded. 
A GPU processor has a CUDA/HIP context associated with it, 
which allows it to launch GPU tasks containing CUDA/HIP kernels. 
However, for the sake of simplicity, we do not launch actual CUDA kernels in the GPU task in this tutorial.

A task has to be registered on processors before the Realm runtime can launch it. 
To register a task, the static method `Processor::register_task_by_kind` is used shown as follows. 
```c++
Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                 MAIN_TASK,
                                 CodeDescriptor(main_task),
                                 ProfilingRequestSet(),
                                 0, 0).wait();
```
This function takes several parameters:

- `target_kind` - describes which kind of processor the task will be launched on. We will introduce the processor API in the next example.
- `global` - if set to true, the task is visible on all nodes. It is noted that in this example, `register_task_by_kind` is called from
the main function, which is performed by every application process when running with mpirun, thus, it is still visible on all nodes even `global` is false. 
However, when calling `register_task_by_kind` from a single task, we need to set the `global` to true if we need to make it visible on all nodes.
- `TaskFuncID` - the task ID we defined in the enumeration.
- `CodeDescriptor` - an object that describes a blob of code as a callable function.
In this case, the implementation of the `MAIN_TASK` is `main_task`.
- `user_data` and `user_data_len` - the data passed into the task (the 3rd and 4th paramters of the task implementation). 

`register_task_by_kind` is an asynchronous function that does not guarantee that task registration is done after it returns. 
For this reason, it returns an `Event` object, allowing us to wait for completion explicitly. The usage of events is introduced in the following tutorials.
As mentioned before, Realm allows a task to have multiple implementations. When the task is launched, Realm automatically selects the appropriate implementation based on the processor where it is being executed.

## Launching Tasks

Before launching a task, we need to pick a processor. In this example, we select the first CPU core, 
GPU, and OpenMP processor to launch the CPU, GPU and OpenMP tasks, respectively. An example of selecting the first CPU
core is shown as follows. We will introduce the `Machine` API in the next tutorial. 
```c++
Processor p = Machine::ProcessorQuery(Machine::get_machine())
  .only_kind(Processor::LOC_PROC)
  .first();
``` 

In the area of high-performance computing, most distributed programs start by invoking a main function across a number of parallel processes
concurrently, in what is known as the Single-Program-Multiple-Data (SPMD) execution model. To transition from the SPMD-style execution model
to the task-based model employed by Realm, the `collective_spawn` method is the most expedient way to bridge this gap.
In this example, the `MAIN_TASK` is launched using the `collective_spawn` method, as seen below:
```c++
Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
```
The main task is not SPMD-style, and now
the program is transitioned from the SPMD model into the task-based one. Additionally, Realm provides the `collective_spawn_by_kind` method, 
which can be used to launch an SPMD task where each process launches one task.

Within the main task, we use the `spawn` method of the Processor object to launch the `HELLO_TASK` on the selected CPU, GPU and 
OpenMP processor, respectively. An example of spawning the `HELLO_TASK` on the CPU processor is shown below:
```c++
Event cpu_e = cpu.spawn(HELLO_TASK, NULL, 0);
```
Like `register_task_by_kind`, the `spawn` and `collective_spawn` are
also asynchronous functions that return an `Event` object. Then we can either
invoke the `wait` method to wait for the completion of the task or pass it as the pre-condition of other Realm
operations. We will introduce more details about Realm events in the following tutorial. 

It is worth mentioning that there is no task hierarchy in Realm, so the completion of a Realm task does not imply that all its sub-tasks 
have also been completed. If the cumulative property is needed, users need to implement it explicitly. For example, to ensure that all 
`HELLO_TASK` are completed before exiting the `MAIN_TASK`, the `wait` method is used.
```c++
launch_task(p).wait();
```

## Launching Tasks without Using collective_spawn

The `HELLO_TASK` can also be launched from the main function without using the `collective_spawn`.
To achieve it, we need to mimic the `collective_spawn` behavior by explicitly picking a process, such as the rank 0, 
to launch the hello world tasks using the `spawn` method. 
```c++
Processor local_proc = Machine::ProcessorQuery(Machine::get_machine())
  .only_kind(Processor::LOC_PROC).local_address_space()
  .first();
if (local_proc.address_space() == 0) {
  Event e = launch_task(p);
  rt.shutdown(e);
}
```
However, it is generally recommended to use `collective_spawn` to launch a main task and then `spawn` tasks within it.

## Shuting Down Runtime

At the end of a Realm program, it is necessary to shut down the runtime using the `shutdown` and `wait_for_shutdown method`s. 
In this example, we instruct the runtime to initiate `shutdown` as soon as the `MAIN_TASK` or all `Hello_Task` are finished, respectively. 
While the `wait_for_shutdown` method must be called by all processes, it is not necessary for `shutdown`.
However, if the `shutdown` is called from all processes, the pre-conditional event must be identical across all processes. 
Therefore, in this program, the `shutdown` is called from all processes when using the `collective_spawn`, but only on rank 0 without
`collective_spawn`.
