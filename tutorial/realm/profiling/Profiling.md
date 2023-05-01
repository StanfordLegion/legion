# Realm Profiling

## Introduction

After developing a functional Realm application, it is often necessary to profile 
and tune its performance. Realm provides a built-in profiling mechanism allowing 
users to query profiling information about all Realm operations. This tutorial 
will explain how to profile a task, instance creation, and copy operation. 
In addition, we also provide a sophisticated use case of utilizing
profiling results to make load balance decisions for task launchers.


Here is a list of covered topics:

* [Creating Profiling Requests](#creating-profiling-requests)
* [Adding Performance Metrics](#adding-performance-metrics)
* [Collecting Profiling Results](#collecting-profiling-results)
* [Advanced Use Case: Load Balancing Task Launcher](#advanced-use-case-load-balancing-task-launcher)
* [References](#references)

## Creating Profiling Requests

Realm profiling allows clients to profile most of the operations in Realm, such as tasks and 
instance creation. To tell Realm which profiling information
we want to collect, we need to create a `ProfilingRequestSet` object. Then we can use the
`add_request` method to create a `ProfilingRequest` object and add it to the ProfilingRequestSet
created before. 
```c++
ProfilingRequestSet task_prs;
task_prs.add_request(profile_proc, COMPUTE_PROF_TASK, &task_result, sizeof(ComputeProfResultWrapper))
```
Realm profiling adopts a callback design, where a profiling task is launched
to report profiling results when an operation is completed. Therefore, as shown on the code above, 
we need to specify the task (`COMPUTE_PROF_TASK`) and processor (`profile_proc`) to be used to launch the task for the ProfilingRequest.

## Adding Performance Metrics

Once the ProfilingRequest is created, we can add performance metrics that need to be profiled into it. 
Realm supports a variety of performance metrics that can be specified by the `add_measurement`
method of the ProfilingRequest object. 
```c++
task_prs.add_request(profile_proc, COMPUTE_PROF_TASK, &task_result, sizeof(ComputeProfResultWrapper))
  .add_measurement<ProfilingMeasurements::OperationTimeline>()
```
The following metrics are used in the tutorial.

- `OperationTimeline` tracks the timestamps of different stages of an operation. In this example,
we use it for tasks and copy operations.
- `OperationProcessorUsage` contains the processor where the task is launched. 
- `InstanceTimeline` tracks the timeline of an instance, including when the instance is created, 
ready for use, and destroyed. 
- `InstanceMemoryUsage` includes the memory where the instance is located and the size of the instance. 
- `OperationCopyInfo` tracks the transfer details for copy and fill operations. 
- `OperationMemoryUsage` tracks the memory usage for copy and fill operations.

Realm supports many other kinds of `ProfilingMeasurements`, and they can be found in the [profiling header file](#profiling-header-file).
Once the metrics are set, we can pass the ProfilingRequestSet object to Realm APIs that
support profiling. For example, the following code is an example of setting the `ProfilingRequestSet` for tasks.
```c++
worker_procs[0].spawn(COMPUTE_TASK, &compute_task_args, sizeof(ComputeTaskArgs), task_prs).wait();
```

## Collecting Profiling Results

As mentioned earlier, after an operation is completed, a profiling task is launched where
profiling results can be collected. Now we demonstrate how to collect
the results of the task launched in the `main_task`. 
1. To gather profiling results of a task, a `ProfilingResponse` object is created by passing the `args` and `arglen`.
```c++
ProfilingResponse resp(args, arglen);
```
It is worth mentioning that for each `ProfilingRequest`, there will be one and only 
one `ProfilingResponse` reported back to the client, such that the client can use 
this feature to make sure all the profiling responses are collected.

2. The argument of the profiling task can be retrieved by the `user_data` method of the
ProfilingResponse object. 
```c++
const ComputeProfResultWrapper *result = static_cast<const ComputeProfResultWrapper *>(resp.user_data());
```
3. Then, the `get_measurement` method can be used to
obtain the results of performance metrics requested by the `add_measurement` method described
in the previous section.
```c++
ProfilingMeasurements::OperationTimeline timeline;
if(resp.get_measurement(timeline)) {
  metrics->ready_time = timeline.ready_time;
  metrics->start_time = timeline.start_time;
  metrics->complete_time = timeline.complete_time;
}
```  

Sometimes, we want to be notified when
the results are ready inside the task where the operation needing profiling is launched. For 
example, in this tutorial, we want to print the results in the `main_task`.
To achieve this, we can create a `UserEvent`, pass it as the argument of a profiling task 
via the `add_request` method,
explicitly trigger it inside the profiling task, and then we can either
wait for the event as seen in this example explicitly or push it into a CompletionQueue and query it later. 

## Advanced Use Case: Load Balancing Task Launcher

Real-world applications often have tasks with different costs, and the same task may run at different 
speeds on different processors. Therefore, a simple round-robin task scheduling may not be sufficient. 
This section explores the benefits of empirical task distribution using profiling 
results.
We repeatedly launch a batch of tasks on a group of worker processors, where each task has a 
different cost. A standard approach is to profile the tasks offline to obtain their execution time and 
compute a static load-balancing strategy. In this program, we start with a round-robin strategy.
Then we profile the tasks and use the profiling results to pick the best worker processor 
for future iterations dynamically.

Here is the performance comparison between round-robin and profiling guided scheduling:
```
$ ./profiling -ll:cpu 6
[0 - 7f59ba4f3800]    0.189922 {3}{app}: With round-robin 38358 us
[0 - 7f59ba4f3800]    0.212061 {3}{app}: With profiling 21299 us
```
It turns out the profiling guided one achieves better task load balance.

## References

<div id="profiling-header-file"></div>
[1]: [profiling header file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/profiling.h)
