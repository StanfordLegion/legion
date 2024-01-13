---
layout: page
permalink: /tutorial/realm/barrier.html
title: Realm Barrier
---

## Introduction

A `Barrier` is a special kind of `Event` (notice that it inherits from the base Event class) that 
requires multiple arrivals in order for it to trigger. Additionally, unlike normal events, 
which only have a single generation, a barrier has multiple generations, named phases. 
Therefore, a barrier can be advanced to the next generation and performed arrivals 
even before the previous generation has been triggered. However, a barrier has to be explicitly destroyed. 
In this example, there are two types of tasks, reader and writer. 
During each iteration, every writer task generates an integer value. Subsequently, 
the collective sum of all these integers is calculated, and the resulting sum is then accessed by the reader tasks. 
Thus, the writer and reader tasks take turns to execute.
We illustrate how to use `Barrier` to go back and forth between 
reader and writer tasks of different iterations.

Here is a list of covered topics:

* [Creating Barriers](#creating-barriers)
* [Creating Reductions](#creating-reductions)
* [Synchronizing Tasks with Barriers](#synchronizing-tasks-with-barriers)
* [Performance Considerations](#performance-considerations)
* [Limitations](#limitations)

## Creating Barriers

As introduced in the first section, each writer task performs a reduction operation on its local integer value. 
The reader tasks can not read the reduction value until all writer tasks have completed writing. 
This kind of synchronization can be implemented by `Barrier`. 
In this example, we need to create two barriers, a `writer_barrier` and `reader_barrier`,
to synchronize writer and reader tasks, respectively. 
To create a barrier, we need to pass the number of concurrent barrier tasks into the `create_barrier` function.
The following code demonstrates creating a `reader_barrier`, where we need to pass the number of participants:

```c++
Barrier reader_barrier = Barrier::create_barrier(TestConfig::num_readers);
```

## Creating Reductions

The summation operation of writer tasks can be implemented using a reduction,
which is declared as `ReductionOpIntAdd`. A reduction operator requires two functions, 
`apply` combines a left-hand side type with a right-hand side type into a new left-hand side type, and `fold`
combines two right-hand side types into a new right-hand side type, as well as an `identity`, a unique integer
for Realm to recognize the reduction. The reduction operator needs to be registered by calling `register_reduction`
before launching the main task, shown as follows:

```c++
rt.register_reduction<ReductionOpIntAdd>(REDOP_ADD);
```

The communication pattern of barrier and all reduction are similar; therefore, in the MPI world, these two
routines have similar implementation. 
Inspired by this, Realm extends the barrier to perform all-reduction-like computations, where each task can pass a number to
the barrier. When reaching the synchronization point, the reduction value of all numbers is available to tasks that
hold the barrier. 
To perform reduction with a barrier, we can attach a reduction operator to the `create_barrier` function.
The `init_value` tells the initial value of the parameter that the reduction operation is performed on. 

```c++
Barrier writer_barrier = Barrier::create_barrier(TestConfig::num_writers, REDOP_ADD,
                                                 &init_value, sizeof(init_value));
``` 
Later, the result of the reduction can be retrieved by `get_result`.
```c++
int result = 0;
bool ready = writer_b.get_result(&result, sizeof(result));
``` 

## Synchronizing Tasks with Barriers

The lifetime of a Barrier consists of one or more phases. Each phase of a Barrier defines a synchronization point, 
where blocks of code wait for other tasks to catch up before proceeding. Tasks can arrive at the barrier 
but defer waiting on the phase synchronization point
by calling `arrive`. The same or other tasks can later block on the phase synchronization point by calling `wait`.
We can utilize the `arrive` and `wait` to synchronize different tasks. 

When a writer finishes work, `arrive` is called to inform the runtime that it arrives at 
the synchronization point. 
Since we use reduction to accumulate the integer produced by 
each writer task, each writer task can pass the integer to the `arrive` function to perform the reduction.
```c++
writer_b.arrive(1, Event::NO_EVENT, &reduce_val, sizeof(reduce_val));
```

On the other side, a reader task calls `wait` to wait until all writer tasks finish calling `arrive`.
The `wait` is blocked until the `arrive` is called N times, where the N matches the number used for `create_barrier`.
The same mechanism is used by writer tasks to continue working after all reader tasks finish reading. 

```c++
writer_b.wait();
```

When a `Barrier` finishes a synchronization point after the `wait` is returned, we can call `advance_barrier`
to enter the next phase. In the example, we need to synchronize reader/writer tasks multiple times (one time per iteration), 
so after an iteration is done, `advance_barrier` is called to go into the next iteration. 
```c++
writer_b = writer_b.advance_barrier();
reader_b = reader_b.advance_barrier();
```

## Performance Considerations

Even though tasks can be preempted, a Processor can only execute one task at a time. Therefore, to increase
task-level parallelism, it is preferred to launch concurrent tasks onto different Processors. 
In this example, we reverse the first 4 CPU Processors for the main task and the three reader tasks, and then
we vary the number of writer tasks and `-ll:cpu` to show how the processor assignment affects the overall performance. 

a. Run with 4 writer tasks on the same processor. 

```
$ ./barrier -ll:cpu 5 -nw 4 -ll:force_kthreads
[0 - 7fa670922800]    0.069117 {3}{app}: start top task on Processor 1d00000000000001, tid 3638294
[0 - 7fa66bffe800]    0.069400 {3}{app}: start writer task 0 on Processor 1d00000000000005, tid 3638298
[0 - 7fa67071e800]    0.069436 {3}{app}: start reader task 0 on Processor 1d00000000000002, tid 3638295
[0 - 7fa67051a800]    0.069511 {3}{app}: start reader task 1 on Processor 1d00000000000003, tid 3638296
[0 - 7fa670316800]    0.069530 {3}{app}: start reader task 2 on Processor 1d00000000000004, tid 3638297
[0 - 7fa66b5ea800]    1.070073 {3}{app}: start writer task 1 on Processor 1d00000000000005, tid 3638303
[0 - 7fa66b3e6800]    2.070547 {3}{app}: start writer task 2 on Processor 1d00000000000005, tid 3638304
[0 - 7fa66b1e2800]    3.070931 {3}{app}: start writer task 3 on Processor 1d00000000000005, tid 3638305
...
[0 - 7fce75bec7c0]   16.061451 {3}{app}: Total time 0.165409(s)
```
The `-ll:force_kthreads` force Realm always to use kernel threads (pthread in Linux), even though 4 writer tasks are
dispatched onto different threads; since they belong to the same processor, Realm does not execute them concurrently. 

The following is the result without forcing to use kernel threads. 

```
$ ./barrier -ll:cpu 5 -nw 4
[0 - 7f27a0afc800]    0.069120 {3}{app}: start top task on Processor 1d00000000000001, tid 3638275
[0 - 7f27a0098800]    0.069369 {3}{app}: start writer task 0 on Processor 1d00000000000005, tid 3638279
[0 - 7f27a0a68800]    0.069404 {3}{app}: start reader task 0 on Processor 1d00000000000002, tid 3638276
[0 - 7f27a00a4800]    0.069457 {3}{app}: start reader task 2 on Processor 1d00000000000004, tid 3638278
[0 - 7f27a00b0800]    0.069462 {3}{app}: start reader task 1 on Processor 1d00000000000003, tid 3638277
[0 - 7f27a0098800]    1.069763 {3}{app}: start writer task 1 on Processor 1d00000000000005, tid 3638279
[0 - 7f27a0098800]    2.070110 {3}{app}: start writer task 2 on Processor 1d00000000000005, tid 3638279
[0 - 7f27a0098800]    3.070357 {3}{app}: start writer task 3 on Processor 1d00000000000005, tid 3638279
...
[0 - 7fd3d6fa27c0]   16.052200 {3}{app}: Total time 0.165527(s)
```

It is noted that all the writer tasks are running on the same kernel thread, even though Realm create green threads
to minimize the creation and context switch overhead of kernel threads; since all task are launched onto the 
same processor, they can not be parallelized.

b. Run with 4 writer tasks on different processors. 

```
$ ./barrier -ll:cpu 8 -nw 4
[0 - 7f4dc9f44800]    0.069260 {3}{app}: start top task on Processor 1d00000000000001, tid 3638318
[0 - 7f4dc80d1800]    0.069490 {3}{app}: start writer task 0 on Processor 1d00000000000005, tid 3638322
[0 - 7f4dc80b9800]    0.069533 {3}{app}: start writer task 2 on Processor 1d00000000000007, tid 3638324
[0 - 7f4dc80c5800]    0.069560 {3}{app}: start writer task 1 on Processor 1d00000000000006, tid 3638323
[0 - 7f4dc80e9800]    0.069596 {3}{app}: start reader task 1 on Processor 1d00000000000003, tid 3638320
[0 - 7f4dc80dd800]    0.069607 {3}{app}: start reader task 2 on Processor 1d00000000000004, tid 3638321
[0 - 7f4dc9eb0800]    0.069593 {3}{app}: start reader task 0 on Processor 1d00000000000002, tid 3638319
[0 - 7f4dc80ad800]    0.069596 {3}{app}: start writer task 3 on Processor 1d00000000000008, tid 3638325
...
[0 - 7faf67fe97c0]    4.065398 {3}{app}: Total time 0.042038(s)
```

All writer tasks occupy different processors, so we achieve 4x speedup over the previous configuration. 

## Limitations

There are two limitations when using barriers:

- A barrier has a maximum number of phases/generations, defined as `MAX_PHASES`. If a barrier is advanced past its last phase
then `advance_barrier` returns `NO_BARRIER`. 
- A barrier's phases/generations must be triggered in order. So you might advance to the next generation of a barrier and 
start doing arrivals on it without arriving on the previous generation. Even if all the arrivals for this next generation
have been performed, the generation will not trigger because the previous generation has not triggered. For example,
the following code leads to deadlock because one generation is skipped in each loop. 

```c++
for (int i = 0; i < 10; i++) {
  barrier.arrive(1);
  barrier.wait();
  barrier = barrier.advance_barrier();
  barrier = barrier.advance_barrier(); // this generation is ignored.
}
```
