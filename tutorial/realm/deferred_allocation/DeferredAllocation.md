# Deferred Allocation

## Introduction
In a previous tutorial, we covered how Realm's programming model
relies on deferred execution, allowing operations to be performed
asynchronously. This model also allows resources, 
such as memory allocations, to be managed in a deferred manner.

While it may seem unnecessary to defer memory allocation, it is always
safe to execute an allocation request immediately since 
no prior operations can utilize the memory resource being allocated.
However, if a task issuing the requests has advanced far ahead of
actual execution, immediate execution of allocation requests can
greatly increase the lifetime of the allocation, leading to the wastage
of memory resources. By deferring the allocation request until the
first user of the allocation is able to execute, the lifetime of the
allocation is minimized.

Here is a list of covered topics:

* [Chaining Allocations](#chaining-allocations)

## Chaining Allocations
In this tutorial, we demonstrate the deferred allocation strategy by
discussing a simple application that processes a number of data
batches by scheduling the execution far out.

The two tasks `writer_task` and `reader_task` require a valid set of region
instances to be available at the execution time:

```c++
void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  verify<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  verify<InstanceLogicalLayout2, long long, double>(task_args.inst,
                                                    task_args.bounds, FID2,
                                                    /*add=*/2);
}

void writer_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  update<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  update<InstanceLogicalLayout2, long long, double>(
      task_args.inst, task_args.bounds, FID2, /*add=*/2);
}
```

To begin, batch allocations are queued up at the beginning of the
program and linked to processing tasks using a
`create_event`:

```c++
  Event prev_event = start_event;
  for (size_t batch_idx = 0; batch_idx < ExampleConfig::num_batches;
       batch_idx++) {
    RegionInstance inst;
    Event create_event = RegionInstance::create_instance(
        inst, memories.front(), bounds,
        field_sizes, 1, ProfilingRequestSet(), prev_event);

    TaskArguments task_args{bounds, inst};
    prev_event = p.spawn(
        READER_TASK, &task_args, sizeof(TaskArguments),
        p.spawn(WRITER_TASK, &task_args, sizeof(TaskArguments), create_event));
    inst.destroy(prev_event);
  }
```

Each allocation must be deleted before the next batch can proceed,
but only after data processing is completed. Therefore, the deletion
operation (`inst.destroy(...)`) is preconditioned with the `prev_event`.
This deferred strategy schedules operations in a "future" and ensures
that the next chain is allocated only when needed.

The program starts by triggering the `start_event`,
writes data to the allocated region instance, reads it, verifies
its validity, and then deletes the allocated data before the next
batch begins.
