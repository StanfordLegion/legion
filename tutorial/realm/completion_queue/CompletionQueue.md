# Realm Completion Queue

## Introduction
In some scenarios, we want to test the completion/triggerness of any event
within a group of events, similar to `MPI_Testany`. 
Realm provides a scalable mechanism named `CompletionQueue` that
provides an event notification whenever any of the events in the CompletionQueue have triggered.
Despite its name, the data structure itself does not really act like a queue with 
push/pop semantics. It is more like a set of events, and notifications 
when any event in the set has been triggered.
Therefore, rather than testing each
event in order using the `has_triggered` method, events can be pushed into a 
CompletionQueue, and completed/triggered events can be popped one by one or in batches.
In this tutorial, we will demonstrate how to use the CompletionQueue.

Here is a list of covered topics:

* [Creating a CompletionQueue](#creating-a-completionqueue)
* [Using CompletionQueue to Track Triggerness of Events](#using-completionqueue-to-track-triggerness-of-events)

## Creating a CompletionQueue

A CompletionQueue can be created by calling:
```c++
CompletionQueue::create_completion_queue(size_t max_size);
```
The `max_size` parameter specifies the maximum number of triggered 
events the CompletionQueue can hold. If the `max_size` is set to 0,
the queue can grow arbitrarily, but at the cost of additional overhead.

## Using CompletionQueue to Track Triggerness of Events

First, we start with the basic APIs of the CompletionQueue.
An event can be added to the CompletionQueue using `add_event`. 
It is important to note that the event won't be added to the queue until triggered.
To test whether events added by `add_event` have been triggered and 
are available in the CompletionQueue, we can use the `get_nonempty_event`.
This method returns an event, which will be triggered once 
at least one triggered event is inside the CompletionQueue. 
The `get_nonempty_event` returns a `NO_EVENT` if the CompletionQueue is not empty. 
Let's take the following code for an example, the `nonempty_test` variable is `NO_EVENT` 
because the event `e` added is triggered and becomes available in
the CompletionQueue after `nonempty.wait()` is returned.
```c++
Event nonempty = completion_queue.get_nonempty_event();
  ...
Event e = p.spawn(WORKER_TASK, &worker_task_args, sizeof(WorkerTaskArgs));
completion_queue.add_event(e);
nonempty.wait();
Event nonempty_test = completion_queue.get_nonempty_event();
assert(nonempty_test == Event::NO_EVENT);
  ...
completion_queue.pop_events(&popped[0], 1);
nonempty_test = completion_queue.get_nonempty_event();
assert(nonempty_test != Event::NO_EVENT);
```
Once the event `e` is popped out of the CompletionQueue using `pop_events`, 
the CompletionQueue becomes empty, and therefore, the new value of `nonempty_test` 
returned by `get_nonempty_event` is no longer `NO_EVENT`.

It is important to note that the `get_nonempty_event` will always return an event, 
if the caller is not on the same process/rank as 
the one where the CompletionQueue is created, even if the CompletionQueue
is not empty.

Then, we demonstrate how to use the CompletionQueue to track
the completion of a batch of events. First, we launch a 
batch of `WORKER_TASK` and add the events returned by the task spawners 
to the CompletionQueue. Second, a `CLEANCQ_TASK` is 
launched to collect all triggered events, but this task is not launched 
until at least one event becomes available in the CompletionQueue. 
Finally, we pass the event of the `CLEANCQ_TASK` spawner to the `destroy` 
method of the CompletionQueue, so the CompletionQueue will be destroyed 
once the `CLEANCQ_TASK` is finished. It is worth noting that, similar to an event, 
the CompletionQueue can be passed as a task argument. Additionally, 
the APIs of the CompletionQueue are task-safe, meaning that they can 
be called concurrently from different tasks.

## Destroying a CompletionQueue

A CompletionQueue can be destroyed by calling the `destroy`, and
it also accepts a pre-condition event.
