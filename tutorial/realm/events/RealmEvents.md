# Realm Events

## Introduction
Realm is a fully asynchronous, event-based runtime, and events form
the backbone of Realm's programming model, describing the dependencies
between operations. Realm operations are deferred by the runtime,
which returns an event that triggers upon completion of the operation.
These events are created by the runtime and can be used as pre- or
post-conditions for other operations. Events provide a mechanism that
allows the runtime to efficiently manage asynchronous program
execution, offering opportunities to hide latencies when
communications are required.

In this tutorial, we'll demonstrate how to use events to take 
advantage of Realm's deferred execution model when writing asynchronous 
applications.

Here is a list of covered topics:

* [Events Basics](#events-basics)
* [Creating Events](#creating-events)
* [Triggering Events](#trigerring-events)
* [Creating Control Dependencies](#creating-control-dependencies)
* [References](#references)

## Events Basics
Usually, Realm creates Events as part of handling application requests
for asynchronous operations. An Event is a lightweight handle
that can be easily transported around the system. The node that
creates an Event owns it, and the space for these handles is statically
divided across all nodes by including the node ID in the upper bits of
the handle. This design ensures that any node can create new handles
without the risk of collision or requiring inter-node communication.

The basic event is implemented as a distributed object and spread
across one or several nodes. Each node uses the event handle to look up
their piece of the object as needed. This lookup uses a monotonic data
structure that allows wait-free queries even when updates are being
performed.

When a new Event is created, the owning node allocates a data
structure to track its state, which is initially `untriggered` but
will eventually become triggered or poisoned. The data structure also
includes a list of `local waiters` and `remote waiters`. Local waiters are
dependent operations on the owner node, and remote waiters are other nodes
that are interested in the Event (event dependencies).

## Creating Events
In this program, we launch several tasks (`reader_task_0` and
`reader_task_1`) responsible for
printing an integer value `x`:

```c++
void reader_task_0(const void *args, size_t arglen, const void *userdata,
                   size_t userlen, Processor p) {
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 0: proc=" << p << " x=" << task_args->x;
}

void reader_task_1(const void *args, size_t arglen, const void *userdata,
                   size_t userlen, Processor p) {
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 1: proc=" << p << " x=" << task_args->x;
}
```

Each task launch is a non-blocking  asynchronous call that returns an
internal event  handle such as `reader_event0` and `reader_event1`.
Once created, the Event handle can be passed around through 
task arguments or shared data structures and eventually used as a 
pre- or post-condition for operations to be executed on other nodes.

When a remote node makes the first reference to `task_event`, it 
allocates the same data structure, sets its state to `untriggered`, and
adds the dependent operation to its own local waiter list. Then, an 
event subscription active message is sent to the owner node to 
indicate that the remote node is interested and should be added to 
the list of remote waiters, so it can be informed when `task_event` 
triggers. Any additional dependent operations on a remote node are 
added to the list of local waiters without requiring communication 
with the owner node. When `task_event` eventually triggers, the owner 
node notifies all local waiters and sends an event trigger message to 
each subscribed node on the list of remote waiters. If the owner node 
receives additional subscription messages after it has been triggered, 
it immediately responds to the new subscribers with a trigger message 
as well.

## Triggering Events
An event can be triggered from any node, not necessarily the owner node.
One common scenario in which this happens is with `UserEvent`. These are
created and triggered from the application code, where
we create `user_event` to start an operation:

```c++
  UserEvent user_event = UserEvent::create_user_event();
```


User events offer greater flexibility in building the event graph by allowing
users to connect different parts of the graph independently. However, it is
important to note that using user events carries the risk of creating cycles,
which can cause the program to hang. Therefore, it is the user's responsibility
to avoid creating cycles while leveraging user events.

When a `user_event` is triggered on a node that does not own it, a
trigger message is sent from the trigger node to the owner node, which then
forwards the message to all other subscribed nodes. If the triggering 
node has any local waiters, it immediately notifies them without 
sending a message back to the owner node. Although triggering a remote
event incurs a latency of at least two active message flight times, it 
limits the number of active messages required per event trigger to 
`2*N - 2`, where `N` is the number of nodes interested in the event.

## Creating Control Dependencies
We will now demonstrate how to establish a control dependency using
events, by making `reader_task_1` dependent on the completion of
`reader_task_0`. We achieve this by passing `reader_event0` to the
task invocation procedure:

```c++
  Event reader_event0 =
      p.spawn(READER_TASK_0, &task_args, sizeof(TaskArgs), user_event);

  Event reader_event1 =
      p.spawn(READER_TASK_1, &task_args, sizeof(TaskArgs), reader_event0);
```

Often, it is necessary to spawn multiple tasks simultaneously and 
express a collective wait using a single event handle. To illustrate 
this, the program runs `num_tasks`, stores the events produced by
`reader_task_1` into an `events` vector and combines them by calling:

```c++
Event::merge_events(events).wait()
```

## References
1. [Event header file](https://github.com/StanfordLegion/legion/blob/stable/runtime/realm/event.h)
2. [Realm: Performance Portability through Composable Asynchrony](https://legion.stanford.edu/pdfs/treichler_thesis.pdf)
