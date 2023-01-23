## Realm Events

### Introduction
This tutorial discusses Realm events.

### General Events
Events are the foundation of Realm's programming model where
they describe dependencies between operations. An event is typically created by
the runtime and can be used as either a pre or post condition for any
Realm operation. Events offer an interface which enables the runtime to
effictively manage a program execution.

### Events Interface
TODO: Update line numbers and describe an interface.
```c++
 1 namespace Realm {
 2     ...
 3     class REALM_PUBLIC_API Event {
 4     public:
 5       ...
 8       bool has_triggered(void) const;
 9       ...
10       void wait(void) const;
15       ...
16       void subscribe(void) const;
18       ...
20       static Event merge_events(const std::vector<Event>& wait_for);
21       ...
22     };
23 }

```

### User Events
General events cannot be triggered by a user and therefore runtime
offers a separate event type `UserEvent`. A user event has all the
properties of a general event and it (unlike to a general event ) can be 
triggered by the application code via the following interface:

```c++
23  namespace Realm {
24     class REALM_PUBLIC_API UserEvent : public Event {
25     public:
26       static UserEvent create_user_event(void);
27       ...
28       void trigger(Event wait_on = Event::NO_EVENT,
29                    bool ignore_faults = false) const;
30       ...
31     };
32    ...
33 }

```

### Example
TODO: Update this example with the latest code.

In the following example we demonstrate how events can be used in order
to create a control dependency. The application creates several reader
tasks `reader_task` at `line:68` that do nothing more but read an
input argument `x`. We create a user event `event1` at `line:32` which
used as a precondition for any reader task launch which means that before
`event1` is triggered `(line:37)` none of the reader tasks will start the
execution. Each task returns an internal event `(line:34)` that is
stored in the event pool `line:36`. The `top_level_task` waits
until all the events in the event pool have been triggered by calling
`Event::merge_events(events).wait()` at `line:40`.

```c++
 1 #include <realm.h>
 2 #include <realm/cmdline.h>
 3 
 4 using namespace Realm;
 5 
 6 enum
 7 {
 8   TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
 9   READER_TASK,
10 };
11 
12 Logger log_app("app");
13 
14 namespace ProgramConfig {
15   size_t num_tasks = 2;
16 };
17 
18 void reader_task(const void *args, size_t arglen, const void *userdata,
19                  size_t userlen, Processor p)
20 {
21   int x = *reinterpret_cast<const int *>(args);
22   log_app.info() << "reader task: proc=" << p << " x=" << x;
23 }
24 
25 void top_level_task(const void *args, size_t arglen, const void *userdata,
26                     size_t userlen, Processor p)
27 {
28   int x = 7;
29 
30   std::vector<Event> events;
31   for(size_t i = 0; i < ProgramConfig::num_tasks; i++) {
32     UserEvent event1 = UserEvent::create_user_event();
33 
34     Event task_event = p.spawn(READER_TASK, &x, sizeof(int), event1);
35 
36     events.push_back(task_event);
37     event1.trigger();
38   }
39 
40   Event::merge_events(events).wait();
41 
42   log_app.info() << "Completed successfully";
43 
44   Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
45 }
46 
47 int main(int argc, const char **argv)
48 {
49   Runtime rt;
50 
51   rt.init(&argc, (char ***)&argv);
52 
53   Processor p = Machine::ProcessorQuery(Machine::get_machine())
54                     .only_kind(Processor::LOC_PROC)
55                     .first();
56 
57   if(!p.exists()) {
58     p = Machine::ProcessorQuery(Machine::get_machine()).first();
59   }
60 
61   assert(p.exists());
62 
63   Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
64                                    CodeDescriptor(top_level_task),
65                                    ProfilingRequestSet())
66       .external_wait();
67 
68   Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK,
69                                    CodeDescriptor(reader_task),
70                                    ProfilingRequestSet())
71       .external_wait();
72 
73   rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
74 
75   int ret = rt.wait_for_shutdown();
76 
77   return ret;
78 }
```
