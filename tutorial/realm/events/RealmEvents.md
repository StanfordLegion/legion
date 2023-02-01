## Realm Events

### Introduction
In this tutorial, we examine the concept of Realm events and provide 
clear example on how to express control dependencies between operations effectively.

Events form the backbone of Realm's programming model, describing the dependencies between operations.
They are created by the runtime and can be used as a pre or post condition for any Realm operation.
Events provide an interface that allows the runtime to effectively manage program execution, ensuring efficient and safe operation.

### Creating Events
There are two types of events: internal runtime events and user events. Internal runtime events are generated
automatically by the runtime system and typically occur in response to an operation such as a `task launch` at line 34.
User events, on the other hand, can be manually triggered from application code, as demonstrated at line 31 and 37.
These events are generally similar in nature to internal runtime events, but allow for greater control over when and how events are triggered.

### Createing Control Dependencies
In this program, we launch several `reader_tasks` that are responsible for printing an integer value `x`. Each task launch is a
non-blocking call and the reader task will not start running until the user event is triggered (line: 34).
Since the launches are asynchronous, they return an internal event handle which can be used to guarantee that the task has completed (line: 37).
To simplify the process, we can merge all the events together and use a single `wait` call (line: 40).
This will cause the calling thread to block until all the events have finished.

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
32     UserEvent user_event = UserEvent::create_user_event();
33 
34     Event task_event = p.spawn(READER_TASK, &x, sizeof(int), user_event);
35 
36     events.push_back(task_event);
37     user_event.trigger();
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
