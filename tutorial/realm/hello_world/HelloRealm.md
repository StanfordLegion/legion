# Realm Hello World

Realm programs are written in C++ and have a standard
application structure that consists of an initialization, execution
(running tasks) and de-initialization stages.

```c++
 1 #include "realm.h"
 2 #include "realm/cmdline.h"
 3 
 4 using namespace Realm;
 5 
 6 Logger log_app("app");
 7 
 8 enum {
 9   HELLO_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
10 };
11 
12 void hello_task(const void *args, size_t arglen, const void *userdata,
13                     size_t userlen, Processor p) {
14   log_app.info() << "Hello World!";
15 }
16 
17 int main(int argc, char **argv) {
18   Runtime rt;
19 
20   rt.init(&argc, &argv);
21   rt.register_task(HELLO_TASK, hello_task);
22 
23   Event e = rt.collective_spawn_by_kind(Processor::LOC_PROC, HELLO_TASK,
24                                         /*args=*/nullptr,
25                                         /*arglen=*/0, /*one_per_node=*/true);
26   rt.shutdown(e);
27   rt.wait_for_shutdown();
28 
29   return 0;
30 }
```

## Realm Namespaces

Each Realm class has its own C++ header file. All classes are 
aggregated in `realm.h` and can be included in an application for
convenience. Each class definition is placed in a `Realm` namespace to
avoid naming conflicts.


## Basic Program Structure

1. `Line 20` initializes a singleton runtime object. The
initialization must be performed by every application process. After
initialization is complete, the runtime remains mostly idle (except system
status checks) and waits for the task launches.

2.  `Line 21` registers a task with a `HELLO_TASK` ID. It
enables us address the task by the ID anywhere in the application.

3. Before a `HELLO_TASK` can run we need to decide which `Processor` it will
be executed on. We are choosing to run the task on a local processor
(`LOC_PROC`) for simplicity which defines CPU processors that request
a dedicated core and user threads when possible. We are going to
discuss realm's machine model in detail later in a dedicated tutorial.

4. `Line 23` spawns the task with a `collective spawn` method. This
method ensures that only a single instance of the task is launched
despite a number of processors over which runtime is distributed.

5. `Line 26` tells the runtime to initiate the shutdown as soon as the
event attributed to the `HELLO_TASK` has triggered.

