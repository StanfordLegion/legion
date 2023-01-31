# Region Instances
## Introduction
In this example, we will show you how to manage application data
inside Realm programs. We will create a single `top_level_task` that
will write, read and verify some simple application data stored in an
entity called `RegionInstance`.

## Creating Instance
`RegionInstances` are used to store application
are allocated in a specific memory location. Once a `RegionInstance` is
created, it cannot be moved. If an application requires the data to be
migrated or replicated, then a new RegionInstance must be created and a
`copy` operation must be initiated to move the data between them.

Realm provides a public API, `create_instance`, for creating instance
objects. This call returns an event handle, which can be used as a
precondition for any subsequent Realm operations. Additionally, the
`block_size` arguments allows users to specify the instance layout;
this can be set to 0 for a Structure of Arrays (SOA) layout, 1 for
Array of Structures (AOS) layout, or 2 for a hybrid layout.

At line:79 we create a region instance with `AOS` as the physical
layout. This region encompasses two logical layouts `InstanceLogicalLayout1`
and `InstanceLogicalLayout2`. The `FieldID` of these layouts are supplied
to an instance interface through the `field_sizes` map (line:74).

The `AOS` layout defines that the fields of an element `i` for
`InstanceLogicalLayout1` should be stored before those of elememt
`i+1`. On the other hand, `SOA` (Structure of Arrays) defines that the
fields `x` for all elements of `InstanceLayout1` are stored
consecutively before the fields `y`.

```c++
  class REALM_PUBLIC_API RegionInstance {
  public:
    ...
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const Rect<N,T>& rect,
				 const std::map<FieldID, size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
         ...
				 Event wait_on = Event::NO_EVENT);
    ...
   }
```

## Accessing Data
It may be necessary for a task to directly access data stored in a
region instance. Realm offers a set of accessors such as
`AffineAccessor`, `MultiAffineAccessor` and `GenericAccessor` that
allow users to access individual elements. For example, `AffineAccessor` 
is designed to work only for data with an affine layout and that is
local to the node on which the task is running. `MultiAffineAccessor`
extends the previous accessor by handling instances with multiple
affine pieces, while `GenericAccessor` (as the name suggests) handles
all layouts, both local and remote. In this example, the region instance is created
with an affine layout and stored locally to the `top_level_task`.
Therefore, it is valid to access data with the `AffineAccessor`, as
demonstrated at lines 38 and 50.

TODO: Discuss hybrid layouts. \
TODO: Discuss reduction instances. \
TODO: Discuss deferred allocation.


```c++
  1 #include <realm.h>
  2 #include <realm/cmdline.h>
  3 
  4 using namespace Realm;
  5 
  6 enum
  7 {
  8   TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  9 };
 10 
 11 enum
 12 {
 13   FID1 = 101,
 14   FID2 = 102,
 15 };
 16 
 17 Logger log_app("app");
 18 
 19 struct CreateRegionArgs {
 20   RegionInstance *inst;
 21   Memory memory;
 22   Rect<1, int> bounds;
 23 };
 24 
 25 struct InstanceLogicalLayout1 {
 26   int x;
 27   float y;
 28 };
 29 
 30 struct InstanceLogicalLayout2 {
 31   long long x;
 32   double y;
 33 };
 34 
 35 template <typename FT, typename T0, typename T1>
 36 void update(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add)
 37 {
 38   AffineAccessor<FT, 1, int> accessor(inst, fid);
 39   PointInRectIterator<1, int> pit(bounds);
 40   while(pit.valid) {
 41     accessor[pit.p].x = static_cast<T0>(pit.p.x + add);
 42     accessor[pit.p].y = static_cast<T1>(pit.p.x + add + 1);
 43     pit.step();
 44   }
 45 }
 46 
 47 template <typename FT, typename T0, typename T1>
 48 void verify(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add)
 49 {
 50   AffineAccessor<FT, 1, int> accessor(inst, fid);
 51   PointInRectIterator<1, int> pit(bounds);
 52   while(pit.valid) {
 53     assert(accessor[pit.p].x == static_cast<T0>(pit.p.x + add));
 54     assert(accessor[pit.p].y == static_cast<T1>(pit.p.x + add + 1));
 55     log_app.info() << "p=" << pit.p << " x=" << accessor[pit.p].x
 56                    << " y=" << accessor[pit.p].y;
 57     pit.step();
 58   }
 59 }
 60 
 61 void top_level_task(const void *args, size_t arglen, const void *userdata,
 62                     size_t userlen, Processor p)
 63 {
 64   Rect<1, int> bounds;
 65   bounds.lo = Point<1, int>(0);
 66   bounds.hi = Point<1, int>(7);
 67 
 68   std::vector<Memory> memories;
 69   Machine::MemoryQuery mq(Machine::get_machine());
 70   mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
 71   memories.assign(mq.begin(), mq.end());
 72   assert(!memories.empty());
 73 
 74   std::map<FieldID, size_t> field_sizes;
 75   field_sizes[FID1] = sizeof(InstanceLogicalLayout1);
 76   field_sizes[FID2] = sizeof(InstanceLogicalLayout2);
 77 
 78   RegionInstance inst = RegionInstance::NO_INST;
 79   RegionInstance::create_instance(inst, *memories.begin(), bounds, field_sizes,
 80                                   /*AOS=*/1, ProfilingRequestSet())
 81       .wait();
 82 
 83   update<InstanceLogicalLayout1, int, float>(inst, bounds, FID1, /*add=*/1);
 84   update<InstanceLogicalLayout2, long long, double>(inst, bounds, FID2,
 85                                                     /*add=*/2);
 86 
 87   verify<InstanceLogicalLayout1, int, float>(inst, bounds, FID1, /*add=*/1);
 88   verify<InstanceLogicalLayout2, long long, double>(inst, bounds, FID2,
 89                                                     /*add=*/2);
 90 
 91   Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
 92 }
 93 
 94 int main(int argc, const char **argv)
 95 {
 96   Runtime rt;
 97 
 98   rt.init(&argc, (char ***)&argv);
 99 
100   Processor p = Machine::ProcessorQuery(Machine::get_machine())
101                     .only_kind(Processor::LOC_PROC)
102                     .first();
103 
104   if(!p.exists()) {
105     p = Machine::ProcessorQuery(Machine::get_machine()).first();
106   }
107 
108   assert(p.exists());
109 
110   Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
111                                    CodeDescriptor(top_level_task),
112                                    ProfilingRequestSet())
113       .external_wait();
114 
115   rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
116 
117   int ret = rt.wait_for_shutdown();
118 
119   return ret;
120 }
```
