#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#include <deque>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WORKER_TASK,
  ALLOC_PROF_TASK,
  ALLOC_INST_TASK
};

int num_iterations = 1;
bool needs_oom = false;

struct WorkerArgs {
  RegionInstance inst;
  Rect<1> bounds;
  std::vector<int> data;
};

struct ProfTimelResult {
  int *called;
  UserEvent done;
};

struct ProfAllocResult {
  int *success;
  UserEvent done;
};

void inst_profiling_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfTimelResult));
  const ProfTimelResult *result = static_cast<const ProfTimelResult *>(resp.user_data());
  ProfilingMeasurements::InstanceTimeline inst_time;
  assert((resp.get_measurement(inst_time)));

  // TODO(apryakhin@): Verify timestamps
  assert(inst_time.create_time != 0);
  assert(inst_time.ready_time != 0);
  *(result->called) = 1;

  result->done.trigger();
}

void alloc_profiling_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfAllocResult));
  const ProfAllocResult *result = static_cast<const ProfAllocResult *>(resp.user_data());
  ProfilingMeasurements::InstanceAllocResult inst_alloc;
  assert((resp.get_measurement(inst_alloc)));
  *(result->success) = inst_alloc.success;
  result->done.trigger();
}

template <int N>
InstanceLayoutGeneric *create_layout(Rect<N> bounds)
{
  std::map<FieldID, size_t> fields;
  fields[0] = sizeof(int);
  InstanceLayoutConstraints ilc(fields, 1);
  int dim_order[N];
  for(int i = 0; i < N; i++)
    dim_order[i] = 0;
  InstanceLayoutGeneric *ilg =
      InstanceLayoutGeneric::choose_instance_layout<N, int>(bounds, ilc, dim_order);
  return ilg;
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::map<NodeID, Memory> memories;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM);
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    Memory memory = *it;
    NodeID owner = ID(*it).memory_owner_node();
    if(!ID(memory).is_ib_memory() && memories.count(owner) == 0) {
      memories[owner] = memory;
    }
  }

  std::vector<Processor> reader_cpus, cpus;
  Machine machine = Machine::get_machine();
  for(size_t i = 0; i < memories.size(); i++) {
    Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
                                     .only_kind(Processor::LOC_PROC)
                                     .same_address_space_as(memories[i]);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it; it++) {
      reader_cpus.push_back(*it);
      break;
    }
  }

  assert(!reader_cpus.empty());
  assert(reader_cpus.size() == memories.size());

  Rect<1> bounds;
  bounds.lo[0] = 0;
  bounds.hi[0] = 512 * 512 - 1;

  std::vector<size_t> field_sizes(1, 4);

  RegionInstance inst1;
  RegionInstance::create_instance(inst1, memories[0], bounds, field_sizes, 0 /*SOA*/,
                                  ProfilingRequestSet())
      .wait();
  assert(inst1.exists());
  Event e1 = inst1.fetch_metadata(reader_cpus[0]);

  UserEvent destroy_event = UserEvent::create_user_event();
  inst1.destroy(destroy_event);

  RegionInstance inst2;
  RegionInstance::create_instance(inst2, memories[0], bounds, field_sizes, 0,
                                  ProfilingRequestSet())
      .wait();

  std::vector<int> data;
  {
    int index = 0;
    AffineAccessor<int, 1, int> acc(inst2, 0);
    IndexSpaceIterator<1, int> it(bounds);
    while(it.valid) {
      PointInRectIterator<1, int> pit(it.rect);
      while(pit.valid) {
        data.push_back(index);
        acc[pit.p] = index++;
        pit.step();
      }
      it.step();
    }
  }

  WorkerArgs worker_args;
  worker_args.inst = inst2;
  worker_args.bounds = bounds;
  worker_args.data = data;
  Event e2 = reader_cpus[0].spawn(WORKER_TASK, &worker_args, sizeof(WorkerArgs),
                                  ProfilingRequestSet(), e1);
  e2.wait();

  destroy_event.trigger();
  usleep(100000);
}

void worker_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  const WorkerArgs *wargs = static_cast<const WorkerArgs *>(args);
  Rect<1> bounds = wargs->bounds;
  RegionInstance inst = wargs->inst;

  Event timel_event;
  Event alloc_event;

  size_t index = 0;
  std::vector<int> alloc_results(num_iterations * 2);
  std::vector<int> timel_results(num_iterations * 2);

  for(int i = 0; i < num_iterations; i++) {
    Rect<1> next_bounds;
    next_bounds.lo[0] = 0;
    next_bounds.hi[0] = needs_oom ? bounds.hi[0] : bounds.hi[0] / 2;

    log_app.info() << "redistrict bounds:" << bounds << " next_bounds:" << next_bounds
                   << " inst:" << inst;

    std::vector<RegionInstance> insts(2);
    const InstanceLayoutGeneric *ilg_a = create_layout(next_bounds);
    const InstanceLayoutGeneric *ilg_b = create_layout(next_bounds);
    std::vector<const InstanceLayoutGeneric *> layouts{ilg_a, ilg_b};

    std::vector<Event> timel_events;
    std::vector<Event> alloc_events;

    std::vector<ProfilingRequestSet> prs(2);
    for(int i = 0; i < 2; i++) {
      {
        UserEvent event = UserEvent::create_user_event();
        ProfTimelResult result;
        result.done = event;
        result.called = &timel_results[index];
        prs[i]
            .add_request(p, ALLOC_INST_TASK, &result, sizeof(ProfTimelResult))
            .add_measurement<ProfilingMeasurements::InstanceTimeline>();
        timel_events.push_back(event);
      }

      {
        UserEvent event = UserEvent::create_user_event();
        ProfAllocResult result;
        result.done = event;
        result.success = &alloc_results[index];
        prs[i]
            .add_request(p, ALLOC_PROF_TASK, &result, sizeof(ProfAllocResult))
            .add_measurement<ProfilingMeasurements::InstanceAllocResult>();
      }
      index++;
    }

    Event e = inst.redistrict(insts.data(), layouts.data(), 2, prs.data());

    bool poisoned = false;
    e.wait_faultaware(poisoned);
    if(needs_oom) {
      assert(poisoned);
      return;
    }

    assert(poisoned == false);

    timel_event = Event::merge_events(timel_events);
    alloc_event = Event::merge_events(alloc_events);

    delete ilg_a;
    delete ilg_b;

    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(insts[0], 0, sizeof(int));
    dsts[0].set_field(insts[1], 0, sizeof(int));
    e = IndexSpace<1>(next_bounds).copy(srcs, dsts, ProfilingRequestSet(), e);
    e.wait();

    int index = 0; /// next_bounds.volume();
    for(size_t i = 1; i < insts.size(); i++) {
      AffineAccessor<int, 1, int> acc(insts[i], 0);
      IndexSpaceIterator<1, int> it(next_bounds);
      while(it.valid) {
        PointInRectIterator<1, int> pit(it.rect);
        while(pit.valid) {
          int val = acc[pit.p];
          assert(val == wargs->data[index++]);
          pit.step();
        }
        it.step();
      }
    }

    bounds = next_bounds;
    inst.destroy();
    inst = insts[0];
    insts[1].destroy();
  }

  inst.destroy();
  alloc_event.wait();
  timel_event.wait();

  for(size_t i = 0; i < alloc_results.size(); i++) {
    assert(alloc_results[i]);
  }

  for(size_t i = 0; i < timel_results.size(); i++) {
    assert(timel_results[i]);
  }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-i")) {
      num_iterations = atoi(argv[++i]);
      continue;
    }
  }

  CommandLineParser cp;
  cp.add_option_int("-i", num_iterations);
  cp.add_option_int("-needs_oom", needs_oom);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(WORKER_TASK, worker_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   ALLOC_PROF_TASK, CodeDescriptor(alloc_profiling_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   ALLOC_INST_TASK, CodeDescriptor(inst_profiling_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  rt.wait_for_shutdown();
  return 0;
}
