#include "realm.h"
#include "realm/id.h"
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
};

int num_iterations = 8;

struct WorkerArgs {
  RegionInstance inst;
  Rect<1> bounds;
};

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
  mq.only_kind( Memory::SYSTEM_MEM );
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    Memory memory = *it;
    NodeID owner = ID(*it).memory_owner_node();
    if(!ID(memory).is_ib_memory() && memories.count(owner) == 0) {
      memories[owner] = memory;
    }
  }

  std::vector<Processor> reader_cpus, cpus;
  Machine machine = Machine::get_machine();
  for(const std::pair<NodeID, Memory> &memory : memories) {
    Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
                                     .only_kind(Processor::LOC_PROC)
                                     .same_address_space_as(memory.second);
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

  WorkerArgs worker_args;
  worker_args.inst = inst2;
  worker_args.bounds = bounds;
  Event e2 = reader_cpus[0].spawn(WORKER_TASK, &worker_args, sizeof(WorkerArgs),
                                  ProfilingRequestSet(), e1);
  e2.wait();

  destroy_event.trigger();
  inst2.destroy();

  usleep(100000);
}

void worker_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  const WorkerArgs *wargs = static_cast<const WorkerArgs *>(args);
  Rect<1> bounds = wargs->bounds;
  RegionInstance inst = wargs->inst;
  for(int i = 0; i < num_iterations; i++) {
    Rect<1> next_bounds;
    next_bounds.lo[0] = 0;
    next_bounds.hi[0] = bounds.hi[0] / 2;

    log_app.info() << "redistrict bounds:" << bounds << " next_bounds:" << next_bounds
                   << " inst:" << inst;

    std::vector<RegionInstance> insts(2);
    InstanceLayoutGeneric *ilg_a = create_layout(next_bounds);
    InstanceLayoutGeneric *ilg_b = create_layout(next_bounds);
    std::vector<InstanceLayoutGeneric *> layouts{ilg_a, ilg_b};

    Event e = inst.redistrict(insts.data(), layouts.data(), 2, ProfilingRequestSet());

    bool poisoned = false;
    e.wait_faultaware(poisoned);
    assert(poisoned == false);

    bounds = next_bounds;
    inst = insts[0];
    insts[1].destroy();
  }
  inst.destroy();
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

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(WORKER_TASK, worker_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
