#include "realm.h"
#include "realm/id.h"

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
  Processor p_worker = p;
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
    p_worker = *it;

  // get a memory close to the target processor
  Memory m_worker = Machine::MemoryQuery(Machine::get_machine())
                        .only_kind(Memory::SYSTEM_MEM)
                        .best_affinity_to(p_worker)
                        .first();
  assert(m_worker.exists());

  Rect<1> bounds;
  bounds.lo[0] = 0;
  bounds.hi[0] = 512 * 512 - 1;

  std::vector<size_t> field_sizes(1, 4);

  RegionInstance inst;
  RegionInstance::create_instance(inst, m_worker, bounds, field_sizes, 0 /*SOA*/,
                                  ProfilingRequestSet());
  assert(inst.exists());
  Event e1 = inst.fetch_metadata(p_worker);

  WorkerArgs worker_args;
  worker_args.inst = inst;
  worker_args.bounds = bounds;
  Event e2 = p_worker.spawn(WORKER_TASK, &worker_args, sizeof(WorkerArgs),
                            ProfilingRequestSet(), e1);
  e2.wait();

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

