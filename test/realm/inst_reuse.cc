#include "realm.h"
#include "realm/id.h"

#include <deque>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  WORKER_TASK,
};

int num_iterations = 1024;  // can't actually test long enough to exhaust IDs
size_t max_live_instances = 4;

struct WorkerArgs {
  RegionInstance inst;
  Rect<2> exp_bounds;
};

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "testing instance reuse: iterations=" << num_iterations << " max_live=" << max_live_instances;

  // choose the last processor to be our worker - it should be remote when
  //  running with more than one node
  Processor p_worker = p;
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = pq.begin();
      it != pq.end();
      ++it)
    p_worker = *it;

  log_app.info() << "top level task on " << p << ", worker is " << p_worker;

  // get a memory close to the target processor
  Memory m_worker = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM).best_affinity_to(p_worker).first();
  log_app.info() << "memory is " << m_worker;
  assert(m_worker.exists());

  std::deque<RegionInstance> instances;
  std::deque<Event> finish_events;
  std::set<RegionInstance> ids_used;

  for(int i = 0; i < num_iterations; i++) {
    // pick different bounds for each instance
    Rect<2> bounds;
    bounds.lo[0] = i;
    bounds.lo[1] = i+1;
    bounds.hi[0] = i + (i % 31);
    bounds.hi[1] = i + 1 + (i >> 5);

    std::vector<size_t> field_sizes(1, 8);

    RegionInstance inst;
    RegionInstance::create_instance(inst,
				    m_worker,
				    bounds,
				    field_sizes,
				    0 /*SOA*/,
				    ProfilingRequestSet());
    assert(inst.exists());

    // since we can't run long enough to actually exhaust instance IDs, we
    //  check that the number of unique IDs we use isn't larger than the
    //  maximum live at any time - this check is only possible when we're
    //  working with a local memory though
    if(m_worker.address_space() == p.address_space()) {
      ids_used.insert(inst);
      assert(ids_used.size() <= max_live_instances);
    }

    // prefetch the metadata for the worker processor - this event includes
    //  the successful instance creation above
    Event e1 = inst.fetch_metadata(p_worker);

    log_app.info() << "master created: " << inst << ", bounds=" << bounds;

    WorkerArgs args;
    args.inst = inst;
    args.exp_bounds = bounds;
    Event e2 = p_worker.spawn(WORKER_TASK, &args, sizeof(WorkerArgs),
			      ProfilingRequestSet(), e1);

    instances.push_back(inst);
    finish_events.push_back(e2);

    while(instances.size() >= max_live_instances) {
      RegionInstance inst = instances.front();
      Event e = finish_events.front();
      instances.pop_front();
      finish_events.pop_front();
      e.wait();
      inst.destroy();
    }
  }

  while(!instances.empty()) {
    RegionInstance inst = instances.front();
    Event e = finish_events.front();
    instances.pop_front();
    finish_events.pop_front();
    e.wait();
    inst.destroy();
  }
}

void worker_task(const void *args, size_t arglen,
		 const void *userdata, size_t userlen, Processor p)
{
  const WorkerArgs *wargs = static_cast<const WorkerArgs *>(args);
  GenericAccessor<void *, 2> acc(wargs->inst, 0);
  log_app.info() << "worker got: " << acc;
  IndexSpace<2> act_bounds = wargs->inst.get_indexspace<2>();
  assert(act_bounds.dense() && (act_bounds.bounds == wargs->exp_bounds));
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

    if(!strcmp(argv[i], "-m")) {
      max_live_instances = atoi(argv[++i]);
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

