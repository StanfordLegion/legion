#include <time.h>

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "philox.h"
#include "realm.h"
#include "realm/id.h"
#include "realm/network.h"
#include "realm/nodeset.h"
#include "realm/threads.h"
#include "osdep.h"

using namespace Realm;

#define REALM_SPARSITY_DELETES

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  NODE_TASK_0,
};

struct TaskArgs {
  NodeID node;
  SparsityMap<1> sparsity_map;
  Event wait_on;
};

void node_task_0(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;
  // add remote reference
  task_args.sparsity_map.add_references(2);
  // remove remote reference
  task_args.sparsity_map.remove_references(2);
  // deferred remote destroy

  task_args.sparsity_map.destroy(task_args.wait_on);

  SparsityMap<1> local_sparsity =
      SparsityMap<1>::construct({Rect<1>(Point<1>(0), Point<1>(50000))}, true, true);
  local_sparsity.destroy(task_args.wait_on);

  task_args.wait_on.wait();
  assert(local_sparsity.impl()->is_valid() == 0);
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(50000)));
  rects.push_back(Rect<1>(Point<1>(50008), Point<1>(50008 * 2)));

  UserEvent done = UserEvent::create_user_event();

  Machine machine = Machine::get_machine();
  std::vector<Memory> memories;
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
      ++it) {
    Memory m = *it;
    if(m.kind() != Memory::SYSTEM_MEM)
      continue;
    memories.push_back(m);
  }

  std::vector<SparsityMap<1>> sparsity_maps;
  std::vector<Event> events;
  for(std::vector<Memory>::const_iterator it = memories.begin(); it != memories.end();
      ++it) {
    Memory m = *it;

    Processor proc = *Machine::ProcessorQuery(machine)
                          .only_kind(Processor::LOC_PROC)
                          .same_address_space_as(m)
                          .begin();
    sparsity_maps.push_back(SparsityMap<1>::construct(rects, true, true));

    {
      TaskArgs args;
      sparsity_maps.back().add_references();
      args.sparsity_map = sparsity_maps.back();
      args.node = Network::my_node_id;
      args.wait_on = done;
      Event e = proc.spawn(NODE_TASK_0, &args, sizeof(args));
      events.push_back(e);
    }
  }

  done.trigger();
  Event::merge_events(events).wait();

  int errors = 0;
#ifdef REALM_SPARSITY_DELETES
  for(auto &sparsity_map : sparsity_maps) {
    auto *impl = sparsity_map.impl();
    assert(impl);
    if(impl->is_valid()) {
      log_app.info() << "sparsity_map=" << sparsity_map << " hasn't been deleted";
      errors++;
    }
  }

  if(errors > 0) {
    log_app.error() << "Test failed with erros=" << errors;
  }
#endif

  usleep(100000);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), errors ? 1 : 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(MAIN_TASK, main_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, NODE_TASK_0,
                                   CodeDescriptor(node_task_0), ProfilingRequestSet(), 0,
                                   0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, MAIN_TASK, 0, 0);
  int ret = rt.wait_for_shutdown();
  return ret;
}
