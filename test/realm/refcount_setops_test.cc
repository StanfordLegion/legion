#include <time.h>

#include <cassert>

#include "realm/cmdline.h"
#include "realm/runtime.h"
#include "realm.h"
#include "realm/id.h"
#include "realm/network.h"
#include "osdep.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  NODE_TASK,
};

constexpr int NUM_INSTS = 1;

struct TaskArgs {
  IndexSpace<1> lhs[NUM_INSTS];
  IndexSpace<1> rhs[NUM_INSTS];
};

namespace TestConfig {
  bool remote_create = false;
};

void node_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;

  // intersection lhs==rhs
  {
    std::vector<IndexSpace<1>> results;
    assert(!task_args.lhs[0].dense());
    Event e2 = IndexSpace<1>::compute_intersections(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs), std::end(task_args.lhs)},
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs), std::end(task_args.rhs)},
        results, ProfilingRequestSet());
    e2.wait();
    for(size_t i = 0; i < results.size(); i++) {
      results[i].sparsity.remove_references();
    }
  }
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
  rects.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

  Machine machine = Machine::get_machine();
  std::vector<Memory> memories;
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
      ++it) {
    Memory m = *it;
    if(m.kind() == Memory::SYSTEM_MEM) {
      memories.push_back(m);
    }
  }

  std::vector<IndexSpace<1>> roots;

  std::vector<Event> events;
  for(std::vector<Memory>::const_iterator it = memories.begin(); it != memories.end();
      ++it) {
    Processor proc = *Machine::ProcessorQuery(machine)
                          .only_kind(Processor::LOC_PROC)
                          .same_address_space_as(*it)
                          .begin();

    roots.push_back(IndexSpace<1>(rects));
    roots.back().sparsity.add_references();

    std::vector<IndexSpace<1>> lhs;
    roots.back()
        .create_equal_subspaces(NUM_INSTS, 1, lhs, Realm::ProfilingRequestSet())
        .wait();
    std::vector<IndexSpace<1>> rhs = lhs;

    /*if((TestConfig::remote_create) &&
       NodeID(ID(*it).memory_owner_node()) ==
           NodeID(ID(ptr_data[0].inst).instance_owner_node())) {
      continue;
    }*/

    TaskArgs args;
    for(int i = 0; i < NUM_INSTS; i++) {
      args.lhs[i] = lhs[i];
      args.rhs[i] = rhs[i];
    }
    Event e = proc.spawn(NODE_TASK, &args, sizeof(args));
    events.push_back(e);
  }

  Event::merge_events(events).wait();
  for(size_t i = 0; i < roots.size(); i++) {
    roots[i].sparsity.destroy();
  }
  usleep(100000);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  CommandLineParser cp;
  cp.add_option_int_units("-remote_create", TestConfig::remote_create);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.init(&argc, &argv);
  rt.register_task(MAIN_TASK, main_task);
  Processor::register_task_by_kind(Processor::LOC_PROC, /*!global=*/false, NODE_TASK,
                                   CodeDescriptor(node_task), ProfilingRequestSet(), 0, 0)
      .wait();

  ModuleConfig *core = Runtime::get_runtime().get_module_config("core");
  assert(core->set_property("enable_sparsity_refcount", 1));

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, MAIN_TASK, 0, 0);
  int ret = rt.wait_for_shutdown();
  return ret;
}
