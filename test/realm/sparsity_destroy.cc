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

struct TaskArgs {
  NodeID node;
  SparsityMap<1> sparsity_map;
  Event wait_on;
};

void node_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;

  task_args.sparsity_map.impl();
  task_args.sparsity_map.destroy();

  {
    Rect<1> bounds{Rect<1>(Point<1>(0), Point<1>(50000))};
    SparsityMap<1> local_sparsity =
        SparsityMap<1>::construct({bounds}, /*always_create=*/true, /*disjoint=*/false);
    IndexSpace<1> is(bounds, local_sparsity);
    is.destroy();
  }

  {
    std::vector<Rect<1>> bounds{Rect<1>(Point<1>(0), Point<1>(20000)),
                                Rect<1>(Point<1>(30000), Point<1>(50000))};
    IndexSpace<1> is(bounds);
    is.destroy(task_args.wait_on);
  }

  {
    std::vector<Point<1>> points{Point<1>(0), Point<1>(20000), Point<1>(30000)};
    IndexSpace<1> is(points);
    is.destroy();
  }
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(50000)));
  rects.push_back(Rect<1>(Point<1>(50008), Point<1>(50008 * 2)));

  Machine machine = Machine::get_machine();
  std::map<NodeID, Memory> memories;
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
      ++it) {
    Memory m = *it;
    if(m.kind() == Memory::SYSTEM_MEM) {
      NodeID node = NodeID(ID(m).memory_owner_node());
      if(memories.find(node) == memories.end()) {
        memories[node] = m;
      }
    }
  }

  {
    UserEvent done = UserEvent::create_user_event();
    std::vector<Event> events;

    for(std::map<NodeID, Memory>::const_iterator it = memories.begin();
        it != memories.end(); ++it) {

      SparsityMap<1> sparsity_map =
          SparsityMap<1>::construct(rects, /*always_create=*/true, /*disjoint=*/true);
      sparsity_map.impl();

      Memory m = it->second;
      Processor proc = *Machine::ProcessorQuery(machine)
                            .only_kind(Processor::LOC_PROC)
                            .same_address_space_as(m)
                            .begin();

      {
        TaskArgs args;
        args.sparsity_map = sparsity_map;
        args.node = Network::my_node_id;
        args.wait_on = done;
        Event e = proc.spawn(NODE_TASK, &args, sizeof(args));
        events.push_back(e);
      }
    }

    done.trigger();
    Event::merge_events(events).wait();
  }

  usleep(300000);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(MAIN_TASK, main_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, NODE_TASK,
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
