#include <time.h>

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "realm/cmdline.h"
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
  IndexSpace<1> ss_inst1[1];
  FieldDataDescriptor<IndexSpace<1>, Point<1>> fd_ptrs1[1];
  IndexSpace<1> root2;
};


namespace TestConfig {
  bool remote_create = false;
};

void node_task_0(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;
  std::vector<IndexSpace<1>> ss_images;
  Event e2 = task_args.root2.create_subspaces_by_image(
      std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>>{task_args.fd_ptrs1[0]},
      std::vector<IndexSpace<1>>{task_args.ss_inst1[0]}, ss_images,
      ProfilingRequestSet());
  for(IndexSpace<1> image : ss_images) {
    image.sparsity.destroy(e2);
  }
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
  rects.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

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

  IndexSpace<1> root1(rects);
  IndexSpace<1> root2(rects);

  size_t num_insts = 1;

  std::vector<IndexSpace<1>> ss_inst1;
  root1.create_equal_subspaces(num_insts, 1, ss_inst1, Realm::ProfilingRequestSet())
      .wait();

  std::vector<size_t> field_sizes;
  field_sizes.push_back(sizeof(int));
  field_sizes.push_back(sizeof(Point<1>));

  std::vector<FieldDataDescriptor<IndexSpace<1>, int>> fd_vals1(num_insts);
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> fd_ptrs1(num_insts);

  for(size_t i = 0; i < num_insts; i++) {
    size_t mem_idx = i % memories.size();
    RegionInstance ri;
    RegionInstance::create_instance(ri, memories[mem_idx], ss_inst1[i], field_sizes, 0,
                                    Realm::ProfilingRequestSet())
        .wait();

    fd_ptrs1[i].index_space = ss_inst1[i];
    fd_ptrs1[i].inst = ri;
    fd_ptrs1[i].field_offset = 0 + sizeof(int);
  }

  std::vector<Event> events;
  for(std::vector<Memory>::const_iterator it = memories.begin(); it != memories.end();
      ++it) {
    Memory m = *it;

    Processor proc = *Machine::ProcessorQuery(machine)
                          .only_kind(Processor::LOC_PROC)
                          .same_address_space_as(m)
                          .begin();

    if((Network::max_node_id > 1 && TestConfig::remote_create) &&
       NodeID(ID(m).memory_owner_node()) ==
           NodeID(ID(fd_ptrs1[0].inst).instance_owner_node())) {
      continue;
    }

    {
      TaskArgs args;
      args.ss_inst1[0] = ss_inst1[0];
      args.fd_ptrs1[0] = fd_ptrs1[0];
      args.root2 = root2;
      Event e = proc.spawn(NODE_TASK_0, &args, sizeof(args));
      events.push_back(e);
    }
  }

  done.trigger();
  Event::merge_events(events).wait();
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
