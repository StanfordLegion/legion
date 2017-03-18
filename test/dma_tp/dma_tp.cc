#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"
#include "lowlevel_impl.h"
#include "channel.h"

#define DIM_X 128
#define DIM_Y 128
#define DIM_Z 64
#define NUM_TEST 1
#define PATH_LEN 2
#define NUM_FIELDS 8
#define NUM_REQUESTS 64
#define MEM_KIND_SIZE 20

using namespace LegionRuntime::LowLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  WORKER_TASK = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

struct InputArgs {
  int argc;
  char **argv;
};

InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

void initialize_region_data(Domain domain, RegionInstance inst, std::vector<size_t> field_order)
{
  // TODO: fix it
  if (get_runtime()->get_instance_impl(inst)->memory.kind() == Memory::GPU_FB_MEM)
    return;
  RegionAccessor<AccessorType::Generic> acc = inst.get_accessor();
  switch(domain.dim) {
  case 0:
    assert(false);
    break;
  case 1:
  {
    Rect<1> rect = domain.get_rect<1>();
    int idx = 0;
    int read_data[NUM_FIELDS];
    for(GenericPointInRectIterator<1> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        read_data[i] = idx * i;
        acc.write_untyped(DomainPoint::from_point<1>(pir.p), &read_data[i], sizeof(int), field_order[i] * sizeof(int));
      }
    }
    break;
  }
  case 2:
  {
    Rect<2> rect = domain.get_rect<2>();
    int idx = 0;
    int read_data[NUM_FIELDS];
    for(GenericPointInRectIterator<2> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        read_data[i] = idx * i;
        acc.write_untyped(DomainPoint::from_point<2>(pir.p), &read_data[i], sizeof(int), field_order[i] * sizeof(int));
      }
    }
    break;
  }
  case 3:
  {
    Rect<3> rect = domain.get_rect<3>();
    int idx = 0;
    int read_data[NUM_FIELDS];
    for(GenericPointInRectIterator<3> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        read_data[i] = idx * i;
        acc.write_untyped(DomainPoint::from_point<3>(pir.p), &read_data[i], sizeof(int), field_order[i] * sizeof(int));
      }
    }
    break;
  }
  default:
    assert(false);
  }
}

bool verify_region_data(Domain domain, RegionInstance inst, std::vector<size_t> field_order)
{
  RegionAccessor<AccessorType::Generic> acc = inst.get_accessor();
  bool check = true;
  switch(domain.dim) {
  case 0:
    assert(false);
    break;
  case 1:
  {
    Rect<1> rect = domain.get_rect<1>();
    int idx = 0;
    int read_data;
    for(GenericPointInRectIterator<1> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        acc.read_untyped(DomainPoint::from_point<1>(pir.p), &read_data, sizeof(int), field_order[i] * sizeof(int));
        check = check && (read_data == idx * i);
        if (read_data != idx * i) {
          printf("read_data = %d, expected = %d\n", read_data, idx * i);
          assert(0);
        }
      }
    }
    break;
  }
  case 2:
  {
    Rect<2> rect = domain.get_rect<2>();
    int idx = 0;
    int read_data;
    for(GenericPointInRectIterator<2> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        acc.read_untyped(DomainPoint::from_point<2>(pir.p), &read_data, sizeof(int), field_order[i] * sizeof(int));
        check = check && (read_data == idx * i);
      }
    }
    break;
  }
  case 3:
  {
    Rect<3> rect = domain.get_rect<3>();
    int idx = 0;
    int read_data;
    for(GenericPointInRectIterator<3> pir(rect); pir; pir++) {
      idx++;
      for (int i = 0; i < NUM_FIELDS; i++) {
        acc.read_untyped(DomainPoint::from_point<3>(pir.p), &read_data, sizeof(int), field_order[i] * sizeof(int));
        check = check && (read_data == idx * i);
      }
    }
    break;
  }
  default:
    assert(false);
  }
  return check;
}

struct WorkerTaskArgs {
  Memory src_mem, dst_mem;
  bool test_xfer;
};

void top_level_task(const void *args, size_t arglen, const void *user_data, size_t user_data_len, Processor p)
{
  printf("top level task - DMA random tests\n");
  bool only_remote = false, test_xfer = false;
  // Parse the input arguments
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)
  {
    InputArgs &inputs = get_input_args();
    char **argv = inputs.argv;
    for (int i = 1; i < inputs.argc; i++)
    {
      BOOL_ARG("-r", only_remote);
      BOOL_ARG("-t", test_xfer);
    }
  }
#undef INT_ARG
#undef BOOL_ARG
  
  Machine machine = Machine::get_machine();
  std::set<Memory> all_mem, src_mem_set, dst_mem_set;
  machine.get_all_memories(all_mem);
  for(std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
    if (gasnet_mynode() == ID(*it).node() && it->kind() != Memory::HDF_MEM && it->kind() != Memory::FILE_MEM)
      src_mem_set.insert(*it);
    if ((gasnet_mynode() != ID(*it).node() || !only_remote) && it->kind() != Memory::HDF_MEM && it->kind() != Memory::FILE_MEM)
      dst_mem_set.insert(*it);
  }

  Processor cur_proc = p;
  for(std::set<Memory>::iterator src_it = src_mem_set.begin(); src_it != src_mem_set.end(); src_it++) 
    for(std::set<Memory>::iterator dst_it = dst_mem_set.begin(); dst_it != dst_mem_set.end(); dst_it++) {
      WorkerTaskArgs args;
      args.src_mem = *src_it;
      args.dst_mem = *dst_it;
      args.test_xfer = test_xfer;
      Event event = cur_proc.spawn(WORKER_TASK, &args, sizeof(args));
      event.wait();
    }
  printf("finish top level task......\n");
}

void worker_task(const void *args, size_t arglen, const void *user_data, size_t user_data_len, Processor p)
{
  printf("start worker task\n");
  const WorkerTaskArgs& worker_args = *(const WorkerTaskArgs *)args;
  Memory src_mem = worker_args.src_mem, dst_mem = worker_args.dst_mem;
  bool test_xfer = worker_args.test_xfer;
  std::vector<size_t> field_sizes;
  for (unsigned i = 0; i < NUM_FIELDS; i++)
    field_sizes.push_back(sizeof(int));
  for (unsigned i = 0; i < NUM_TEST; i++) {
    printf("Test case #%u:\n", i);
    int dim = 3;
    Domain domain;
    switch (dim) {
      case 0:
      case 1:
      {
        Point<1> lo = make_point(0), hi = make_point(DIM_X - 1);
        Rect<1> rect(lo, hi);
        domain = Domain::from_rect<1>(rect);
        break;
      }
      case 2:
      {
        Point<2> lo = make_point(0, 0), hi = make_point(DIM_X - 1, DIM_Y - 1);
        Rect<2> rect(lo, hi);
        domain = Domain::from_rect<2>(rect);
        break;
      }
      case 3:
      {
        Point<3> lo = make_point(0, 0, 0), hi = make_point(DIM_X - 1, DIM_Y - 1, DIM_Z - 1);
        Rect<3> rect(lo, hi);
        domain = Domain::from_rect<3>(rect);
        break;
      }
      default:
        assert(false);
    }
    RegionInstance src_inst, dst_inst;
    std::deque<RegionInstance> inst_vec;
    std::vector<size_t> field_order[PATH_LEN];
    int block_size_vec[PATH_LEN];
    for (unsigned j = 0; j < PATH_LEN; j++) {
      //block_size_vec[j] = rand() % domain.get_volume() + 1;
      if (j == 0 || !test_xfer)
        block_size_vec[j] = domain.get_volume();
      else
        block_size_vec[j] = 1;
      //printf("node = %d, kind = %d\n", ID(*it).node(), it->kind());
      // random field order of this region instance
      std::vector<size_t> rand_order;
      rand_order.clear();
      for (size_t k = 0; k < NUM_FIELDS; k++)
        rand_order.push_back(k);
      field_order[j].clear();
      while (!rand_order.empty()) {
        size_t idx = rand() % rand_order.size();
        std::vector<size_t>::iterator field_iter = rand_order.begin();
        while (idx > 0) {idx--; field_iter++;}
        field_order[j].push_back(*field_iter);
        rand_order.erase(field_iter);
      }
      if (j == 0) {
        // we initialize the first region instance
        src_inst = domain.create_instance(src_mem, field_sizes, block_size_vec[j]);
        inst_vec.push_back(src_inst);
        initialize_region_data(domain, inst_vec[0], field_order[0]);
      }
      else {
        assert(j == 1);
        std::set<Event> wait_on;
        UserEvent start_event = UserEvent::create_user_event();
        for (unsigned k = 0; k < NUM_REQUESTS; k++) {
          if (k == 0 || dst_mem.kind() != Memory::GPU_FB_MEM) {
            dst_inst = domain.create_instance(dst_mem, field_sizes, block_size_vec[j]);
            inst_vec.push_back(dst_inst);
          }
          std::vector<Domain::CopySrcDstField> src_fields, dst_fields;
          src_fields.clear();
          dst_fields.clear();
          // we submit a copy request
          for (size_t k = 0; k < NUM_FIELDS; k++) {
            Domain::CopySrcDstField src_field(src_inst, field_order[j-1][k] * sizeof(int), sizeof(int));
            Domain::CopySrcDstField dst_field(dst_inst, field_order[j][k] * sizeof(int), sizeof(int));
            src_fields.push_back(src_field);
            dst_fields.push_back(dst_field);
          }
          Event copyEvent = domain.copy(src_fields, dst_fields, start_event);
          wait_on.insert(copyEvent);
        }
        double starttime = Realm::Clock::current_time_in_microseconds();
        start_event.trigger();
        Event::merge_events(wait_on).wait();
        printf("[%d]	inst[%u] {%d (%d) -> %d (%d)}: ", gasnet_mynode(), j,
               get_runtime()->get_instance_impl(inst_vec[j-1])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j-1])->memory).node(),
               get_runtime()->get_instance_impl(inst_vec[j])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j])->memory).node());
        double stoptime = Realm::Clock::current_time_in_microseconds();
        double totalsize = domain.get_volume();
        totalsize = totalsize * NUM_REQUESTS;
        double throughput = totalsize * sizeof(int) * NUM_FIELDS / (stoptime - starttime);
        printf("time = %.2lfus, tp = %.2lfMB/s\n", stoptime - starttime, throughput);
      }
    }

    while (!inst_vec.empty()) {
      RegionInstance inst = inst_vec.front();
      inst_vec.pop_front();
      get_runtime()->get_memory_impl(get_runtime()->get_instance_impl(inst)->memory)->destroy_instance(inst, false);
    }
  }

  printf("finish worker task..\n");
  return;
}

int main(int argc, char **argv)
{
  Runtime rt;
  
  bool ok = rt.init(&argc, &argv);
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(WORKER_TASK, worker_task);

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++)
      if(it->kind() == Processor::LOC_PROC) {
	p = *it;
	break;
      }
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
