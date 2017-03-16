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
#define NUM_TEST 4
#define PATH_LEN 10
#define NUM_FIELDS 25
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

Processor get_next_processor(Processor cur)
{
  Machine machine = Machine::get_machine();
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    if (*it == cur)
    {
      // Advance the iterator once to get the next, handle
      // the wrap around case too
      it++;
      if (it == all_procs.end())
      {
        return *(all_procs.begin());
      }
      else
      {
        return *it;
      }
    }
  }
  // Should always find one
  assert(false);
  return Processor::NO_PROC;
}

void top_level_task(const void *args, size_t arglen, const void *user_data, size_t user_data_len, Processor p)
{
  printf("top level task - DMA random tests\n");
  int num_workers = 1;
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
      INT_ARG("-w", num_workers);
    }
    assert(num_workers > 0);
  }
#undef INT_ARG
#undef BOOL_ARG

  Processor cur_proc = p;
  Event* events = (Event*) malloc(sizeof(Event) * num_workers);
  for (int i = 0; i < num_workers; i++) {
    cur_proc = get_next_processor(cur_proc);
    events[i] = cur_proc.spawn(WORKER_TASK, NULL, 0);
  }

  for (int i = 0; i < num_workers; i++) {
    events[i].wait();
  }
  printf("finish top level task......\n");
}

void worker_task(const void *args, size_t arglen, const void *user_data, size_t user_data_len, Processor p)
{
  printf("start worker task\n");
  double tp[2][MEM_KIND_SIZE][MEM_KIND_SIZE], max_tp[2][MEM_KIND_SIZE][MEM_KIND_SIZE];
  int count[2][MEM_KIND_SIZE][MEM_KIND_SIZE];
  for (int i = 0; i < MEM_KIND_SIZE; i++)
    for (int j = 0; j < MEM_KIND_SIZE; j++) {
      tp[0][i][j] = tp[1][i][j] = 0;
      max_tp[0][i][j] = max_tp[1][i][j] = 0;
      count[0][i][j] = count[1][i][j] = 0;
    }
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
    RegionInstance inst_vec[PATH_LEN];
    std::vector<size_t> field_order[PATH_LEN];
    int block_size_vec[PATH_LEN];
    for (unsigned j = 0; j < PATH_LEN; j++) {
      // random a memory in which we should create instance
      // remove HDF memory and File memory
      Machine machine = Machine::get_machine();
      std::set<Memory> mem;
      if(j == 0 || j == PATH_LEN - 1) {
        std::set<Memory> all_mem;
        machine.get_all_memories(all_mem);
        for(std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
          if (gasnet_mynode() == ID(*it).node() && it->kind() != Memory::HDF_MEM && it->kind() != Memory::FILE_MEM && it->kind() != Memory::GPU_FB_MEM)
            mem.insert(*it);
        }
      }
      else {
        std::set<Memory> all_mem;
        machine.get_all_memories(all_mem);
        for(std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
          if (it->kind() != Memory::HDF_MEM && it->kind() != Memory::FILE_MEM)
            mem.insert(*it);
        }
      }

      int mem_idx = rand() % mem.size();
      std::set<Memory>::iterator it = mem.begin();
      while (mem_idx > 0) {
        it++;
        mem_idx--;
      }
      //block_size_vec[j] = rand() % domain.get_volume() + 1;
      block_size_vec[j] = domain.get_volume();
      //printf("node = %d, kind = %d\n", ID(*it).node(), it->kind());
      inst_vec[j] = domain.create_instance(*it, field_sizes, block_size_vec[j]);
      assert(ID(inst_vec[j]).type() == ID::ID_INSTANCE);
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
        initialize_region_data(domain, inst_vec[0], field_order[0]);
      }
      else {
       std::vector<Domain::CopySrcDstField> src_fields, dst_fields;
        src_fields.clear();
        dst_fields.clear();
        // we submit a copy request
        for (size_t k = 0; k < NUM_FIELDS; k++) {
          Domain::CopySrcDstField src_field(inst_vec[j - 1], field_order[j-1][k] * sizeof(int), sizeof(int));
          Domain::CopySrcDstField dst_field(inst_vec[j], field_order[j][k] * sizeof(int), sizeof(int));
          src_fields.push_back(src_field);
          dst_fields.push_back(dst_field);
        }
        double starttime = Realm::Clock::current_time_in_microseconds();
        Event copyEvent = domain.copy(src_fields, dst_fields, Event::NO_EVENT);
        printf("[%d]	inst[%u] {%d (%d) -> %d (%d)}: ", gasnet_mynode(), j,
               get_runtime()->get_instance_impl(inst_vec[j-1])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j-1])->memory).node(),
               get_runtime()->get_instance_impl(inst_vec[j])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j])->memory).node());
        copyEvent.wait();
        double stoptime = Realm::Clock::current_time_in_microseconds();
        double totalsize = domain.get_volume();
        double throughput = totalsize * sizeof(int) * NUM_FIELDS / (stoptime - starttime);
        int src_kind = get_runtime()->get_instance_impl(inst_vec[j-1])->memory.kind();
        int dst_kind = get_runtime()->get_instance_impl(inst_vec[j])->memory.kind();
        if (ID(get_runtime()->get_instance_impl(inst_vec[j])->memory).node() == ID(get_runtime()->get_instance_impl(inst_vec[j-1])->memory).node()) {
          tp[0][src_kind][dst_kind] += throughput;
          count[0][src_kind][dst_kind] ++;
          if (throughput > max_tp[0][src_kind][dst_kind])
            max_tp[0][src_kind][dst_kind] = throughput;
        } else {
          tp[1][src_kind][dst_kind] += throughput;
          count[1][src_kind][dst_kind] ++;
          if (throughput > max_tp[1][src_kind][dst_kind])
            max_tp[1][src_kind][dst_kind] = throughput;
        }
        printf("time = %.2lfus, tp = %.2lfMB/s\n", stoptime - starttime, throughput);
        //if (verify_region_data(domain, inst_vec[j], field_order[j]))
          //printf("check passed...\n ");
        //else
          //printf("check failed...\n");
      }
    }

    if (verify_region_data(domain, inst_vec[PATH_LEN - 1], field_order[PATH_LEN - 1]))
      printf("All check passed...\n");
    else {
      printf("Some check failed...\n");
      assert(false);
    }
    for (unsigned j = 0; j < PATH_LEN; j++) {
      get_runtime()->get_memory_impl(get_runtime()->get_instance_impl(inst_vec[j])->memory)->destroy_instance(inst_vec[j], false);
    }
  }

  // print average tp stats
  for (int k = 0; k < 2; k++) {
    if (k == 0)
      printf("Local xfer path throughput stats:\n");
    else
      printf("Remote xfer path throughput stats:\n");
    for (int i = 0; i < MEM_KIND_SIZE; i++)
      for (int j = 0; j < MEM_KIND_SIZE; j++) {
        if (count[k][i][j] > 0)
          printf("(%d)->(%d): %.2lfMB/s\n", i, j, tp[k][i][j] / count[k][i][j]);
      }
  }

  // print max tp stats
  for (int k = 0; k < 2; k++) {
    if (k == 0)
      printf("Local xfer path max throughput stats:\n");
    else
      printf("Remote xfer path max throughput stats:\n");
    for (int i = 0; i < MEM_KIND_SIZE; i++)
      for (int j = 0; j < MEM_KIND_SIZE; j++) {
        if (max_tp[k][i][j] > 0)
          printf("(%d)->(%d): %.2lfMB/s\n", i, j, max_tp[k][i][j]);
      }
  }
  printf("all check passed......\n");
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
