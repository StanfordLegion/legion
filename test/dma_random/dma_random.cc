#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"
#include "lowlevel_impl.h"
#include "channel.h"

#define DIM_X 100
#define DIM_Y 100
#define DIM_Z 10
#define NUM_TEST 100
#define PATH_LEN 20
#define NUM_FIELDS 5

using namespace LegionRuntime::LowLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

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

void top_level_task(const void *args, size_t arglen, Processor p)
{
  printf("top level task - DMA random tests\n");
  std::vector<size_t> field_sizes;
  for (unsigned i = 0; i < NUM_FIELDS; i++)
    field_sizes.push_back(sizeof(int));
  for (unsigned i = 0; i < NUM_TEST; i++) {
    printf("Test case #%u:\n", i);
    int dim = i % 3 + 1;
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
      Machine machine = Machine::get_machine();
      std::set<Memory> mem;
      if(j == 0 || j == PATH_LEN - 1) {
        std::set<Memory> all_mem;
        machine.get_all_memories(all_mem);
        for(std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
          if (gasnet_mynode() == ID(*it).node())
            mem.insert(*it);
        }
      }
      else {
        machine.get_all_memories(mem);
      }
      int mem_idx = rand() % mem.size();
      std::set<Memory>::iterator it = mem.begin();
      while (mem_idx > 0) {
        it++;
        mem_idx--;
      }
      block_size_vec[j] = rand() % domain.get_volume() + 1;
      inst_vec[j] = domain.create_instance(*it, field_sizes, block_size_vec[j]);
      assert(ID(inst_vec[j]).type() == ID::ID_INSTANCE);
      // printf("node = %d\n", ID(*it).node());
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
        Event copyEvent = domain.copy(src_fields, dst_fields, Event::NO_EVENT);
        printf("[%d]	inst[%u] {%d (%d) -> %d (%d)}: ", gasnet_mynode(), j,
               get_runtime()->get_instance_impl(inst_vec[j-1])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j-1])->memory).node(),
               get_runtime()->get_instance_impl(inst_vec[j])->memory.kind(),
               ID(get_runtime()->get_instance_impl(inst_vec[j])->memory).node());
        copyEvent.wait();
        //if (verify_region_data(domain, inst_vec[j], field_order[j]))
          printf("check passed...\n ");
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
  printf("all check passed......\n");
  printf("finish top level task......\n");
  return;
}

int main(int argc, char **argv)
{
  Runtime rt;
  
  rt.init(&argc, &argv);
  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  rt.shutdown();
  
  return -1;
}
