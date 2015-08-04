#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"
#include "lowlevel_impl.h"
#include "channel.h"

#define DIM_X 100

using namespace LegionRuntime::LowLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

bool test_single_xferDes(Memory::Kind src_mem_kind, Memory::Kind dst_mem_kind)
{
  // create Domain that is to be mapped to HDF file
  Point<1> lo = make_point(0), hi = make_point(DIM_X - 1);
  Rect<1> rect(lo, hi);
  Domain domain = Domain::from_rect<1>(rect);

  // get HDF memory instance
  Machine machine = Machine::get_machine();
  std::set<Memory> mem;
  Memory src_mem = Memory::NO_MEMORY, dst_mem = Memory::NO_MEMORY;
  machine.get_all_memories(mem);
  for(std::set<Memory>::iterator it = mem.begin(); it != mem.end(); it++) {
    if (it->kind() == src_mem_kind) {
      src_mem = *it;
    }
    if (it->kind() == dst_mem_kind) {
      dst_mem = *it;
    }
  }
  assert(src_mem != Memory::NO_MEMORY);
  assert(dst_mem != Memory::NO_MEMORY);

  std::vector<size_t> field_sizes;
  field_sizes.push_back(sizeof(int));
  field_sizes.push_back(sizeof(int));
  field_sizes.push_back(sizeof(int));
  // create instance in mem
  RegionInstance src_inst = domain.create_instance(src_mem, field_sizes, 1);
  RegionInstance dst_inst = domain.create_instance(dst_mem, field_sizes, DIM_X);
  // create an accessor and read HDF file
  int read_data[3];
  RegionAccessor<AccessorType::Generic> src_acc = src_inst.get_accessor();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    int i = pir.p.x[0];
    read_data[0] = 1 * i;
    read_data[1] = 2 * i;
    read_data[2] = 3 * i;
    src_acc.write_untyped(DomainPoint::from_point<1>(pir.p), &read_data[0], sizeof(int), 0);
    src_acc.write_untyped(DomainPoint::from_point<1>(pir.p), &read_data[1], sizeof(int), sizeof(int));
    src_acc.write_untyped(DomainPoint::from_point<1>(pir.p), &read_data[2], sizeof(int), 2 * sizeof(int));
  }

  std::vector<Domain::CopySrcDstField> src_fields, dst_fields;
  for (int i = 0; i < 3; i++) {
    Domain::CopySrcDstField src_field(src_inst, i * sizeof(int), sizeof(int));
    Domain::CopySrcDstField dst_field(dst_inst, (2 - i) * sizeof(int), sizeof(int));
    src_fields.push_back(src_field);
    dst_fields.push_back(dst_field);
  }

  Event copyEvent = domain.copy(src_fields, dst_fields, Event::NO_EVENT);

  copyEvent.wait();

  RegionAccessor<AccessorType::Generic> dst_acc = dst_inst.get_accessor();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    int i = pir.p.x[0];
    dst_acc.read_untyped(DomainPoint::from_point<1>(pir.p), &read_data[0], sizeof(int), 0);
    dst_acc.read_untyped(DomainPoint::from_point<1>(pir.p), &read_data[1], sizeof(int), sizeof(int));
    dst_acc.read_untyped(DomainPoint::from_point<1>(pir.p), &read_data[2], sizeof(int), 2 * sizeof(int));
    assert(read_data[0] == 3 * i);
    assert(read_data[1] == 2 * i);
    assert(read_data[2] == 1 * i);
  }
  return true;
}

void top_level_task(const void *args, size_t arglen, Processor p)
{
  printf("top level task - test DMA functionality\n");
  if(test_single_xferDes(Memory::SYSTEM_MEM, Memory::SYSTEM_MEM))
    printf("Test cpu->cpu XferDes passed.\n");
  if(test_single_xferDes(Memory::SYSTEM_MEM, Memory::DISK_MEM))
    printf("Test cpu->disk XferDes passed.\n");
  if(test_single_xferDes(Memory::DISK_MEM, Memory::SYSTEM_MEM))
    printf("Test disk->cpu XferDes passed.\n");
  if(test_single_xferDes(Memory::SYSTEM_MEM, Memory::GPU_FB_MEM))
    printf("Test cpu->gpu XferDes passed.\n");
  if(test_single_xferDes(Memory::GPU_FB_MEM, Memory::SYSTEM_MEM))
    printf("Test gpu->cpu XferDes passed.\n");
  if(test_single_xferDes(Memory::DISK_MEM, Memory::DISK_MEM))
    printf("Test disk->mem->disk XferDes passed.\n");
  if(test_single_xferDes(Memory::DISK_MEM, Memory::GPU_FB_MEM))
    printf("Test disk->mem->gpu XferDes passed.\n");

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
